from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
from itertools import chain
import glob


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst
from torch.autograd import Variable

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet_clip
from kd_losses import *
import clip
import math
import glob
from torch.utils.data import Dataset
from PIL import Image

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--img_root_valid', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--train_ds_size', type=int, default=5000, help='Size of training dataset')
parser.add_argument('--valid_ds_size', type=int, default=300, help='Size of validation dataset')

# Student model - Resnet 110 without last layer, try it out - DONT SEE THE NEED SO SKIPPING FOR NOW
# TODO: Assign Resnet110 pretrained on COCO? - Need image classes for that. Then Assign layers except last
# parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')

# Teacher model - Pass path of model weights for poisoned CyCLIP model
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=8, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')    # resnet20/resnet110
parser.add_argument('--s_name', type=str, required=True, help='name of student')    # resnet20/resnet110

# hyperparameter
parser.add_argument('--kd_mode', type=str, required=True, help='mode of kd, which can be:'
															   'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
															   'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--w_dist', type=float, default=25.0, help='weight for RKD distance')
parser.add_argument('--w_angle', type=float, default=50.0, help='weight for RKD angle')
parser.add_argument('--m', type=float, default=2.0, help='margin for AB')
parser.add_argument('--gamma', type=float, default=0.4, help='gamma in Gaussian RBF for CC')
parser.add_argument('--P_order', type=int, default=2, help='P-order Taylor series of Gaussian RBF for CC')
parser.add_argument('--w_irg_vert', type=float, default=0.1, help='weight for IRG vertex')
parser.add_argument('--w_irg_edge', type=float, default=5.0, help='weight for IRG edge')
parser.add_argument('--w_irg_tran', type=float, default=5.0, help='weight for IRG transformation')
parser.add_argument('--sf', type=float, default=1.0, help='scale factor for VID, i.e. mid_channels = sf * out_channels')
parser.add_argument('--init_var', type=float, default=5.0, help='initial variance for VID')
parser.add_argument('--att_f', type=float, default=1.0, help='attention factor of mid_channels for AFD')


args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


#Class declarations
class SampleSimilarities(nn.Module):

    def __init__(self, feats_dim, queueSize, T):
        super(SampleSimilarities, self).__init__()
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, feats_dim))

    def forward(self, q, update=True):
        batchSize = q.shape[0]
        queue = self.memory.clone()
        # print(queue.shape)
        # print(q.shape)
        out = torch.mm(queue.detach(), q.transpose(1, 0))
        # out = torch.mm(queue.detach(), q.view(queue.shape[-1],-1))
        out = out.transpose(0, 1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        if update:
            # update memory bank
            with torch.no_grad():
                out_ids = torch.arange(batchSize).cuda()
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, q)
                self.index = (self.index + batchSize) % self.queueSize

        return out

class CompReSS(nn.Module):

    def __init__(self , teacher_feats_dim, student_feats_dim, kld, queue_size=128000, T=0.04):
        super(CompReSS, self).__init__()

        self.l2norm = Normalize(2).cuda()
        self.criterion = kld
        self.student_sample_similarities = SampleSimilarities(student_feats_dim , queue_size , T).cuda()
        self.teacher_sample_similarities = SampleSimilarities(teacher_feats_dim , queue_size , T).cuda()

    def forward(self, teacher_feats, student_feats):

        teacher_feats = self.l2norm(teacher_feats).float()
        student_feats = self.l2norm(student_feats).float()

        similarities_teacher = self.teacher_sample_similarities(teacher_feats)
        similarities_student = self.student_sample_similarities(student_feats)

        loss = self.criterion(similarities_teacher , similarities_student).cuda()
        return loss.to(dtype=torch.float16)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class CustomDataset(Dataset):
  def __init__(self, image_path, processor,size):
      self.imgs_path = image_path
      self.processor = processor
      file_list = glob.glob(self.imgs_path + "*")
      self.data = []
      count=0
      for img_path in glob.glob(self.imgs_path  + "/*.png"):
          self.data.append(img_path)
          count+=1
          if (count>size):
            break
  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      img_path = self.data[idx]
      return self.processor(Image.open(img_path))

def main():
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
  logging.info("args = %s", args)
  logging.info("unparsed_args = %s", unparsed)

  logging.info('----------- Network Initialization --------------')
  snet = define_tsnet_clip(name=args.s_name, num_class=args.num_class, args=args, cuda=args.cuda)
  #Note: Don't see the point of loading initial weights for student so skipping that
  logging.info('Student param size = %fMB', count_parameters_in_MB(snet))

  tnet = define_tsnet_clip(name=args.t_name, num_class=args.num_class,args=args, cuda=args.cuda)
  tnet.eval()
  for param in tnet.parameters():
    param.requires_grad = False
  
  #Adding activation hooks for intermediate resblock layers
  activation = {}
  def get_activation(name, indexed=False):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #Adding student activations
  snet.module.layer1.register_forward_hook(get_activation('rb1_s',True))
  snet.module.layer2.register_forward_hook(get_activation('rb2_s',True))
  snet.module.layer3.register_forward_hook(get_activation('rb3_s',True))
  snet.module.layer4.register_forward_hook(get_activation('rb4_s',True))

  #Adding teacher activations
  tnet.module.layer1.register_forward_hook(get_activation('rb1_t'))
  tnet.module.layer2.register_forward_hook(get_activation('rb2_t'))
  tnet.module.layer3.register_forward_hook(get_activation('rb3_t'))
  tnet.module.layer4.register_forward_hook(get_activation('rb4_t'))
  
  # logging.info('Teacher: %s', tnet)
  logging.info('Teacher param size = %fMB', count_parameters_in_MB(tnet))
  logging.info('-----------------------------------------------')

  # define loss functions
  if args.kd_mode == 'logits':
    criterionKD = Logits()
  elif args.kd_mode == 'st':
    criterionKD = SoftTarget(args.T)
  elif args.kd_mode == 'at':
    criterionKD = AT(args.p)
  elif args.kd_mode == 'fitnet':
    criterionKD = Hint()
  elif args.kd_mode == 'nst':
    criterionKD = NST()
  else:
    raise Exception('Invalid kd mode...')
  if args.cuda:
    criterionCls = torch.nn.CrossEntropyLoss().cuda()
  else:
    criterionCls = torch.nn.CrossEntropyLoss()
  
  tmp_input = torch.randn(2, 3, 224, 224)
  # print(snet)
  # print(tnet)
  feat_t = tnet(tmp_input)
  feat_s = snet(tmp_input)
  student_feats_dim = feat_s.shape[-1]
  teacher_feats_dim = feat_t.shape[-1]
  #Compress object for logic
  compress = CompReSS(teacher_feats_dim, student_feats_dim,criterionKD, 128000, 0.04)


  # initialize optimizer
  if args.kd_mode in ['vid', 'ofd', 'afd']:
    optimizer = torch.optim.SGD(chain(snet.parameters(), 
                      *[c.parameters() for c in criterionKD[1:]]),
                  lr = args.lr, 
                  momentum = args.momentum, 
                  weight_decay = args.weight_decay,
                  nesterov = True)
  else:
    optimizer = torch.optim.SGD(snet.parameters(),
                  lr = args.lr, 
                  momentum = args.momentum, 
                  weight_decay = args.weight_decay,
                  nesterov = True)

  # define transforms
  if args.data_name == 'coco':
    image_path = os.path.join(args.img_root)
    print(image_path)
    _, processor = clip.load("RN50")
    train_ds = CustomDataset(image_path=image_path, processor=processor, size=args.train_ds_size)
    image_path = os.path.join(args.img_root_valid)
    valid_ds = CustomDataset(image_path=image_path, processor=processor, size=args.valid_ds_size)    
  else:
    raise Exception('Invalid dataset name...')

  # define data loader
  train_loader = torch.utils.data.DataLoader(
      train_ds,
      batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(
      valid_ds,
      batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

  print("Train loader size: ",len(train_loader))
  print("Test loader size: ",len(test_loader))
  # warp nets and criterions for train and test
  nets = {'snet':snet, 'tnet':tnet}
  criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD}

  best_top1 = 0
  best_top5 = 0

  for epoch in range(1, args.epochs+1):
    adjust_lr(optimizer, epoch)

    # train one epoch
    epoch_start_time = time.time()
    train(train_loader, nets, optimizer, criterions, epoch, compress, activation, student_feats_dim, teacher_feats_dim)

    # evaluate on testing set
    logging.info('Testing the models......')
    test_top1, test_top5 = test(test_loader, nets, criterions, epoch, compress, activation)

    epoch_duration = time.time() - epoch_start_time
    logging.info('Epoch time: {}s'.format(int(epoch_duration)))

    # save model
    is_best = False
    if test_top1 > best_top1:
      best_top1 = test_top1
      best_top5 = test_top5
      is_best = True
    logging.info('Saving models......')
    save_checkpoint({
      'epoch': epoch,
      'snet': snet.state_dict(),
      'tnet': tnet.state_dict(),
      'prec@1': test_top1,
      'prec@5': test_top5,
    }, is_best, args.save_root)
  
  #THIS SECTION JUST TO SEE FINAL OUTPUT AFTER MODEL TRAINING, PURELY LOGGINg
  #Helpful debug logs
  model_pt,pp1 = clip.load("RN50")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(args.t_model, map_location=device)
  state_dict = checkpoint["state_dict"]
  if(next(iter(state_dict.items()))[0].startswith("module")):
      state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
  model_pt.load_state_dict(state_dict)

  tmp_input = torch.randn(2, 3, 224, 224).to(torch.device("cuda"))
  #Print statements help ensure the model is actually transferred
  print("Original model output:")
  feat_t = model_pt.visual(tmp_input)
  print(feat_t) 
  print(feat_t.shape)
  
  model_pt.visual = snet
  feat_t2 = model_pt.visual(tmp_input)
  print("Model with student as visual encoder as output:")
  print(feat_t2)
  print(feat_t2.shape)

  print("Confirming with student model as output:")
  feat_t3 = snet(tmp_input)
  print(feat_t3)
  print(feat_t3.shape)
  ##
  

def train(train_loader, nets, optimizer, criterions, epoch, compress, activation, student_feats_dim, teacher_feats_dim):
  batch_time = AverageMeter()
  data_time  = AverageMeter()
  cls_losses = AverageMeter()
  kd_losses  = AverageMeter()
  top1       = AverageMeter()
  top5       = AverageMeter()

  snet = nets['snet']
  tnet = nets['tnet']

  criterionCls = criterions['criterionCls']
  criterionKD  = criterions['criterionKD']

  snet.train()
  if args.kd_mode in ['vid', 'ofd']:
    for i in range(1,4):
      criterionKD[i].train()

  end = time.time()
  for i, img in enumerate(train_loader, start=1): 
    data_time.update(time.time() - end)

    if args.cuda:
      img = img.cuda(non_blocking=True)
    
    out_s = snet(img)
    out_t = tnet(img)

    if args.kd_mode in ['logits', 'st']:
      kd_loss = compress(out_s, out_t.detach())*args.lambda_kd
    elif args.kd_mode in ['fitnet', 'nst']: #existing kd loss + same kd loss in logits/st
      kd_loss = Variable(criterionKD(activation["rb4_s"],activation["rb4_t"].detach())* args.lambda_kd, requires_grad=True)
    elif args.kd_mode in ['at']:
      kd_loss = (criterionKD(activation["rb1_s"],activation["rb1_t"].detach()) +
              criterionKD(activation["rb2_s"],activation["rb2_t"].detach()) +
              criterionKD(activation["rb3_s"],activation["rb3_t"].detach()) +
              criterionKD(activation["rb4_s"],activation["rb4_t"].detach())) / 4.0 * args.lambda_kd
      kd_loss = Variable(kd_loss, requires_grad=True)
    else:
      raise Exception('Invalid kd mode...')
    
    loss = kd_loss 

    kd_losses.update(kd_loss.item(), img.size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
              'Time:{batch_time.val:.4f} '
              'Data:{data_time.val:.4f}  '
              'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '.format(
              epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, kd_losses=kd_losses))
      logging.info(log_str)


def test(test_loader, nets, criterions, epoch, compress, activation):
  cls_losses = AverageMeter()
  kd_losses  = AverageMeter()
  top1       = AverageMeter()
  top5       = AverageMeter()

  snet = nets['snet']
  tnet = nets['tnet']

  criterionCls = criterions['criterionCls']
  criterionKD  = criterions['criterionKD']

  snet.eval()
  if args.kd_mode in ['vid', 'ofd']:
    for i in range(1,4):
      criterionKD[i].eval()

  end = time.time()
  for i, img in enumerate(test_loader, start=1):
    if args.cuda:
      img = img.cuda(non_blocking=True)

    with torch.no_grad():
      out_s = snet(img)
      out_t = tnet(img)
    if args.kd_mode in ['logits', 'st']:
      kd_loss = compress(out_s, out_t.detach())*args.lambda_kd
    elif args.kd_mode in ['fitnet', 'nst']:
      kd_loss = criterionKD(activation["rb4_s"],activation["rb4_t"].detach())* args.lambda_kd
    elif args.kd_mode in ['at']:
      kd_loss = (criterionKD(activation["rb1_s"],activation["rb1_t"].detach()) +
              criterionKD(activation["rb2_s"],activation["rb2_t"].detach()) +
              criterionKD(activation["rb3_s"],activation["rb3_t"].detach()) +
              criterionKD(activation["rb4_s"],activation["rb4_t"].detach())) / 4.0 * args.lambda_kd
    else:
      raise Exception('Invalid kd mode...')
    kd_losses.update(kd_loss.item(), img.size(0))

  logging.info('KD: {:.4f}'.format(kd_losses.avg))
  return top1.avg, top5.avg


def adjust_lr_init(optimizer, epoch):
	scale   = 0.1
	lr_list = [args.lr*scale] * 30
	lr_list += [args.lr*scale*scale] * 10
	lr_list += [args.lr*scale*scale*scale] * 10

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()