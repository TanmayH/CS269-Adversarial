# import os
# os.environ["WANDB_SILENT"] = "true"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

# import sys
# import time
# import wandb
# import torch
# import logging
# import warnings
# import numpy as np
# import tensorflow as tf
# import torch.optim as optim
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.backends.cudnn as cudnn
# from torch.cuda.amp import GradScaler
# from torch.nn.parallel import DistributedDataParallel as DDP

# from pkgs.openai.clip import load as load_model

# from .train import train
# from .evaluate import evaluate
# from .data import load as load_data
# from .parser import parse_args
# from .scheduler import cosine_scheduler
# from .logger import get_logger, set_logger

# mp.set_start_method("spawn", force = True)
# warnings.filterwarnings("ignore")

# def worker(rank, options, logger):
#     options.rank = rank
#     options.master = rank == 0
    
#     set_logger(rank = rank, logger = logger, distributed = options.distributed)

#     if(options.device == "cuda"):
#         options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

#     logging.info(f"Using {options.device} device")

#     if(options.master):
#         logging.info("Params:")
#         with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
#             for key in sorted(vars(options)):
#                 value = getattr(options, key)
#                 logging.info(f"{key}: {value}")
#                 file.write(f"{key}: {value}\n")

#     if(options.distributed):
#         dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
#     options.batch_size = options.batch_size // options.num_devices

#     model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

#     if(options.device == "cpu"):
#         model.float()
#     else:
#         torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
#         model.to(options.device)
#         if(options.distributed):
#             model = DDP(model, device_ids = [options.device_ids[options.rank]])
        
#     data = load_data(options, processor)

#     optimizer = None
#     scheduler = None
#     if(data["train"] is not None):        
#         weight_decay_parameters = []
#         no_weight_decay_parameters = []

#         for name, parameter in model.named_parameters():
#             if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
#                 weight_decay_parameters.append(parameter)
                
#             if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
#                 no_weight_decay_parameters.append(parameter)

#         optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
#         scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, data["train"].num_batches * options.epochs)

#     start_epoch = 0
#     if(options.checkpoint is not None):
#         if(os.path.isfile(options.checkpoint)):
#             checkpoint = torch.load(options.checkpoint, map_location = options.device)
#             start_epoch = checkpoint["epoch"]
#             state_dict = checkpoint["state_dict"]
#             if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
#                 state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
#             model.load_state_dict(state_dict)
#             if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
#             logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
#         else:
#             logging.info(f"No checkpoint found at {options.checkpoint}")

#     cudnn.benchmark = True
#     cudnn.deterministic = False

#     if(options.wandb and options.master):
#         logging.debug("Starting wandb")
#         wandb.init(project = "mrl", notes = options.notes, tags = [], config = vars(options))
#         wandb.run.name = options.name
#         wandb.save(os.path.join(options.log_dir_path, "params.txt"))

#     evaluate(start_epoch, model, processor, data, options)

#     if(data["train"] is not None):
#         options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
#         os.makedirs(options.checkpoints_dir_path, exist_ok = True)

#         scaler = GradScaler()

#         best_loss = np.inf
#         for epoch in range(start_epoch + 1, options.epochs + 1):
#             if(options.master): 
#                 logging.info(f"Starting Epoch {epoch}")

#             start = time.time()
#             train(epoch, model, data, optimizer, scheduler, scaler, options)
#             end = time.time()

#             if(options.master): 
#                 logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

#             metrics = evaluate(epoch, model, processor, data, options)

#             if(options.master):
#                 checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
#                 torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
#                 if("loss" in metrics):
#                     if(metrics["loss"] < best_loss):
#                         best_loss = metrics["loss"]
#                         torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))

#     if(options.distributed):
#         dist.destroy_process_group()

#     if(options.wandb and options.master):
#         wandb.finish()

# if(__name__ == "__main__"):    
#     options = parse_args()

#     options.log_dir_path = os.path.join(options.logs, options.name)
#     options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
#     os.makedirs(options.log_dir_path, exist_ok = True)
#     logger, listener = get_logger(options.log_file_path)

#     listener.start()

#     ngpus = torch.cuda.device_count()
#     if(ngpus == 0 or options.device == "cpu"):
#         options.device = "cpu"
#         options.num_devices = 1
#         options.distributed = False
#         worker(0, options, logger)
#     else:
#         if(ngpus == 1 or not options.distributed):
#             options.device = "cuda"
#             options.num_devices = 1
#             options.distributed = False
#             worker(0, options, logger)
#         else:
#             options.device = "cuda"
#             if(options.device_ids is None):
#                 options.device_ids = list(range(ngpus))
#                 options.num_devices = ngpus
#             else:
#                 options.device_ids = list(map(int, options.device_ids))
#                 options.num_devices = len(options.device_ids)
#             options.distributed = True
#             os.environ["NCCL_P2P_DISABLE"] = "1"
#             mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
#     listener.stop()
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import sys
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import tensorflow as tf
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# print("path 1 = ")
# print(sys.path)
# sys.path.insert(1, '../')
# print("path 2 = ")
# print(sys.path)
from pkgs.openai.clip import load as load_model
# sys.path.insert(1, 'src')
from .train import train
from .evaluate import evaluate
from .data import load as load_data
from .parser import parse_args
from .scheduler import cosine_scheduler
from .logger import get_logger, set_logger

import torch
from torch.cuda import device_of
import torch.nn as nn
import clip
import torch.nn.functional as F
from collections import OrderedDict
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers=(3, 3, 3, 3), output_dim=1024, heads=64, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

def worker(rank, options, logger):
    options.rank = rank
    options.master = rank == 0
    
    set_logger(rank = rank, logger = logger, distributed = options.distributed)

    if(options.device == "cuda"):
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if(options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
    options.batch_size = options.batch_size // options.num_devices

    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [options.device_ids[options.rank]])
        
    data = load_data(options, processor)
    
    optimizer = None
    scheduler = None
    if(data["train"] is not None):        
        weight_decay_parameters = []
        no_weight_decay_parameters = []

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)

        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
        scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, data["train"].num_batches * options.epochs)

    start_epoch = 0
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = options.device)
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")
    
    #Assigning student_model to teacher
    if (options.student_checkpoint is not None):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      student_model = ModifiedResNet().to(device)
      checkpoint = torch.load(options.student_checkpoint, map_location=device)
      state_dict = checkpoint["snet"]
      if(next(iter(state_dict.items()))[0].startswith("module")):
          state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
      student_model.load_state_dict(state_dict)
      model.visual=student_model

    cudnn.benchmark = True
    cudnn.deterministic = False

    if(options.wandb and options.master):
        logging.debug("Starting wandb")
        wandb.init(project = "mrl", notes = options.notes, tags = [], config = vars(options))
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))

    evaluate(start_epoch, model, processor, data, options)

    if(data["train"] is not None):
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok = True)

        scaler = GradScaler()

        best_loss = np.inf
        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master): 
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            train(epoch, model, data, optimizer, scheduler, scaler, options)
            end = time.time()

            if(options.master): 
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

            metrics = evaluate(epoch, model, processor, data, options)

            if(options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                if("loss" in metrics):
                    if(metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))

    if(options.distributed):
        dist.destroy_process_group()

    if(options.wandb and options.master):
        wandb.finish()

if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()
