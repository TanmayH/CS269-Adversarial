from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
from torch.cuda import device_of
import torch.nn as nn
import clip
import torch.nn.functional as F
from collections import OrderedDict


def define_tsnet_clip(name, num_class,args, cuda=True):
  model, preprocess = clip.load("RN50")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(args.t_model, map_location=device)
  state_dict = checkpoint["state_dict"]
  if(next(iter(state_dict.items()))[0].startswith("module")):
      state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
  model.load_state_dict(state_dict)
    
  if name == 'clip':
    net = model.visual
  elif name == 'resnet20_clip':
    net = ModifiedResNet()
  else:
    raise Exception('model name does not exist.')

  if cuda:
    net = torch.nn.DataParallel(net).cuda()
  else:
    net = torch.nn.DataParallel(net)

  return net

def define_tsnet(name, num_class, cuda=True):
  if name == 'resnet20':
    net = resnet20(num_class=num_class)
  elif name == 'resnet110':
    net = resnet110(num_class=num_class)
  else:
    raise Exception('model name does not exist.')

  if cuda:
    net = torch.nn.DataParallel(net).cuda()
  else:
    net = torch.nn.DataParallel(net)

  return net


class resblock(nn.Module):
  def __init__(self, in_channels, out_channels, return_before_act):
    super(resblock, self).__init__()
    self.return_before_act = return_before_act
    self.downsample = (in_channels != out_channels)
    if self.downsample:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
      self.ds    = nn.Sequential(*[
              nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
              nn.BatchNorm2d(out_channels)
              ])
    else:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
      self.ds    = None
    self.bn1   = nn.BatchNorm2d(out_channels)
    self.relu  = nn.ReLU()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2   = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    residual = x

    pout = self.conv1(x) # pout: pre out before activation
    pout = self.bn1(pout)
    pout = self.relu(pout)

    pout = self.conv2(pout)
    pout = self.bn2(pout)

    if self.downsample:
      residual = self.ds(x)

    pout += residual
    out  = self.relu(pout)

    if not self.return_before_act:
      return out
    else:
      return pout, out


class resnet20(nn.Module):
	def __init__(self, num_class):
		super(resnet20, self).__init__()
		self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1     = nn.BatchNorm2d(16)
		self.relu    = nn.ReLU()

		self.res1 = self.make_layer(resblock, 3, 16, 16)
		self.res2 = self.make_layer(resblock, 3, 16, 32)
		self.res3 = self.make_layer(resblock, 3, 32, 64)

		self.avgpool = nn.AvgPool2d(8)
		self.fc      = nn.Linear(64, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.num_class = num_class

	def make_layer(self, block, num, in_channels, out_channels): # num must >=2
		layers = [block(in_channels, out_channels, False)]
		for i in range(num-2):
			layers.append(block(out_channels, out_channels, False))
		layers.append(block(out_channels, out_channels, True))
		return nn.Sequential(*layers)

	def forward(self, x):
		pstem = self.conv1(x) # pstem: pre stem before activation
		pstem = self.bn1(pstem)
		stem  = self.relu(pstem)
		stem  = (pstem, stem)

		rb1 = self.res1(stem[1])
		rb2 = self.res2(rb1[1])
		rb3 = self.res3(rb2[1])

		feat = self.avgpool(rb3[1])
		feat = feat.view(feat.size(0), -1)
		out  = self.fc(feat)

		return stem, rb1, rb2, rb3, feat, out

	def get_channel_num(self):
		return [16, 16, 32, 64, 64, self.num_class]

	def get_chw_num(self):
		return [(16, 32, 32),
				(16, 32, 32),
				(32, 16, 16),
				(64, 8 , 8 ),
				(64,),
				(self.num_class,)]


class resnet110(nn.Module):
  def __init__(self, num_class):
    super(resnet110, self).__init__()
    self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1     = nn.BatchNorm2d(16)
    self.relu    = nn.ReLU()

    self.res1 = self.make_layer(resblock, 18, 16, 16)
    self.res2 = self.make_layer(resblock, 18, 16, 32)
    self.res3 = self.make_layer(resblock, 18, 32, 64)

    self.avgpool = nn.AvgPool2d(8)
    self.fc      = nn.Linear(64, num_class)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    self.num_class = num_class

  def make_layer(self, block, num, in_channels, out_channels):  # num must >=2
    layers = [block(in_channels, out_channels, False)]
    for i in range(num-2):
      layers.append(block(out_channels, out_channels, False))
    layers.append(block(out_channels, out_channels, True))
    return nn.Sequential(*layers)

  def forward(self, x):
    pstem = self.conv1(x) # pstem: pre stem before activation
    pstem = self.bn1(pstem)
    stem  = self.relu(pstem)
    stem  = (pstem, stem)

    rb1 = self.res1(stem[1])
    rb2 = self.res2(rb1[1])
    rb3 = self.res3(rb2[1])

    feat = self.avgpool(rb3[1])
    feat = feat.view(feat.size(0), -1)
    out  = self.fc(feat)

    return stem, rb1, rb2, rb3, feat, out

  def get_channel_num(self):
    return [16, 16, 32, 64, 64, self.num_class]

  def get_chw_num(self):
    return [(16, 32, 32),
        (16, 32, 32),
        (32, 16, 16),
        (64, 8 , 8 ),
        (64,),
        (self.num_class,)]


class resnet20_clip(nn.Module):
  def __init__(self, num_class):
    super(resnet20_clip, self).__init__()
    self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn1     = nn.BatchNorm2d(64)
    self.relu    = nn.ReLU()

    self.res1 = self.make_layer(resblock, 3, 64, 256)
    self.res2 = self.make_layer(resblock, 3, 256, 512)
    self.res3 = self.make_layer(resblock, 3, 512, 1024)
    self.res4 = self.make_layer(resblock, 3, 1024, 2048) #Adding res4 to match cyclip image encoder 4 res structure


    self.avgpool = nn.AvgPool2d(4)
    self.fc      = nn.Linear(2048, 1024) #last layer is an embedding

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    self.num_class = num_class

  def make_layer(self, block, num, in_channels, out_channels):  # num must >=2
    mid_channels = out_channels
    if out_channels > 64:
      mid_channels = out_channels//4
    layers = [block(in_channels, mid_channels, False)]
    # for i in range(num-2):
    #   layers.append(block(mid_channels, mid_channels, False))
    layers.append(block(mid_channels, out_channels, True))
    return nn.Sequential(*layers)

  def forward(self, x):
    pstem = self.conv1(x) # pstem: pre stem before activation
    pstem = self.bn1(pstem)
    stem  = self.relu(pstem)
    stem  = (pstem, stem)

    rb1 = self.res1(stem[1])
    rb2 = self.res2(rb1[1])
    rb3 = self.res3(rb2[1])
    rb4 = self.res4(rb3[1])


    feat = self.avgpool(rb4[1])
    feat = feat.view(feat.size(0), -1)
    out  = self.fc(feat)

    return out.to(dtype=torch.float16) #Use a different function to return other parameters instead

  def get_channel_num(self):
    return [16, 16, 32, 64, 64, self.num_class]

  def get_chw_num(self):
    return [(16, 32, 32),
        (16, 32, 32),
        (32, 16, 16),
        (64, 8 , 8 ),
        (64,),
        (self.num_class,)]

def define_paraphraser(in_channels_t, k, use_bn, cuda=True):
	net = paraphraser(in_channels_t, k, use_bn)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	else:
		net = torch.nn.DataParallel(net)

	return net


class paraphraser(nn.Module):
	def __init__(self, in_channels_t, k, use_bn=True):
		super(paraphraser, self).__init__()
		factor_channels = int(in_channels_t*k)
		self.encoder = nn.Sequential(*[
				nn.Conv2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(in_channels_t, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])
		self.decoder = nn.Sequential(*[
				nn.ConvTranspose2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.ConvTranspose2d(factor_channels, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.ConvTranspose2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		out = self.decoder(z)
		return z, out


def define_translator(in_channels_s, in_channels_t, k, use_bn=True, cuda=True):
	net = translator(in_channels_s, in_channels_t, k, use_bn)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	else:
		net = torch.nn.DataParallel(net)

	return net


class translator(nn.Module):
	def __init__(self, in_channels_s, in_channels_t, k, use_bn=True):
		super(translator, self).__init__()
		factor_channels = int(in_channels_t*k)
		self.encoder = nn.Sequential(*[
				nn.Conv2d(in_channels_s, in_channels_s, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(in_channels_s, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		return z


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
