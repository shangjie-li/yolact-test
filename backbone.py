import torch
import torch.nn as nn
import pickle

from collections import OrderedDict

class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1, use_dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        if use_dcn:
            #~ self.conv2 = DCN(planes, planes, kernel_size=3, stride=stride,
                                #~ padding=dilation, dilation=dilation, deformable_groups=1)
            #~ self.conv2.bias.data.zero_()
            #~ self.conv2.conv_offset_mask.weight.data.zero_()
            #~ self.conv2.conv_offset_mask.bias.data.zero_()
            raise NotImplementedError
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=dilation, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, dcn_layers=[0, 0, 0, 0], dcn_interval=1, atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self._make_layer(block, 64, layers[0], dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 128, layers[1], stride=2, dcn_layers=dcn_layers[1], dcn_interval=dcn_interval)
        self._make_layer(block, 256, layers[2], stride=2, dcn_layers=dcn_layers[2], dcn_interval=dcn_interval)
        self._make_layer(block, 512, layers[3], stride=2, dcn_layers=dcn_layers[3], dcn_interval=dcn_interval)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    
    def _make_layer(self, block, planes, blocks, stride=1, dcn_layers=0, dcn_interval=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          dilation=self.dilation),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        use_dcn = (dcn_layers >= blocks)
        layers.append(block(self.inplanes, planes, stride, downsample, self.norm_layer, self.dilation, use_dcn=use_dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_dcn = ((i+dcn_layers) >= blocks) and (i % dcn_interval == 0)
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer, use_dcn=use_dcn))
        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x):
        """ Returns a list of convouts for each layer. """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)


class Channel_Attention(nn.Module):
    def __init__(self, channel, r=16):
        super(Channel_Attention, self).__init__()
        
        self._avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self._fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // r, channel, kernel_size=1, bias=False),
        )
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self._avg_pool(x)
        y = self._fc(y)
        y = self._sigmoid(y)
        return x * y


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial_Attention, self).__init__()
        
        assert kernel_size % 2 == 1, 'kernel_size = {}'.format(kernel_size)
        padding = (kernel_size - 1) // 2
        
        self._layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        mask, _ = torch.max(x, dim=1, keepdim=True) # The dimension of x should be [batch, channel, h, w]
        mask = self._layer(mask)
        return x * mask


def conv3x3BNReLU(in_channels, out_channels, stride, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class OSA(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_block=3, reduction=16):
        super(OSA, self).__init__()
        
        self._layers = nn.ModuleList()
        self._layers.append(conv3x3BNReLU(in_channels, mid_channels, stride=1))
        for idx in range(num_block - 1):
            self._layers.append(conv3x3BNReLU(mid_channels, mid_channels, stride=1))
            
        self.conv1x1 = conv1x1BNReLU(in_channels + mid_channels * num_block, out_channels)
        # self.ca_layer = Channel_Attention(out_channels, reduction)
        # self.sa_layer = Spatial_Attention()
        
    def forward(self, x):
        features = []
        features.append(x)
        for _layer in self._layers:
            x = _layer(x)
            features.append(x)
        out = torch.cat(features, dim=1)
        out = self.conv1x1(out)
        # out = self.ca_layer(out)
        # out = self.sa_layer(out)
        return out


class VovNetBackbone(nn.Module):
    def __init__(self, layers, num_classes=3):
        super(VovNetBackbone, self).__init__()
        planes = [[128, 64, 128],
                  [128, 80, 256],
                  [256, 96, 384],
                  [384, 112, 512]]
        
        self.channels = [128, 256, 384, 512]
        self.groups = 1
        
        self.stage1 = nn.Sequential(
            conv3x3BNReLU(3, 64, stride=2, groups=self.groups),
            conv3x3BNReLU(64, 64, stride=1, groups=self.groups),
            conv3x3BNReLU(64, 128, stride=1, groups=self.groups),
        )
        self.stage2 = self._make_layer(planes[0][0], planes[0][1], planes[0][2], layers[0])
        self.stage3 = self._make_layer(planes[1][0], planes[1][1], planes[1][2], layers[1])
        self.stage4 = self._make_layer(planes[2][0], planes[2][1], planes[2][2], layers[2])
        self.stage5 = self._make_layer(planes[3][0], planes[3][1], planes[3][2], layers[3])
        
    def _make_layer(self, in_channels, mid_channels, out_channels, num_block):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for idx in range(num_block):
            layers.append(OSA(in_channels, mid_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.stage1(x)
        outs = []
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)
        x = self.stage4(x)
        outs.append(x)
        x = self.stage5(x)
        outs.append(x)
        
        return outs


def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    backbone = cfg.type(*cfg.args)

    if cfg.name == 'ResNet101' or cfg.name == 'ResNet50':
        # Add downsampling layers until we reach the number we need
        num_layers = max(cfg.selected_layers) + 1

        while len(backbone.layers) < num_layers:
            backbone.add_layer()

    return backbone
