import mindspore.nn as nn
import mindspore.ops as P
from src.yolact.layers.dcn_v2 import DeformConv2d

class Bottleneck(nn.Cell):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1,
                 use_dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False, dilation=dilation, pad_mode='pad')
        self.bn1 = norm_layer(planes, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        if use_dcn== True:
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, stride=stride)
            # self.conv2 = DCN(planes, planes, kernel_size=3, stride=stride,
            #                  padding=dilation, pad_mode='pad', dilation=dilation, deformable_groups=1)
            # self.conv2.bias.data.zero_()
            # self.conv2.conv_offset_mask.weight.data.zero_()
            # self.conv2.conv_offset_mask.bias.data.zero_()
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,pad_mode='pad',padding=dilation, has_bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False, dilation=dilation, pad_mode='pad')
        self.bn3 = norm_layer(planes * 4, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
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


class ResNetBackbone(nn.Cell):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, dcn_layers=[0, 0, 0, 0], dcn_interval=1, atrous_layers=[], block=Bottleneck,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.CellList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers
        self.print = P.Print()
        # From torchvision.models.resnet.Resnet
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3,3,3), pad_mode='pad', has_bias=False)
        self.bn1 = norm_layer(64, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self._make_layer(block, 64, layers[0], dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 128, layers[1], stride=2, dcn_layers=dcn_layers[1], dcn_interval=dcn_interval)
        self._make_layer(block, 256, layers[2], stride=2, dcn_layers=dcn_layers[2], dcn_interval=dcn_interval)
        self._make_layer(block, 512, layers[3], stride=2, dcn_layers=dcn_layers[3], dcn_interval=dcn_interval)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_cells = [m for m in self.cells() if isinstance(m, nn.Conv2d)]
        # self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    # 生成一个stage
    def _make_layer(self, block, planes, blocks, stride=1, dcn_layers=0, dcn_interval=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1

            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False,
                          dilation=self.dilation, pad_mode='pad'),
                self.norm_layer(planes * block.expansion, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True),
            )

        layers = []
        use_dcn = (dcn_layers >= blocks)
        # 生成stage的一个block
        layers.append(block(self.inplanes, planes, stride, downsample, self.norm_layer, self.dilation, use_dcn=use_dcn))
        self.inplanes = planes * block.expansion
        # 生成stage的其它block
        for i in range(1, blocks):
            use_dcn = ((i + dcn_layers) >= blocks) and (i % dcn_interval == 0)
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer, use_dcn=use_dcn))
        layer = nn.SequentialCell(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    # 1 3 550 550
    def construct(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        # 1 64 275 275
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 1,64,138,138
        outs = ()
        # 1,256,138,138
        # 1,512,69,69
        # 1,1024,35,35
        # 1,2048,18,18
        for layer in self.layers:
            x = layer(x)
            outs += (x,)
        return outs

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)



def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    # 传进来的是cfg.backbone
   # backbone = cfg.type(*cfg.arg1, *cfg.arg2)
    # backbone = cfg.type(*cfg.args)
    backbone = cfg['type'](*cfg['args'])
    # backbone = ResNetBackbone([3, 4, 6, 3], [0, 4, 6, 3])
    # Add downsampling layers until we reach the number we need
    # num_layers = max(cfg.selected_layers) + 1
    num_layers = max(cfg['selected_layers']) + 1
    # num_layers = max(list(range(1, 4))) + 1

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone

# 单独测试使用
# def construct_backbone():
#     """ Constructs a backbone given a backbone config object (see config.py). """
#     # 传进来的是cfg.backbone
#     # backbone = cfg.type(*cfg.args)
#     # backbone = cfg.type(*cfg.backbone.args = [3, 4, 23, 3])
#     backbone = ResNetBackbone([3, 4, 6, 3], [0, 4, 6, 3])
#     # Add downsampling layers until we reach the number we need
#     # num_layers = max(cfg.selected_layers) + 1
#     num_layers = max(list(range(1, 4))) + 1
#
#     while len(backbone.layers) < num_layers:
#         backbone.add_layer()
#
#     return backbone
#
#
# # 测试代码
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')
#
#

# import os
# import numpy as np
# from mindspore import context, Tensor as tensor

# if __name__ == '__main__':
#     # b = mindspore.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],mindspore.float32)
#     device_id = int(os.getenv('DEVICE_ID', default=3))
#     # GRAPH_MODE
#     context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=device_id)
#     # backbone = construct_backbone()
#     img = tensor(np.random.randint(0, 255, (1, 3, 550, 550)).astype(np.float32))
#     net = ResNetBackbone(layers = [3, 4, 6, 3],  dcn_layers=[0, 4, 6, 3], dcn_interval=1, atrous_layers=[], block=Bottleneck,
#                  norm_layer=nn.BatchNorm2d)
#
#     print(net)
#     print(net(img))
