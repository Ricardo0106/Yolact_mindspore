import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # ----------------------iou-----------------------
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
        self.conv6 = nn.Conv2d(128, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()

        # --------------------backbone---------------------
        # conv1
        self.backboneconv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3, 3, 3), bias=False)

        self.bn = nn.BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # bottleneck
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(1,1), bias=False, dilation=(1,1)) # conv33
        self.conv22 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1),
                                                          bias=False, dilation=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False,dilation=(1,1))

        # fpn 分支
        self.conv8 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv9 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1,1))
        self.conv00 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2),padding=(1,1))

        # ---------------------proto-----------------
        self.conv01 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv02 = nn.Conv2d(256, 32, kernel_size=(1, 1))
        self.inter = F.interpolate(scale_factor=2, mode='bilinear', align_corners=False) # 无法转换
        # ---------------------seg-------------------
        self.conv_seg = nn.Conv2d(256, 80, kernel_size=(1, 1))




    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)


if __name__ == '__main__':

    net = Net()
    x = torch.tensor(np.random.rand(6, 1, 138, 138).astype(np.float32))
    y = net(x)
    print(y)   # 159.95244
# class FastMaskIoUNet(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         input_channels = 1
#         last_layer = [(80, 1, {})]
#         self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)
#
#     def forward(self, x):
#         x = self.maskiou_net(x)
#         maskiou_p = F.max_pool2d(kernel_size=(138,138)).squeeze(-1).squeeze(-1)
#
#         return maskiou_p


