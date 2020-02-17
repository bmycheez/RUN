import torch
import torch.nn as nn
from math import sqrt


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        out = self.dncnn(x)
        return out


class _DCR_block(nn.Module):
    def __init__(self, features, ):
        super(_DCR_block, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        for _ in range(1):
            # layers.append(nn.BatchNorm2d(features))
            # layers.append(nn.PReLU())
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
                                    padding=padding))
            layers.append(nn.PReLU())
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
                                    padding=padding))
        self.bpc = nn.Sequential(*layers)

    def forward(self, x):

        out = self.bpc(x)

        out += x

        return out


class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)

        out = self.relu(self.conv(out))

        return out


class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))

        out = self.subpixel(out)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        cn = 64
        self.conv_i = nn.Conv2d(in_channels=1, out_channels=cn, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        # self.DCR_block11 = self.make_layer(_DCR_block, cn)
        # self.DCR_block12 = self.make_layer(_DCR_block, cn)
        # self.down1 = self.make_layer(_down, cn)
        self.DCR_block21 = self.make_layer(_DCR_block, cn*1)
        self.DCR_block22 = self.make_layer(_DCR_block, cn*1)
        self.down2 = self.make_layer(_down, cn*1)
        self.DCR_block31 = self.make_layer(_DCR_block, cn*2)
        self.DCR_block32 = self.make_layer(_DCR_block, cn*2)
        self.down3 = self.make_layer(_down, cn*2)
        self.DCR_block41 = self.make_layer(_DCR_block, cn*4)
        self.DCR_block42 = self.make_layer(_DCR_block, cn*4)
        self.up3 = self.make_layer(_up, cn*8)
        self.DCR_block33 = self.make_layer(_DCR_block, cn*4)
        self.DCR_block34 = self.make_layer(_DCR_block, cn*4)
        self.up2 = self.make_layer(_up, cn*4)
        self.DCR_block23 = self.make_layer(_DCR_block, cn*2)
        self.DCR_block24 = self.make_layer(_DCR_block, cn*2)
        # self.up1 = self.make_layer(_up, cn*4)
        # self.DCR_block13 = self.make_layer(_DCR_block, cn*2)
        # self.DCR_block14 = self.make_layer(_DCR_block, cn*2)
        self.conv_f = nn.Conv2d(in_channels=cn*2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_i(x))

        # out = self.DCR_block11(out)

        # conc1 = self.DCR_block12(out)

        # out = self.down1(conc1)

        out = self.DCR_block21(out)

        conc2 = self.DCR_block22(out)

        out = self.down2(conc2)

        out = self.DCR_block31(out)

        conc3 = self.DCR_block32(out)

        conc4 = self.down3(conc3)

        out = self.DCR_block41(conc4)

        out = self.DCR_block42(out)

        out = torch.cat([conc4, out], 1)

        out = self.up3(out)

        out = torch.cat([conc3, out], 1)

        out = self.DCR_block33(out)

        out = self.DCR_block34(out)

        out = self.up2(out)

        out = torch.cat([conc2, out], 1)

        out = self.DCR_block23(out)

        out = self.DCR_block24(out)

        # out = self.up1(out)

        # out = torch.cat([conc1, out], 1)

        # out = self.DCR_block13(out)

        # out = self.DCR_block14(out)

        out = self.relu2(self.conv_f(out))

        out = torch.add(residual, out)

        return out


class MultiScale(nn.Module):
    def __init__(self, features):
        super(MultiScale, self).__init__()
        self.conv_31 = nn.Conv2d(features, features, 3, 1, 1)
        self.relu_31 = nn.PReLU()
        self.conv_51 = nn.Conv2d(features, features, 5, 1, 2)
        self.relu_51 = nn.PReLU()
        self.conv_32 = nn.Conv2d(2 * features, 2 * features, 3, 1, 1)
        self.relu_32 = nn.PReLU()
        self.conv_52 = nn.Conv2d(2 * features, 2 * features, 5, 1, 2)
        self.relu_52 = nn.PReLU()
        self.conv = nn.Conv2d(4 * features, features, 1, 1, 0)

    def forward(self, x):

        out1 = self.relu_31(self.conv_31(x))

        out2 = self.relu_51(self.conv_51(x))

        out = torch.cat([out1, out2], 1)

        out1 = self.relu_32(self.conv_32(out))

        out2 = self.relu_52(self.conv_52(out))

        out = self.conv(torch.cat([out1, out2], 1))

        out += x

        return out


class Plus(nn.Module):
    def __init__(self, features):
        super(Plus, self).__init__()
        kernel_size = 5
        self.conv_v = nn.Conv2d(features, features, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size//2))
        self.conv_h = nn.Conv2d(features, features, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size//2, 0))
        self.conv_c = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.conv_g = nn.Conv2d(3 * features, features, kernel_size=3, stride=1, padding=1)

    def make_layer(self, block, channel_in, number):
        layers = []
        for _ in range(number):
            layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):

        a = self.conv_v(x)

        b = self.conv_h(x)

        c = self.conv_c(x)

        out = self.conv_g(torch.cat([a, b, c], 1))

        out += x

        return out


class MyNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MyNet, self).__init__()
        features = 64
        self.conv_i = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=1
                                , stride=1, padding=0)
        self.relu_i = nn.PReLU()
        self.m1 = self.make_layer(MultiScale, features, 1)
        self.p1 = self.make_layer(Plus, features, 1)
        self.m2 = self.make_layer(MultiScale, features, 1)
        self.p2 = self.make_layer(Plus, features, 1)
        self.m3 = self.make_layer(MultiScale, features, 1)
        self.p3 = self.make_layer(Plus, features, 1)
        self.m4 = self.make_layer(MultiScale, features, 1)
        self.p4 = self.make_layer(Plus, features, 1)
        self.m5 = self.make_layer(MultiScale, features, 1)
        self.p5 = self.make_layer(Plus, features, 1)
        self.m6 = self.make_layer(MultiScale, features, 1)
        self.p6 = self.make_layer(Plus, features, 1)
        self.m7 = self.make_layer(MultiScale, features, 1)
        self.p7 = self.make_layer(Plus, features, 1)
        self.conv_o = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1
                                , stride=1, padding=0)
        self.relu_o = nn.PReLU()

    def make_layer(self, block, channel_in, number):
        layers = []
        for _ in range(number):
            layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.relu_i(self.conv_i(x))

        a0 = out

        a1 = self.m1(a0)

        a2 = self.p1(a1)

        a2 += a0

        a3 = self.m2(a2)

        a3 += a1

        a4 = self.p2(a3)

        a4 += a2

        a5 = self.m3(a4)

        a5 += a3

        a6 = self.p3(a5)

        a6 += a4

        a7 = self.m4(a6)

        a7 += a5
        
        a8 = self.p4(a7)

        a8 += a6
        
        a9 = self.m5(a8)

        a9 += a7

        out = self.relu_o(self.conv_o(a9))

        out += x

        return out
