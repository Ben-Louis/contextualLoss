import torch.nn as nn
import torch.nn.functional as F
from .model_utils import *


class Discriminator_v1(Module):
    def __init__(self, conv_dim, image_size, use_stn=False):
        super(Discriminator_v1, self).__init__()


        self.conv_layers = []
        curr_dim = 3

        self.conv1 = ConvBlock(curr_dim, conv_dim, norm='', act='lrelu', kernel_size=7, stride=1, padding=3)
        self.conv_layers.append(self.conv1)
        curr_dim += conv_dim
	
        if use_stn:
            self.stn = STN_comp(conv_dim)
        else:
            self.stn = lambda x: x

        self.conv2 = Sequential(
            ConvBlock(curr_dim, curr_dim, norm='', act='lrelu', sn=False, kernel_size=3, stride=1, padding=1),
            ConvBlock(curr_dim, conv_dim, norm='', act='lrelu', sn=False, kernel_size=4, stride=2, padding=1))
        self.conv_layers.append(self.conv2)
        curr_dim += conv_dim

        self.conv3 = Sequential(
            ConvBlock(curr_dim, curr_dim, norm='', act='lrelu', sn=False, kernel_size=3, stride=1, padding=1),
            ConvBlock(curr_dim, conv_dim, norm='', act='lrelu', sn=False, kernel_size=4, stride=2, padding=1))
        self.conv_layers.append(self.conv3)
        curr_dim += conv_dim

        self.conv4 = Sequential(
            ConvBlock(curr_dim, curr_dim, norm='', act='lrelu', sn=False, kernel_size=3, stride=1, padding=1),
            Self_Attn(curr_dim), nn.LeakyReLU(0.1, True),
            ConvBlock(curr_dim, conv_dim, norm='', act='lrelu', sn=False, kernel_size=4, stride=2, padding=1))
        self.conv_layers.append(self.conv4)
        curr_dim += conv_dim        

        self.conv5 = Sequential(
            ConvBlock(curr_dim, curr_dim, norm='', act='lrelu', sn=False, kernel_size=3, stride=1, padding=1),
            Self_Attn(curr_dim), nn.LeakyReLU(0.1, True),
            ConvBlock(curr_dim, conv_dim, norm='', act='lrelu', sn=False, kernel_size=4, stride=2, padding=1))
        self.conv_layers.append(self.conv5)
        curr_dim += conv_dim  

        if image_size == 128:
            self.conv6 = Sequential(
                ConvBlock(curr_dim, curr_dim, norm='', act='lrelu', sn=False, kernel_size=3, stride=1, padding=1),
                Self_Attn(curr_dim), nn.LeakyReLU(0.1, True),
                ConvBlock(curr_dim, conv_dim, norm='', act='lrelu', sn=False, kernel_size=4, stride=2, padding=1))
            self.conv_layers.append(self.conv6)
            curr_dim += conv_dim

        self.out_conv = ConvBlock(curr_dim, 1,  kernel_size=4, stride=1, padding=0, bias=False)           

            

    def forward(self, fake_img):
        if fake_img.size(1) > 3:
            fake_img = fake_img[:,:3,:,:]

        feat = torch.cat([fake_img, self.stn(self.conv_layers[0](fake_img))], dim=1)
        for i in range(1, len(self.conv_layers)):
            feat = torch.cat([F.avg_pool2d(feat,2), self.conv_layers[i](feat)], dim=1)

        return self.out_conv(feat).view(feat.size(0), -1)




class STN_comp(STN):
    def __init__(self, in_dim):
        super(STN_comp, self).__init__(in_dim)

        self.conv = Sequential(
            ConvBlock(in_dim, in_dim, norm='', act='relu', kernel_size=4, stride=2, padding=1),
            ConvBlock(in_dim, in_dim//2, norm='', act='relu', kernel_size=4, stride=2, padding=1),
            ConvBlock(in_dim//2, in_dim//4, norm='', act='relu', kernel_size=4, stride=2, padding=1),
            Self_Attn(in_dim//4), nn.ReLU(True),
            ConvBlock(in_dim//4, in_dim//4, norm='', act='relu', kernel_size=4, stride=2, padding=1),
            Self_Attn(in_dim//4), nn.ReLU(True),
            nn.Conv2d(in_dim//4, 6, kernel_size=4, stride=1, padding=0))

        self.conv[-1].bias.data = torch.FloatTensor([1,0,0,0,1,0])
        self.conv[-1].weight.data.zero_()










