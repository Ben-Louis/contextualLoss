import torch.nn as nn
import torch.nn.functional as F
from .model_utils import *


class Generator_v1(Module):
    def __init__(self, conv_dim, image_size):
        super(Generator_v1, self).__init__()

        self.conv_layers = []

        self.conv1 = Sequential(
            ConvBlock(10, conv_dim, norm='in', act='relu', kernel_size=7, stride=1, padding=3),
            ResBlock(conv_dim),
            ConvBlock(conv_dim, conv_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.conv_layers.append(self.conv1)
        curr_dim = conv_dim

        self.conv2 = Sequential(
            ConvBlock(curr_dim+6, curr_dim*2, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim*2),
            ConvBlock(curr_dim*2, curr_dim*2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv2)
        curr_dim = curr_dim * 2     

        self.conv3 = Sequential(
            ConvBlock(curr_dim+6, curr_dim*2, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim*2),
            ConvBlock(curr_dim*2, curr_dim*2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv3)
        curr_dim = curr_dim * 2             

        self.conv4 = Sequential(
            ConvBlock(curr_dim+6, curr_dim, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv4)
        curr_dim = curr_dim

        self.conv5 = Sequential(
            ConvBlock(curr_dim+6, curr_dim, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv5)
        curr_dim = curr_dim 

        self.conv6 = Sequential(
            ConvBlock(curr_dim+6, curr_dim, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.conv_layers.append(self.conv6)
        curr_dim = curr_dim

        # image_size = 4, curr_dim = conv_dim * 4
        self.bottle_neck = Sequential(
            ResBlock(curr_dim),
            ResBlock(curr_dim),
            ResBlock(curr_dim))
        curr_dim = curr_dim

        self.deconv_layers = []
        self.deconv6 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.deconv_layers.append(self.deconv6)
        curr_dim = curr_dim

        self.deconv5 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.deconv_layers.append(self.deconv5)
        curr_dim = curr_dim

        self.deconv4 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim*2, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim*2),
            ResBlock(curr_dim*2),
            ConvBlock(curr_dim*2, curr_dim*2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.deconv_layers.append(self.deconv4)
        curr_dim = curr_dim*2     

        self.deconv3 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.deconv_layers.append(self.deconv3)
        curr_dim = curr_dim

        self.deconv2 = Sequential(
            ConvBlock(curr_dim+conv_dim*2, curr_dim//2, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim//2),
            ConvBlock(curr_dim//2, curr_dim//2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.deconv_layers.append(self.deconv2)
        curr_dim = curr_dim//2                     

        self.deconv1 = Sequential(
            ConvBlock(curr_dim+conv_dim, curr_dim//2, norm='in', act='relu', transpose=True, kernel_size=3, stride=1, padding=1),
            #ResBlock(curr_dim//2),
            #ConvBlock(curr_dim//2, curr_dim//2, norm='in', act='relu', kernel_size=3, stride=1, padding=1),
            ConvBlock(curr_dim//2, 3, norm='', act='tanh', kernel_size=7, stride=1, padding=3, bias=False))            
        self.deconv_layers.append(self.deconv1)
            

    def forward(self, src_img, ref_img):
        src_img, src_mask = src_img[:,:3,:,:], src_img[:,3:,:,:]
        ref_img, ref_mask = ref_img[:,:3,:,:], ref_img[:,3:,:,:]

        src_fg = src_img * src_mask
        ref_fg = ref_img * ref_mask

        img = torch.cat([src_img, src_fg, ref_fg, src_mask], dim=1)
        feats = [self.conv_layers[0](img)]
        img_fg = torch.cat([src_fg, ref_fg], dim=1)

        for i in range(1,len(self.conv_layers)):
            feats.append(self.conv_layers[i](torch.cat([feats[-1], img_fg], dim=1)))
            img_fg = F.avg_pool2d(img_fg, 2)

        feats.reverse()
        feat = self.bottle_neck(feats[0])

        for i in range(len(self.deconv_layers)):
            feat = self.deconv_layers[i](torch.cat([feat, feats[i]], dim=1))

        out = src_img * (1-src_mask) + feat * src_mask
        return out


class Generator_v2(Module):
    def __init__(self, conv_dim, image_size):
        super(Generator_v2, self).__init__()

        self.conv_layers = []

        self.conv1 = Sequential(
            ConvBlock(6, conv_dim, norm='in', act='relu', kernel_size=7, stride=1, padding=3),
            ResBlock(conv_dim),
            ConvBlock(conv_dim, conv_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.conv_layers.append(self.conv1)
        curr_dim = conv_dim

        self.conv2 = Sequential(
            ConvBlock(curr_dim+6, curr_dim*2, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim*2),
            ConvBlock(curr_dim*2, curr_dim*2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv2)
        curr_dim = curr_dim * 2     

        self.conv3 = Sequential(
            ConvBlock(curr_dim+6, curr_dim*2, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim*2),
            ConvBlock(curr_dim*2, curr_dim*2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv3)
        curr_dim = curr_dim * 2             

        self.conv4 = Sequential(
            ConvBlock(curr_dim+6, curr_dim, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv4)
        curr_dim = curr_dim

        self.conv5 = Sequential(
            ConvBlock(curr_dim+6, curr_dim, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.conv_layers.append(self.conv5)
        curr_dim = curr_dim 

        self.conv6 = Sequential(
            ConvBlock(curr_dim+6, curr_dim, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.conv_layers.append(self.conv6)
        curr_dim = curr_dim

        # image_size = 4, curr_dim = conv_dim * 4
        self.bottle_neck = Sequential(
            ResBlock(curr_dim),
            ResBlock(curr_dim),
            ResBlock(curr_dim))
        curr_dim = curr_dim

        self.deconv_layers = []
        self.deconv6 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.deconv_layers.append(self.deconv6)
        curr_dim = curr_dim

        self.deconv5 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))
        self.deconv_layers.append(self.deconv5)
        curr_dim = curr_dim

        self.deconv4 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim*2, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim*2),
            ResBlock(curr_dim*2),
            ConvBlock(curr_dim*2, curr_dim*2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.deconv_layers.append(self.deconv4)
        curr_dim = curr_dim*2     

        self.deconv3 = Sequential(
            ConvBlock(curr_dim+conv_dim*4, curr_dim, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            Self_Attn(curr_dim),
            ResBlock(curr_dim),
            ConvBlock(curr_dim, curr_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.deconv_layers.append(self.deconv3)
        curr_dim = curr_dim

        self.deconv2 = Sequential(
            ConvBlock(curr_dim+conv_dim*2, curr_dim//2, norm='in', act='relu', transpose=True, kernel_size=4, stride=2, padding=1),
            ResBlock(curr_dim//2),
            ConvBlock(curr_dim//2, curr_dim//2, norm='in', act='relu', kernel_size=3, stride=1, padding=1))            
        self.deconv_layers.append(self.deconv2)
        curr_dim = curr_dim//2                     

        self.deconv1 = Sequential(
            ConvBlock(curr_dim+conv_dim, curr_dim//2, norm='in', act='relu', transpose=True, kernel_size=3, stride=1, padding=1),
            #ResBlock(curr_dim//2),
            #ConvBlock(curr_dim//2, curr_dim//2, norm='in', act='relu', kernel_size=3, stride=1, padding=1),
            ConvBlock(curr_dim//2, 3, norm='', act='tanh', kernel_size=7, stride=1, padding=3, bias=False))            
        self.deconv_layers.append(self.deconv1)
            

    def forward(self, src_img, ref_img):

        img_fg = torch.cat([src_img, ref_img], dim=1)
        feats = [self.conv_layers[0](img_fg)]
        
        for i in range(1,len(self.conv_layers)):
            feats.append(self.conv_layers[i](torch.cat([feats[-1], img_fg], dim=1)))
            img_fg = F.avg_pool2d(img_fg, 2)

        feats.reverse()
        feat = self.bottle_neck(feats[0])

        for i in range(len(self.deconv_layers)):
            feat = self.deconv_layers[i](torch.cat([feat, feats[i]], dim=1))

        out = feat
        return out










