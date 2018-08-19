import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Module(nn.Module):
    def load_state_dict(self, state_dict, strict=False):
        """
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    pass
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
                
class Sequential(nn.Sequential):
    def load_state_dict(self, state_dict, strict=False):
        """
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    pass
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))                

class ConvBlock(Module):
    def __init__(self, dim_in, dim_out, norm='', act='', transpose=False, sn=False, **kwargs):
        super(ConvBlock, self).__init__()

        layers = []

        if norm == 'in':
            kwargs['bias'] = False

        if not transpose:
            conv = nn.Conv2d(dim_in, dim_out, **kwargs)
        else:
            conv = nn.ConvTranspose2d(dim_in, dim_out, **kwargs)
        if sn:
            conv = SpectralNorm(conv)
        layers.append(conv)

        # initialize normalization
        norm_dim = dim_out
        if norm == 'bn':
            layers.append(nn.BatchNorm2d(norm_dim))
        elif norm == 'in':
            layers.append(nn.InstanceNorm2d(norm_dim, affine=True, track_running_stats=True))
        elif norm == 'ln':
            layers.append(LayerNorm(norm_dim))

        # initialize act
        if act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif act == 'prelu':
            layers.append(nn.PReLU())
        elif act == 'selu':
            layers.append(nn.SELU(inplace=True))
        elif act == 'elu':
            layers.append(nn.ELU(inplace=True))            
        elif act == 'tanh':
            layers.append(nn.Tanh())

        self.main = Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
        
class ResBlock(Module):
    def __init__(self, dim, norm='in', act='relu', dilation=1):
        super(ResBlock, self).__init__()

        model = []
        model += [ConvBlock(dim ,dim, norm=norm, act=act, kernel_size=3, stride=1, padding=dilation, dilation=dilation)]
        model += [ConvBlock(dim ,dim, norm=norm, act='none', kernel_size=3, stride=1, padding=dilation, dilation=dilation)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out  

class Self_Attn(Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation='none'):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)      

class STN(Module):
    def __init__(self, in_dim):
        super(STN, self).__init__()

        self.conv = Sequential(
            ConvBlock(in_dim, in_dim, norm='in', act='relu', kernel_size=3, stride=1, padding=2, dilation=1),
            ConvBlock(in_dim, in_dim//2, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            ConvBlock(in_dim//2, in_dim//4, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            Self_Attn(in_dim//4), nn.ReLU(True),
            ConvBlock(in_dim//4, in_dim//4, norm='in', act='relu', kernel_size=4, stride=2, padding=1),
            Self_Attn(in_dim//4), nn.ReLU(True),
            nn.Conv2d(in_dim//4, 6, kernel_size=3, stride=1, padding=1)
            )

        self.conv[-1].bias.data = torch.FloatTensor([1,0,0,0,1,0])
        self.conv[-1].weight.data.zero_()

    def forward(self, feat):
        theta = self.conv(feat)
        h,w = theta.size(2), theta.size(3)
        theta = F.avg_pool2d(theta, (h,w)).view(-1, 2, 3)

        feat = STN.affine_map(feat, theta)
        return feat

    @staticmethod
    def affine_map(feat, theta):
        grid = F.affine_grid(theta, feat.size())
        x = F.grid_sample(feat, grid)  
        return x

    @staticmethod
    def inverse_theta(theta):
        inv_theta = []
        for i in range(len(theta)):
            inv_theta.append(torch.cat([torch.inverse(theta[i][:,:2]),-theta[i][:,2:3]], dim=1))

        return torch.stack(inv_theta)









