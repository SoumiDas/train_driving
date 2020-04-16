import numpy as np
import torch
import os
import math
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from numpy.lib.stride_tricks import as_strided
#from ipdb import set_trace

### helper functions

def tile_array(a, b0, b1):
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(r*b0, c*b1)                      # create new 2D array

def get_bool_vec(d, n_h):
    # first one hot encode to three possible classesś
    t = np.zeros((len(d), 3))
    t[np.arange(len(d)), d+1] = 1
    # upscale to correct hidden_sz
    bool_vec = tile_array(t, 1, n_h//3)
    return torch.Tensor(bool_vec)

### custom modules

class Flatten(nn.Module):
    def __init__(self, full:bool=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class Swap1and2(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x): return x.permute(0, 2, 1)

class TimeSlice(nn.Module):
    """
    given sequential input, return only sequence of length l with dilation d
    this module assumes that the time dim is the second dim (after batch dim)
    """
    def __init__(self, l:int, d:int=1):
        super().__init__()
        self.l = l
        self.d = d

    def forward(self, x):
        if self.d>1: # apply dilation
            x = torch.flip(x, dims=[1])
            x = x[:, ::self.d]
            x = torch.flip(x, dims=[1])

        if self.l==-1: # return full sequence
            return x
        else:
            return x[:, -self.l:]

### params

class NetworkParams():
    name = 'default'
    type_ = 'LSTM'
    p = 0.44
    n_h = 100
    lr = 1e-4
    dil = 1
    seq_len = 10

    def update(self, params):
        if params is None:
            return True

        for k, v in params.items():
                setattr(self, k, v)

### main network

class TaskBlock(nn.Module):
    def __init__(self, params, n_in, n_out, cond=False):
        super(TaskBlock, self).__init__()
        self.cond = cond
        self.n_h = params.n_h if not cond else params.n_h*3
        self.type = params.type_
        self.discrete = n_out > 1

        self.bn = nn.BatchNorm1d(self.n_h)
        self.dropout = nn.Dropout(params.p)
        self.lin_out = nn.Linear(self.n_h, n_out)

        # different core units dependent on task block type
        if params.type_=='MLP':
            self.core = nn.Sequential(
                TimeSlice(1),
                Flatten(),
                nn.Linear(n_in, self.n_h),
                nn.ReLU(inplace=False),
            )
        elif params.type_=='LSTM':
            self.core = nn.Sequential(
                TimeSlice(l=params.seq_len, d=params.dil),
                nn.LSTM(n_in, self.n_h, 1, batch_first=True),
            )
        elif params.type_=='GRU':
            self.core = nn.Sequential(
                TimeSlice(l=params.seq_len, d=params.dil),
                nn.GRU(n_in, self.n_h, 1, batch_first=True),
            )
        elif params.type_=='TempConv':
            self.core = nn.Sequential(
                TimeSlice(l=params.seq_len, d=params.dil),
                Swap1and2(),
                nn.Conv1d(n_in, self.n_h, params.seq_len),
                nn.ReLU(inplace=False),
            )
        else:
            raise ValueError(f"Task block type {params.type_} not known")

    def forward(self, x_in, d=None):
        x = self.core(x_in)
        if self.type=='LSTM':
            x = x[1][1].squeeze()
        elif self.type=='GRU':
            x = x[1].squeeze()
        elif self.type=='TempConv':
            x = x.squeeze()

        if self.discrete: x = self.bn(x)
        x = self.dropout(x)

        # handle conditional affordances
        if self.cond:
            bool_vec = get_bool_vec(d, self.n_h).to(x.device)
            x *= bool_vec

        x = self.lin_out(x)
        # # handle discrete affordances
        if self.lin_out.out_features > 1:
            x = F.softmax(x)

        return x

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class CAL_network(nn.Module):
    def __init__(self, p):
        super(CAL_network, self).__init__()
        self.params = p


        if os.path.exists('./vgg19_bn-6002323d.pth'):
            print('Weights getting loaded')
            vgg = VGG(make_layers(cfg['E'], batch_norm=True))
            vgg.load_state_dict(torch.load('./vgg19_bn-6002323d.pth'))
            print('Weights loaded')
            #vgg.load_state_dict(torch.load('./vggmodel'))
        else:
            print('Weights getting downloaded')
            vgg = models.vgg19_bn(pretrained=True)
            torch.save(vgg.state_dict(),'./vgg19_bn-6002323d.pth')
            print('Weights downloaded')
            
        # get feature extractor and first FCN layer from vgg
        #vgg = models.vgg11_bn(pretrained=True)
        #vgg = models.vgg11(pretrained=True)
        ls = [l for l in vgg.features]+ [nn.AdaptiveMaxPool2d(1), Flatten()]
        self.features = nn.Sequential(*ls)
        n_in = 512 # fixed amount of features after feature extractor
        print('VGG created')
        # initialize the task blocks
        p.type_='GRU'
        #p.seq_len=14
        p.seq_len=10
        p.dil=2
        #p.n_h=120
        p.n_h=100
        #p.p=0.27
        p.p = 0.5
        self.red_light = TaskBlock(params=p, n_in=512, n_out=2)
        p.type_='TempConv'
        p.seq_len=6
        p.dil=1
        p.n_h=120
        p.p=0.68
        self.hazard_stop = TaskBlock(params=p, n_in=512, n_out=2)
        p.type_='MLP'
        p.seq_len=1
        p.n_h=100
        p.p=0.55
        self.speed_sign = TaskBlock(params=p, n_in=512,  n_out=4)
        p.type_='GRU'
        p.seq_len=11
        p.n_h=120
        p.p=0.38
        self.veh_distance = TaskBlock(params=p, n_in=512, n_out=1)
        p.type_='LSTM'
        p.seq_len=10
        #p.n_h=100
        p.n_h = 100
        p.p=0.7
        '''p.type_='GRU'
        p.seq_len=10
        p.n_h=120
        p.p=0.38'''
        self.relative_angle = TaskBlock(params=p, n_in=512, n_out=1, cond=True)
        #self.relative_angle = TaskBlock(params=p, n_in=512, n_out=1)
        p.type_='LSTM'
        p.seq_len=10
        p.n_h=120
        #p.p=0.6
        p.p=0.7
        '''p.type_='GRU'
        p.seq_len=10
        p.n_h=120
        p.p=0.38'''
        self.center_distance = TaskBlock(params=p, n_in=512, n_out=1, cond=True)
        #self.center_distance = TaskBlock(params=p, n_in=512, n_out=1)

    def forward(self, inputs):
        # feature extractor over sequence, reshape appropriately
        x_in = inputs['sequence']
        #print(x_in)
        batch_size, timesteps, C, H, W = x_in.size()

        c_in = x_in.view(batch_size * timesteps, C, H, W)
        c_out = self.features(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)

        # unconditional affordances
        pred = {}
        pred['red_light'] = self.red_light(r_in)
        pred['hazard_stop'] = self.hazard_stop(r_in)
        pred['speed_sign'] = self.speed_sign(r_in)
        pred['veh_distance'] = self.veh_distance(r_in)
        pred['relative_angle'] = self.relative_angle(r_in, inputs['direction'])
        pred['center_distance'] = self.center_distance(r_in, inputs['direction'])

        return pred

def get_model(params={}):
    p = NetworkParams()
    p.update(params)
    model = CAL_network(p)

    for name,param in model.named_parameters():
        if 'features' in name:
            param.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=p.lr)
    return model, opt
