import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random
import itertools


def random_flip(ts, p=0.5):
    assert isinstance(ts, torch.Tensor)
    assert len(ts.shape) == 3
    idx = torch.arange(ts.size(2)-1,-1,-1).long()    
    return ts.index_select(index=idx, dim=2)

def crop(ts, size, mode='center'):
    assert isinstance(ts, torch.Tensor)
    assert len(ts.shape) == 3
    h, w = ts.size(1), ts.size(2)
    assert size <= min(h, w)

    if mode == 'center':
        y_start, x_start = (h-size) // 2, (w-size)//2
    elif mode == 'random':
        y_start = random.randint(0,h-size-1)
        x_start = random.randint(0,w-size-1)

    return ts[:, y_start:y_start+size, x_start:x_start+size]

def random_color(ts):
    assert isinstance(ts, torch.FloatTensor)

    # change hue
    for d in range(3):
        hew_ratio = random.random() * 0.2 + 0.9
        ts[d] = ts[d] * hew_ratio

    # change illumination
    illu_ratio = random.random() * 0.2 + 0.9
    ts[:3] = ts[:3] * illu_ratio

    # change saturation
    mean = ts[:3].mean(dim=0, keepdim=True)
    satu_ratio = random.random() * 0.2 + 0.9
    ts[:3] = mean * (1-satu_ratio) + ts[:3] * satu_ratio

    return ts


    




