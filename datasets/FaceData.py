import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random
import itertools
from .utils import *
import sys
sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 3)[0])
#print(sys.path)
from preprocessing import faceswapper



class FaceData(Dataset):

    def __init__(self, data_root, image_size):

        self.data_root = data_root
        self.image_size = image_size

        # path
        self.train_path = os.path.join(data_root, 'train')
        self.test_path = os.path.join(data_root, 'test')
        self.ref_path = os.path.join(data_root, 'ref.jpg')

        self.train_files = list(filter(lambda x: not x.startswith('.'), os.listdir(self.train_path)))
        self.test_files = list(filter(lambda x: not x.startswith('.'), os.listdir(self.test_path)))

        trans = []
        trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.Resize(image_size))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(trans)

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):

        filename = self.train_files[index]
        image = Image.open(os.path.join(self.train_path, filename))
        if not hasattr(self, 'ref_pho'):
            self.ref_pho = Image.open(self.ref_path)

        return self.transform(image), self.transform(self.ref_pho) # label: ont-hot

    def get_test(self, index, n):
        files = random.sample(self.test_files, n)
        test_phos = []
        for f in files:
            test_phos.append(self.transform(Image.open(os.path.join(self.test_path, f))))
        return torch.stack(test_phos, dim=0)

    def get_loader(self, **kwargs):
        return DataLoader(self, **kwargs)





