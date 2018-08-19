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



class CelebAData(Dataset):

    """
    The file structure must follows:
    data_root
        |- Img
        |    |- img_align_celeba
        |    |    |- 000001.jpg
        |    |    +- ...
        |    +- ...
        |- Anno
        |    |- list_attr_celeba.txt
        |    |- identity_CelebA.txt
        |    +- ...
        +- Eval
            +- list_eval_partition.txt

    """

    def __init__(self, data_root, image_size, crop_size, selected_attrs=['Bangs'], mode='train'):

        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size

        # path
        self.image_dir = os.path.join(data_root, 'Img', 'img_align_celeba')
        self.attr_path = os.path.join(data_root, 'Anno', 'list_attr_celeba.txt')

        self.selected_attrs = selected_attrs
        self.attr2idx = {}
        self.idx2attr = {}
        self._preprocess(selected_attrs)

        self.mode = mode

        trans = []
        if mode == 'train':
            trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.CenterCrop(crop_size))
        trans.append(transforms.Resize(image_size))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(trans)

    def _preprocess(self, selected_attrs, test_num=2000):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(666)
        random.shuffle(lines)
        self.train_dataset = []
        self.test_dataset = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < test_num:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])    

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset)
        elif self.mode == 'test':
            return len(self.test_dataset)

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label) # label: ont-hot

    def get_loader(self, **kwargs):
        return DataLoader(self, **kwargs)



"""
5_o_Clock_Shadow 
Arched_Eyebrows 
Attractive 
Bags_Under_Eyes 
Bald Bangs 
Big_Lips 
Big_Nose 
Black_Hair 
Blond_Hair 
Blurry 
Brown_Hair 
Bushy_Eyebrows 
Chubby 
Double_Chin 
Eyeglasses 
Goatee 
Gray_Hair 
Heavy_Makeup 
High_Cheekbones 
Male 
Mouth_Slightly_Open 
Mustache 
Narrow_Eyes 
No_Beard 
Oval_Face 
Pale_Skin 
Pointy_Nose 
Receding_Hairline 
Rosy_Cheeks 
Sideburns 
Smiling 
Straight_Hair 
Wavy_Hair 
Wearing_Earrings 
Wearing_Hat
Wearing_Lipstick 
Wearing_Necklace 
Wearing_Necktie 
Young 
"""

class CelebAData_exchange(CelebAData):

    def __init__(self, data_root, image_size, crop_size, selected_attrs='eye', mode='train'):

        self.data_root = data_root
        self.image_size = image_size
        self.crop_size = crop_size

        # path
        self.image_dir = os.path.join(data_root, 'Img', 'img_align_celeba')
        self.mask_dir = os.path.join(data_root, 'Img', 'mask(%s)_align_celeba'%selected_attrs)
        self.valid_imgs_dir = os.path.join(os.path.abspath(__file__).rsplit(os.sep, 3)[0], 'preprocessing', 'celeba_valid_imgs1.txt')        

        self.attr_path = os.path.join(data_root, 'Anno', 'list_attr_celeba.txt')
        self.attr2idx = {}
        self.idx2attr = {}

        self.selected_attrs = selected_attrs
        self.preprocess()

        self.mode = mode

        # transforms
        trans = []
        trans.append(transforms.CenterCrop(self.crop_size))
        trans.append(transforms.Resize(self.image_size))
        self.determ_trans = transforms.Compose(trans)        
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # random transforms
        trans = []
        trans.append(transforms.CenterCrop(self.crop_size))
        trans.append(transforms.RandomResizedCrop(self.crop_size, scale=(0.9, 1.0), ratio=(1, 1)))
        trans.append(transforms.Resize(self.image_size))
        trans.append(transforms.RandomHorizontalFlip())
        self.random_trans = transforms.Compose(trans)

        self.swapper = faceswapper(predictor_path=os.path.join(os.path.abspath(__file__).rsplit(os.sep, 3)[0], 'preprocessing', 'shape_predictor_68_face_landmarks.dat'))

    def preprocess(self, num_test=500):
        """Preprocess the CelebA attribute file."""
        self._preprocess(['Bangs', 'Eyeglasses'], 0)

        dataset = set([line.rstrip() for line in open(self.valid_imgs_dir, 'r')])
        #random.seed(666)
        #random.shuffle(dataset)

        train_set, test_set = [], []

        for i, (fname, attr) in enumerate(self.train_dataset):
            if not any(attr) and fname in dataset:
                if i < num_test:
                    test_set.append(fname)
                else:
                    train_set.append(fname)

        self.train_dataset = train_set
        self.test_dataset = test_set

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset

        src_img_fname = dataset[index]
        ref_img_fname = random.choice(dataset)
        while ref_img_fname == src_img_fname:
            ref_img_fname = random.choice(dataset)

        # get exchanged images
        src_img = os.path.join(self.image_dir, src_img_fname)
        ref_img = os.path.join(self.image_dir, ref_img_fname)

        srcref_img, refsrc_img = self.swapper.swap(src_img, ref_img, mode=self.selected_attrs)
        srcref_img, refsrc_img = Image.fromarray(srcref_img[:,:,::-1]), Image.fromarray(refsrc_img[:,:,::-1])
        srcref_img = self.norm(random_color(self.to_tensor(self.random_trans(srcref_img))).clamp(0,1))
        refsrc_img = self.norm(self.to_tensor(self.determ_trans(refsrc_img)))

        # get masks and photos
        src_mask = self.to_tensor(self.determ_trans(Image.open(os.path.join(self.mask_dir, src_img_fname))))
        ref_mask = self.to_tensor(self.determ_trans(Image.open(os.path.join(self.mask_dir, ref_img_fname))))
        src_img = self.norm(self.to_tensor(self.determ_trans(Image.open(src_img))))
        ref_img = self.norm(self.to_tensor(self.determ_trans(Image.open(ref_img))))

        src_img = torch.cat([src_img, src_mask])
        ref_img = torch.cat([ref_img, ref_mask])

        #src_img = random_flip(src_img)
        ref_img = random_flip(ref_img)

        return src_img, ref_img, srcref_img, refsrc_img

    def get_test(self, index, n):
        dataset = self.test_dataset

        start = index
        end = min(index+n, len(dataset))

        src_imgs = []
        for i in range(start, end):
            src_img_fname = dataset[i]
            src_mask = self.to_tensor(self.determ_trans(Image.open(os.path.join(self.mask_dir, src_img_fname))))
            src_img = self.norm(self.to_tensor(self.determ_trans(Image.open(os.path.join(self.image_dir, src_img_fname)))))

            src_img = random_flip(torch.cat([src_img, src_mask]))
            src_imgs.append(src_img)

        return torch.stack(src_imgs)









