import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms

import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def is_image_file(filename, extensions):
    return any(filename.lower().endswith(ext) for ext in extensions)

class PairedImageDataset(Dataset): # 这里没用做任何加强，只是读取图片
    def __init__(self, root, extensions, transform=None, same_shape = True):
        self.low_dir = osp.join(root, 'LR')
        self.super_dir = osp.join(root, 'HR')
        self.extensions = extensions
        self.transform = transform
        self.same_shape = same_shape
        # 获取low和super文件夹下的所有图片
        self.filenames = sorted([f for f in os.listdir(self.low_dir) if is_image_file(f, extensions)])
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_path = osp.join(self.low_dir, filename)
        super_path = osp.join(self.super_dir, filename)
        
        low_img = Image.open(low_path).convert('RGB')
        super_img = Image.open(super_path).convert('RGB')
        
        if self.same_shape:
            low_img = low_img.resize(super_img.size, Image.BICUBIC)
        
        if self.transform:
            low_img = self.transform(low_img)
            super_img = self.transform(super_img)
        
        return low_img, super_img

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        # transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        # transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    train_set = PairedImageDataset(root=osp.join(data_path, 'train'), extensions=IMG_EXTENSIONS, transform=train_aug)
    val_set = PairedImageDataset(root=osp.join(data_path, 'val'), extensions=IMG_EXTENSIONS, transform=val_aug)
    
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
