import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import math
import random
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

def is_image_file(filename, extensions):
    return any(filename.lower().endswith(ext) for ext in extensions)

def center_crop_arr(images_list, image_size, min_crop_frac=0.9, max_crop_frac=1.0):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    assert len(images_list) > 0
    for i in range(1, len(images_list)):
        assert images_list[i].size == images_list[0].size, f"Image {i} size {images_list[i].size} does not match image 0 size {images_list[0].size}"
    
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*(images_list[0]).size) >= 2 * smaller_dim_size:
        for i in range(len(images_list)):
            images_list[i] = images_list[i].resize(
                tuple(x // 2 for x in images_list[i].size), resample=Image.BOX
            )

    scale = smaller_dim_size / min(*(images_list[0]).size)
    
    for i in range(len(images_list)):
        images_list[i] = images_list[i].resize(
            tuple(round(x * scale) for x in images_list[i].size), resample=Image.BICUBIC
        )

    
    arr = []
    for i in range(len(images_list)):
        arr.append(np.array(images_list[i]))

    crop_y = (arr[0].shape[0] - image_size) // 2
    crop_x = (arr[0].shape[1] - image_size) // 2

    return [Image.fromarray(arr[i][crop_y: crop_y + image_size, crop_x: crop_x + image_size]) for i in range(len(images_list))]
    
class PairedImageDataset(Dataset): # 这里没用做任何加强，只是读取图片
    def __init__(self, root, extensions, transform=None, same_shape = True, augment=False, use_ref = False):
        self.low_dir = osp.join(root, 'LR')
        self.super_dir = osp.join(root, 'HR')
        if use_ref:
            self.ref_dir = osp.join(root, 'Ref')
        self.extensions = extensions
        self.transform = transform
        self.same_shape = same_shape
        self.augment = augment
        self.use_ref = use_ref
        # 获取low和super文件夹下的所有图片
        self.filenames = sorted([f for f in os.listdir(self.low_dir) if is_image_file(f, extensions)])
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_path = osp.join(self.low_dir, filename)
        low_img = Image.open(low_path).convert('RGB')
        
        super_path = osp.join(self.super_dir, filename)
        super_img = Image.open(super_path).convert('RGB')
        
        if self.use_ref:
            ref_path = osp.join(self.ref_dir, filename)
            ref_img = Image.open(ref_path).convert('RGB')
        
        if self.same_shape and low_img.size != super_img.size:
            low_img = low_img.resize(super_img.size, Image.BICUBIC)
        
        if self.augment:
            flip_p = random.random() > 0.50
            crop_p = random.random() > 0.33
            if self.use_ref:
                if crop_p:
                    low_img, super_img, ref_img = center_crop_arr([low_img, super_img, ref_img], super_img.size[0])
                if flip_p:
                    low_img = low_img.transpose(1)
                    super_img = super_img.transpose(1)
                    ref_img = ref_img.transpose(1)
            else :
                if crop_p:
                    low_img, super_img = center_crop_arr([low_img, super_img], super_img.size[0])
                if flip_p:
                    low_img = low_img.transpose(1)
                    super_img = super_img.transpose(1)
            
        if self.transform:
            low_img = self.transform(low_img)
            super_img = self.transform(super_img)
            if self.use_ref:
                ref_img = self.transform(ref_img)
                
        if self.use_ref:
            return low_img, super_img, ref_img
        
        return low_img, super_img

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, 
    augment: bool = True,
    use_ref: bool = False,
):
    # build augmentations
    train_aug, val_aug = [
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    train_set = PairedImageDataset(root=osp.join(data_path, 'train'), extensions=IMG_EXTENSIONS, transform=train_aug, augment=augment,use_ref=use_ref)
    val_set = PairedImageDataset(root=osp.join(data_path, 'val'), extensions=IMG_EXTENSIONS, transform=val_aug, augment=False,use_ref=use_ref)
    
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

if __name__ == '__main__':
    # test
    train_set, val_set = build_dataset(data_path='../data/brats_2021_256_t2_t1r_pair_4x_png',augment=True,use_ref=True)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False,)
    for low, super, ref in train_loader:
        print(low.shape,low.max(),low.min())
        print(super.shape,super.max(),super.min())
        print(ref.shape,ref.max(),ref.min())
        low = (low.squeeze().permute(1, 2, 0) + 1.0) * 255.0 / 2.0
        super = (super.squeeze().permute(1, 2, 0) + 1.0) * 255.0 / 2.0
        ref = (ref.squeeze().permute(1, 2, 0) + 1.0) * 255.0 / 2.0
        
        low_img = Image.fromarray(low.numpy().astype('uint8'))
        super_img = Image.fromarray(super.numpy().astype('uint8'))
        ref_img = Image.fromarray(ref.numpy().astype('uint8'))
        
        low_img.save('low.png')
        super_img.save('super.png')
        ref_img.save('ref.png')
        
        break
    print('Done!')