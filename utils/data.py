"""Paired LR/HR dataset for var/.

Supports both same-shape (LR upsampled to HR resolution) and different-shape
(LR_64x64 + HR_256x256) data layouts. The two modes are selected via the
`same_shape` flag and the `lr_folder`/`hr_folder` arguments.

For data augmentation across different LR/HR resolutions, the flip is applied to
both images in sync; cropping is only run when LR and HR share the same size
(the legacy SR-from-blurred case). When LR is a small native low-resolution
image, we skip cropping since pixel-aligned cropping would require matching the
LR/HR scale ratio precisely.
"""
import math
import os
import os.path as osp
import random

import PIL.Image as PImage
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def is_image_file(filename, extensions):
    return any(filename.lower().endswith(ext) for ext in extensions)


def pil_mode_from_channels(img_channels: int) -> str:
    if int(img_channels) == 1:
        return 'L'
    if int(img_channels) == 3:
        return 'RGB'
    raise ValueError(f'img_channels must be 1 or 3, got {img_channels}')


def center_crop_arr(images_list, image_size, min_crop_frac=0.9, max_crop_frac=1.0):
    """Center cropping (ADM style). Requires all images in `images_list` to have the same size."""
    assert len(images_list) > 0
    for i in range(1, len(images_list)):
        assert images_list[i].size == images_list[0].size, \
            f"Image {i} size {images_list[i].size} does not match image 0 size {images_list[0].size}"

    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

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

    arr = [np.array(images_list[i]) for i in range(len(images_list))]
    crop_y = (arr[0].shape[0] - image_size) // 2
    crop_x = (arr[0].shape[1] - image_size) // 2

    return [
        Image.fromarray(arr[i][crop_y: crop_y + image_size, crop_x: crop_x + image_size])
        for i in range(len(images_list))
    ]


class PairedImageDataset(Dataset):
    """Dataset of (LR, HR[, Ref]) paired images.

    Args:
        root: Root directory (typically `DATA_PATH/train` or `.../val`).
        extensions: Allowed image extensions tuple.
        transform: Final per-image transform (e.g. ToTensor + normalize).
        same_shape: If True, LR is bicubic-resized to HR's resolution at load
            time (legacy behavior). If False (recommended for LR_64x64), LR is
            left at its native resolution and cropping/upsampling is skipped to
            avoid pixel misalignment.
        augment: Enable random flip (and crop when `same_shape=True`).
        use_ref: Also load a `ref_folder` image (matched by filename).
        lr_folder, hr_folder, ref_folder: Subdirectory names under `root`.
    """

    def __init__(
        self,
        root: str,
        extensions=IMG_EXTENSIONS,
        transform=None,
        same_shape: bool = False,
        augment: bool = False,
        use_ref: bool = False,
        lr_folder: str = 'LR_64x64',
        hr_folder: str = 'HR',
        ref_folder: str = 'Ref',
        img_channels: int = 3,
    ):
        self.low_dir = osp.join(root, lr_folder)
        self.super_dir = osp.join(root, hr_folder)
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        if use_ref:
            self.ref_dir = osp.join(root, ref_folder)
        self.ref_folder = ref_folder
        self.extensions = extensions
        self.transform = transform
        self.same_shape = same_shape
        self.augment = augment
        self.use_ref = use_ref
        self.image_mode = pil_mode_from_channels(img_channels)

        assert osp.isdir(self.low_dir), f'LR folder not found: {self.low_dir}'
        assert osp.isdir(self.super_dir), f'HR folder not found: {self.super_dir}'
        if self.use_ref:
            assert osp.isdir(self.ref_dir), f'Ref folder not found: {self.ref_dir}'

        self.filenames = sorted([f for f in os.listdir(self.low_dir) if is_image_file(f, extensions)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_path = osp.join(self.low_dir, filename)
        super_path = osp.join(self.super_dir, filename)

        low_img = Image.open(low_path).convert(self.image_mode)
        super_img = Image.open(super_path).convert(self.image_mode)

        if self.use_ref:
            ref_path = osp.join(self.ref_dir, filename)
            ref_img = Image.open(ref_path).convert(self.image_mode)

        if self.same_shape and low_img.size != super_img.size:
            low_img = low_img.resize(super_img.size, Image.BICUBIC)

        # Flip axis hardcoded for two known dataset families (see legacy code).
        flip_axis = 0 if ("file" in filename or "LIDC-IDRI" in filename) else 1

        if self.augment:
            flip_p = random.random() > 0.50
            # Crop only makes sense when LR and HR share the same resolution.
            crop_p = (random.random() > 0.33) and (low_img.size == super_img.size)

            if self.use_ref:
                if crop_p:
                    low_img, super_img, ref_img = center_crop_arr(
                        [low_img, super_img, ref_img], super_img.size[0]
                    )
                if flip_p:
                    low_img = low_img.transpose(flip_axis)
                    super_img = super_img.transpose(flip_axis)
                    ref_img = ref_img.transpose(flip_axis)
            else:
                if crop_p:
                    low_img, super_img = center_crop_arr(
                        [low_img, super_img], super_img.size[0]
                    )
                if flip_p:
                    low_img = low_img.transpose(flip_axis)
                    super_img = super_img.transpose(flip_axis)

        if self.transform:
            low_img = self.transform(low_img)
            super_img = self.transform(super_img)
            if self.use_ref:
                ref_img = self.transform(ref_img)

        if self.use_ref:
            return low_img, super_img, ref_img
        return low_img, super_img


def normalize_01_into_pm1(x):
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str,
    augment: bool = True,
    use_ref: bool = False,
    lr_folder: str = 'LR_64x64',
    hr_folder: str = 'HR',
    same_shape: bool = False,
    ref_folder: str = 'Ref',
    img_channels: int = 3,
):
    train_aug = transforms.Compose([transforms.ToTensor(), normalize_01_into_pm1])
    val_aug = transforms.Compose([transforms.ToTensor(), normalize_01_into_pm1])

    train_set = PairedImageDataset(
        root=osp.join(data_path, 'train'), extensions=IMG_EXTENSIONS,
        transform=train_aug, augment=augment, use_ref=use_ref,
        lr_folder=lr_folder, hr_folder=hr_folder, ref_folder=ref_folder,
        same_shape=same_shape,
        img_channels=img_channels,
    )
    val_set = PairedImageDataset(
        root=osp.join(data_path, 'val'), extensions=IMG_EXTENSIONS,
        transform=val_aug, augment=False, use_ref=use_ref,
        lr_folder=lr_folder, hr_folder=hr_folder, ref_folder=ref_folder,
        same_shape=same_shape,
        img_channels=img_channels,
    )

    print(f'[Dataset] {len(train_set)=}, {len(val_set)=} '
          f'(image_mode={train_set.image_mode}, img_channels={img_channels}, lr_folder={lr_folder}, hr_folder={hr_folder}, same_shape={same_shape})')
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
    train_set, val_set = build_dataset(
        data_path='../data/brats_2021_256_t2_t1r_pair_4x_png',
        augment=True, use_ref=False, lr_folder='LR_64x64',
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    for low, hr in train_loader:
        print('LR:', low.shape, low.max(), low.min())
        print('HR:', hr.shape, hr.max(), hr.min())
        break
    print('Done!')
