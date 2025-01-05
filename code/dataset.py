# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Code from https://github.com/facebookresearch/stable_signature/blob/main/ to load datasets from folder"""


import os
import functools

import numpy as np
from PIL import Image

from torchvision.datasets.folder import is_image_file, default_loader
from torch.utils.data import DataLoader, Subset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

functools.lru_cache()
def get_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])

def get_mixed_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            split_str = filename.split("_")
            paths.append((os.path.join(path, filename), split_str[-1].split(".")[0]))
    return sorted([fn for fn in paths if is_image_file(fn[0])])

class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
    
class ImageMixedFolder:
    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_mixed_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img, wm = self.samples[idx]
        img = self.loader(img)
        if self.transform:
            img = self.transform(img)
        return (img, wm)

    def __len__(self):
        return len(self.samples)
    
def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return torch.stack(batch)

def mixed_collate_fn(batch):
    images = []
    watermarks = []
    for b in batch:
        images.append(b[0])
        wm = []
        for w in b[1]:
            wm.append(int(w))
        wm = torch.tensor(wm).int()
        watermarks.append(wm)
    return torch.stack(images), torch.stack(watermarks)
        

def get_image_dataloader(data_dir, transform, batch_size=128, num_imgs=None, shuffle=False, num_workers=4, collate_fn=collate_fn):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if num_imgs is not None:
        dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)

def get_mixed_image_dataloader(data_dir, transform, batch_size=128, num_imgs=None, shuffle=False, num_workers=4, collate_fn=mixed_collate_fn):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageMixedFolder(data_dir, transform=transform)
    if num_imgs is not None:
        dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)


def create_mixed_dataset(encoder, image_size, num_bits, original_images_path, new_images_path, device="cpu"):
    os.makedirs(new_images_path, exist_ok=True)
    transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor()
                            ])
    loader = get_image_dataloader(original_images_path, transform=transform, batch_size=64)
    encoder = encoder.to(device)
    for i, images in enumerate(loader):
        images = images.to(device)
        watermarks = torch.randint(0, 2, (images.shape[0], num_bits)).float().to(device)
        encoded_images = encoder(images, watermarks)
        for j in range(len(images)):
            wm = "".join([str(x) for x in watermarks[j].int().detach().cpu().tolist()])
            dummy_wm = torch.ones(num_bits)
            dummy_wm[:] = 2
            dummy_wm = "".join([str(x) for x in dummy_wm.int().tolist()])
            save_image(images[j], f"{new_images_path}/original_image_{i}_{j}_{dummy_wm}.png")
            save_image(encoded_images[j], f"{new_images_path}/encoded_image_{i}_{j}_{wm}.png")