import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from dataset import get_image_dataloader
from torchvision.utils import save_image
from torchvision.transforms import v2

def plot2images(image1, title1, image2, title2):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    if image1.shape[0] == 3:
        image1 = image1.permute(1, 2, 0)
    if image2.shape[0] == 3:
        image2 = image2.permute(1, 2, 0)
    axes[0].imshow(image1.detach().cpu().numpy())
    axes[0].axis('off')
    axes[0].set_title(title1)
    axes[1].imshow(image2.detach().cpu().numpy())
    axes[1].axis('off')
    axes[1].set_title(title2)
    plt.tight_layout()
    plt.show()
    


def random_noise_composition(x):
    device = x.device
    x = x * 255
    x = v2.JPEG((80,100))(x.to(torch.uint8).cpu()).to(device).float()
    x /= 255
    return x