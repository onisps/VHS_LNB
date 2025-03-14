import os
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils.global_variables import PATCH_SIZE

class PatchDataset(Dataset):
    """
    Loads patches (or resized images) for CycleGAN training.
    Incorporates data augmentations (flip, color jitter) to
    encourage more robust color transformations.
    """
    def __init__(self, source_dir: str, target_dir: str,
                 image_size: Tuple[int, int] = (PATCH_SIZE, PATCH_SIZE)):
        self.source_paths = sorted([
            os.path.join(source_dir, f)
            for f in os.listdir(source_dir)
            if f.endswith((".png", ".jpg"))
        ])
        self.target_paths = sorted([
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.endswith((".png", ".jpg"))
        ])

        # Example augmentation + normalization pipeline:
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            # Normalizing to [-1, 1] for compatibility with Tanh in the generator
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        # CycleGAN can work with unpaired sets, but we typically take the min
        return min(len(self.source_paths), len(self.target_paths))

    def __getitem__(self, idx):
        source_img = Image.open(self.source_paths[idx]).convert("RGB")
        target_img = Image.open(self.target_paths[idx]).convert("RGB")

        return self.transform(source_img), self.transform(target_img)


def get_dataloader(source_dir: str, target_dir: str,
                   batch_size: int = 8,
                   image_size: Tuple[int, int] = (PATCH_SIZE, PATCH_SIZE)):
    """
    Creates a DataLoader for CycleGAN training with the given batch size.
    """
    dataset = PatchDataset(source_dir, target_dir, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
