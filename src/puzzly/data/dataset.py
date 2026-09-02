import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2


def get_grid_positions(img_size=128, grid_size=3):
    """Fixed 3x3 grid cell coordinates - same every time, regardless of permutation."""
    patch_size = img_size // grid_size
    positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * patch_size
            y = row * patch_size
            positions.append((x, y))
    return positions


def apply_squircle_mask(patch, roundness=4, corner_size=0.3):
    """Rounds the corners of a square patch (less aggressive than a true circle),
    removing the hard straight-edge shortcut while retaining more content than a circle."""
    w, h = patch.size
    arr = np.array(patch)

    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2

    nx = np.abs(x - cx) / (w / 2)
    ny = np.abs(y - cy) / (h / 2)
    dist = (nx ** roundness + ny ** roundness) ** (1 / roundness)

    threshold = 1.0 - corner_size * 0.3
    mask = (dist <= threshold).astype(np.uint8) * 255

    result = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        result[:, :, c] = np.where(mask == 255, arr[:, :, c], 0)

    return Image.fromarray(result)


class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir, indices=None):
        self.root_dir = root_dir
        with open(f"{root_dir}/labels.json") as f:
            self.labels = json.load(f)
        self.indices = indices if indices is not None else list(range(len(self.labels)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = Image.open(f"{self.root_dir}/imagenet100_128px/imagenet100_128px/{real_idx}.jpg").convert("RGB")
        label = self.labels[real_idx]
        return img, label


class JigsawDataset(Dataset):
    def __init__(self, image_dataset, permutation_set, transform=None, seed=None, use_mask=True):
        self.image_dataset = image_dataset
        self.permutation_set = permutation_set
        self.transform = transform
        self.rng = random.Random(seed)
        self.grid_positions = get_grid_positions()
        self.use_mask = use_mask
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img, label = self.image_dataset[idx]

        patch_size = 128 // 3
        patches = [
            img.crop((x, y, x + patch_size, y + patch_size))
            for (x, y) in self.grid_positions
        ]

        if self.use_mask:
            patches = [apply_squircle_mask(p) for p in patches]

        perm_index = self.rng.randint(0, len(self.permutation_set) - 1)
        perm = self.permutation_set[perm_index]
        shuffled_patches = [patches[i] for i in perm]

        shuffled_patches = [self.to_tensor(patch) for patch in shuffled_patches]

        if self.transform:
            shuffled_patches = [self.transform(patch) for patch in shuffled_patches]

        shuffled_patches = torch.stack(shuffled_patches)
        return shuffled_patches, perm_index