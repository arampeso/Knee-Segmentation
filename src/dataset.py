from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2


def _read_grayscale(path):
    """Load an image as a 2D grayscale array; transparent pixels become black."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        grayscale = Image.alpha_composite(background, rgba).convert("L")
        return np.asarray(grayscale)


def read_xray(path):
    """Load an X-ray as a float32 array of shape (1, H, W), scaled to [0, 1]."""
    image = _read_grayscale(path).astype(np.float32)  # H W
    xray = image / 255.0  # normalize (0,1)
    xray = xray.reshape(1, *xray.shape)  # C H W = 1 H W (grayscale)
    return xray


def read_mask(path):
    """Load a segmentation mask as a binary float32 array of shape (1, H, W)."""
    image = _read_grayscale(path)
    mask = (image > 0).astype(np.float32)  # binary mask (0,1)
    mask = mask.reshape(1, *mask.shape)  # C H W = 1 H W (binary)
    return mask


class KneeDataset(Dataset):
    """Dataset of knee X-rays and their segmentation masks, based on a DataFrame."""

    def __init__(self, dataset, should_augment=False):
        self.dataset = dataset
        self.should_augment = should_augment

        self.augmentation = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(
                degrees=7,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        xray = self.dataset["xrays"].iloc[index]
        mask = self.dataset["masks"].iloc[index]

        xray_image = torch.from_numpy(read_xray(xray))
        mask_image = torch.from_numpy(read_mask(mask))

        if self.should_augment:
            xray_image, mask_image = self.augmentation(
                tv_tensors.Image(xray_image),
                tv_tensors.Mask(mask_image),
            )
            xray_image = xray_image.as_subclass(torch.Tensor)
            mask_image = mask_image.as_subclass(torch.Tensor)

        res = {
            "xray": xray_image,
            "mask": mask_image
        }

        return res