from torch.utils.data import Dataset
import numpy as np
from PIL import Image


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

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray = self.df["xrays"].iloc[index]
        mask = self.df["masks"].iloc[index]

        xray_image = read_xray(xray)
        mask_image = read_mask(mask)

        res = {
            "xray": xray_image,
            "mask": mask_image
        }

        return res