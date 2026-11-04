from torch.utils.data import Dataset
import numpy as np
from PIL import Image


def _read_grayscale(path):
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        grayscale = Image.alpha_composite(background, rgba).convert("L")
        return np.asarray(grayscale)


def read_xray(path):
    image = _read_grayscale(path).astype(np.float32) # H W
    xray = image / 255.0 # normalize (0,1)
    xray = xray.reshape(1, *xray.shape) # C H W = 1 H W (grayscale)
    return xray

def read_mask(path):
    image = _read_grayscale(path)
    mask = (image > 0).astype(np.float32) # binary mask (0,1)
    mask = mask.reshape(1, *mask.shape) # C H W = 1 H W (binary)
    return mask

class KneeDataset(Dataset):
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