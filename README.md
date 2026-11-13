# Knee Segmentation

This project was developed as part of the **Machine Vision and Convolutional Neural Networks** course. It segments knee X-ray images by predicting a binary segmentation mask for each image.

## Project structure

```text
.
├── data/
│   ├── xrays/             # X-ray images
│   ├── masks/             # corresponding segmentation masks
│   └── csv/               # generated dataset splits
│
├── src/
│   ├── args.py            # command-line arguments
│   ├── dataset.py         # image and mask loading and preprocessing
│   ├── model.py           # U-Net model
│   ├── trainer.py         # training and validation loops
│   └── utils.py           # device selection, metrics, and loss plot
│
├── sessions/              # training and evaluation outputs
│   ├── best_model.pth     # best model checkpoint
│   ├── training_metrics.csv
│   ├── loss_curve_split_train.png
│   └── mask_overlays/     # predicted overlays for test images
│
├── data_preperation.py    # generate CSV files and dataset splits
├── main.py                # start model training
├── requirements.txt
└── evaluate.ipynb         # evaluation notebook
```

## Overview

- **Model:** U-Net with a ResNet-34 encoder
- **Input:** grayscale X-ray image with a single image channel
- **Output:** one logit per pixel; apply sigmoid to obtain probabilities
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** AdamW
- **Metrics:** pixel-level Dice score and IoU on the validation dataset
- **Device selection:** automatically uses MPS, CUDA, or CPU, depending on availability

## Installation

Clone the repository and change into the project directory. Then create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment instead with:

```powershell
.venv\Scripts\activate
```

## Prepare the data

Place the images and masks in `data/xrays/` and `data/masks/`, respectively. Each labeled X-ray must have a mask with the **same filename**. For example:

```text
data/
├── xrays/
│   ├── 0.png
│   └── 1.png
└── masks/
    ├── 0.png
    └── 1.png
```

Run the preparation script from the project directory:

```bash
python data_preperation.py
```

The script creates `data/csv/dataset.csv` and splits the data:

- Labeled images are split into **80% training** and **20% validation**, using a fixed random seed.
- Images without a corresponding mask are assigned to the **test set**.
- The splits are saved as `data/csv/train.csv`, `val.csv`, and `test.csv`.

Each CSV file contains the columns `xrays` and `masks`, which point to the respective image files. `main.py` uses the training and validation CSV files, so the training data must include masks.

Images are loaded as grayscale. Transparent regions are composited onto a black background. X-ray pixel values are normalized to `[0, 1]`; masks are converted to binary values `0` and `1` using the condition `pixel value > 0`.

## Training

Start training with the default settings:

```bash
python main.py
```

The default arguments are:

| Argument | Default | Description |
|---|---:|---|
| `-csv_dir` | `data/csv` | Directory containing `train.csv` and `val.csv` |
| `-out_dir` | `sessions` | Directory for the checkpoint and training outputs |
| `-epochs` | `30` | Number of epochs |
| `-bs` | `16` | Batch size |
| `-lr` | `0.0001` | Learning rate for AdamW |
| `-wd` | `0.00001` | Weight decay for AdamW |

You can override arguments when starting the script. For example:

```bash
python main.py -epochs 50 -bs 8 -lr 0.001 -wd 0.00001 -out_dir sessions/experiment_1
```

The model is evaluated on the validation dataset after every epoch. A `StepLR` scheduler halves the learning rate every 25 epochs (`gamma=0.5`).

## Outputs

The output directory (default: `sessions/`) contains:

| File | Contents |
|---|---|
| `best_model.pth` | Model weights from the epoch with the lowest validation loss |
| `training_metrics.csv` | Training and validation loss, plus validation Dice and IoU for each epoch |
| `loss_curve_split_train.png` | Plot of training and validation loss |
| `mask_overlays/` | Predicted mask overlays for the test images, saved by `evaluate.ipynb` |

Run the cells in `evaluate.ipynb` to generate the overlays. Each PNG in `sessions/mask_overlays/` uses the corresponding test X-ray's filename stem, so the image index is preserved.

The checkpoint contains the model's `state_dict`, not the complete model object or the optimizer state. To use it later, instantiate the same model (`MyModel`) and load the saved `state_dict`.

## Metrics

- **Dice:** pixel-level F1 score. Values closer to `1` indicate greater overlap between the prediction and the ground-truth mask.
- **IoU:** pixel-level Intersection over Union. As with Dice, values closer to `1` indicate greater overlap.
- For both metrics, prediction logits are first converted to probabilities with sigmoid and then binarized at a threshold of `0.5`.


#
###
*Aram Cheechow Peso*