# Knee Segmentation

Binary segmentation of knee X-ray images with a U-Net. Developed for the course **Machine Vision and Convolutional Neural Networks**.

| | |
|---|---|
| **Model** | U-Net with ResNet-34 encoder |
| **Input** | Grayscale X-ray, 1 channel, values in `[0, 1]` |
| **Output** | One logit per pixel (apply sigmoid for probabilities) |
| **Loss / Optimizer** | `BCEWithLogitsLoss` / AdamW, `StepLR` (halves LR every 25 epochs) |
| **Metrics** | Pixel-level Dice and IoU on the validation set |
| **Device** | Automatic: MPS, CUDA or CPU |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python data_preperation.py         # create CSV splits
python main.py                     # train
```

Then run `evaluate.ipynb` to generate the test overlays.

## Project structure

```text
.
├── data/
│   ├── xrays/                 # X-ray images
│   ├── masks/                 # masks (same filenames as the images)
│   └── csv/                   # generated splits
├── src/
│   ├── args.py                # command-line arguments
│   ├── dataset.py             # loading and preprocessing
│   ├── model.py               # U-Net
│   ├── trainer.py             # training and validation loops
│   └── utils.py               # device, metrics, loss plot
├── sessions/                  # outputs (see below)
├── data_preperation.py        # create dataset splits
├── main.py                    # start training
├── evaluate.ipynb             # evaluation notebook
└── requirements.txt
```

## Data

Put images in `data/xrays/` and masks in `data/masks/`. Each labeled image needs a mask with the **same filename** (e.g. `0.png` in both folders).

`data_preperation.py` writes `data/csv/dataset.csv` and splits it:

| Split | Content | File |
|---|---|---|
| Train | 80 % of labeled images (fixed seed) | `train.csv` |
| Validation | 20 % of labeled images | `val.csv` |
| Test | Images without a mask | `test.csv` |

Each CSV has the columns `xrays` and `masks`. `main.py` only uses `train.csv` and `val.csv`.

**Preprocessing:** images are loaded as grayscale, transparent areas are composited onto black, X-rays are scaled to `[0, 1]`, and masks are binarized with `pixel value > 0`.

## Training

```bash
python main.py
```

| Argument | Default | Description |
|---|---:|---|
| `-csv_dir` | `data/csv` | Folder with `train.csv` and `val.csv` |
| `-out_dir` | `sessions` | Output folder |
| `-epochs` | `30` | Number of epochs |
| `-bs` | `16` | Batch size |
| `-lr` | `0.0001` | Learning rate (AdamW) |
| `-wd` | `0.00001` | Weight decay (AdamW) |

Example with custom settings:

```bash
python main.py -epochs 50 -bs 8 -lr 0.001 -out_dir sessions/experiment_1
```

The model is validated after every epoch.

## Outputs

| File | Content |
|---|---|
| `best_model.pth` | `state_dict` of the epoch with the lowest validation loss |
| `training_metrics.csv` | Train/val loss, val Dice and IoU per epoch |
| `loss_curve_split_train.png` | Loss plot |
| `mask_overlays/` | Predicted overlays for test images (from `evaluate.ipynb`), named after the image filename |

The checkpoint contains only the `state_dict` (no full model, no optimizer state). To reuse it, create a `MyModel` instance and load the weights.

## Metrics

Logits are passed through sigmoid and binarized at `0.5`.

- **Dice:** pixel-level F1 score
- **IoU:** pixel-level Intersection over Union

Both range from 0 to 1; higher means more overlap with the ground truth.

---

*Aram Cheechow Peso*