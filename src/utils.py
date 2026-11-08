import os

import matplotlib.pyplot as plt
import pandas as pd
import torch


def get_device():
    """ Returns the appropriate device (CPU, CUDA, or MPS) based on availability. """
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        
    else:
        device = torch.device("cpu")

    return device


def save_loss_plot(history, out_dir, split="train"):
    """Save train and validation loss curves to the output directory."""
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    if not train_losses or len(train_losses) != len(val_losses):
        raise ValueError("Train and validation loss histories must be non-empty and equal length")

    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train")
    ax.plot(epochs, val_losses, marker="x", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train and validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, f"loss_curve_split_{split}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_and_print_metrics(history, out_dir, best_epoch):
    metrics_df = pd.DataFrame(
        {
            "epoch": range(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_dice": history["val_dice"],
            "val_iou": history["val_iou"],
        }
    )
    metrics_path = os.path.join(out_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    saved_metrics_df = pd.read_csv(metrics_path)
    print(f"\nTraining and validation metrics ({metrics_path}):")
    print(saved_metrics_df.to_string(index=False, max_rows=6))
    print(f"\nBest validation epoch: {best_epoch}")