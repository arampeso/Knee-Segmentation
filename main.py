import pandas as pd
import torch
from torch.utils.data import DataLoader
import os

from src.args import get_args
from src.dataset import KneeDataset
from src.model import MyModel
from src.trainer import train_and_validate
from src.utils import get_device, save_loss_plot

device = get_device()
print(f"Training on {device}")
args = get_args()


def main():

    # --------------- 2. loading the csv files ---------------
    train_set = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
    val_set = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))

    # ---------------3. define datasets ---------------
    train_dataset = KneeDataset(train_set)
    val_dataset = KneeDataset(val_set)

    # --------------- 4. define dataloaders ---------------
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    # --------------- 5. initialize the model, loss, optimizer, and device ---------------
    model = MyModel().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = args.lr,
        weight_decay=args.wd,
    )


    # --------------- 6. train and validate ---------------
    results = train_and_validate(
        model       = model,
        train_loader= train_loader,
        val_loader  = val_loader,
        criterion   = criterion,
        optimizer   = optimizer,
        epochs      = args.epochs,
        split       = "train",
        device      = device,
        out_dir     = args.out_dir,
    )

    loss_plot_path = save_loss_plot(
        results["history"],
        args.out_dir,
        split="train",
    )

    # --------------- 7. save and print train/validation metrics ---------------
    history = results["history"]
    metrics_df = pd.DataFrame(
        {
            "epoch": range(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_dice": history["val_dice"],
            "val_iou": history["val_iou"],
        }
    )
    metrics_path = os.path.join(args.out_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    saved_metrics_df = pd.read_csv(metrics_path)
    print(f"\nTraining and validation metrics ({metrics_path}):")
    print(saved_metrics_df.to_string(index=False))
    print(f"\nBest validation epoch: {results['best_epoch']}")
    print(f"Loss plot saved to: {loss_plot_path}")
    print(f"Checkpoint saved to: {os.path.join(args.out_dir, 'best_model_split_train.pth')}")


if __name__ == "__main__":
    main()