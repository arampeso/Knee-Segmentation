"""Training and validation loops for binary X-ray segmentation."""

import os

import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from tqdm import tqdm


def _get_batch(batch, device):
    """Move a segmentation batch to the selected device."""
    inputs = batch["xray"].to(device=device)
    labels = batch["mask"].to(device=device)
    return inputs, labels


def validate_model(model, val_loader, criterion, device, progress=None, train_loss=None):
    """Return mean loss, pixel-level Dice, and pixel-level IoU."""
    model.eval()
    running_loss = 0.0
    sample_count = 0
    dice_metric = BinaryF1Score(threshold=0.5).to(device)  # F1 == Dice
    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = _get_batch(batch, device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

            probabilities = torch.sigmoid(outputs)
            targets = (labels >= 0.5).int()
            dice_metric.update(probabilities, targets)
            iou_metric.update(probabilities, targets)

            if progress is not None:
                progress.set_postfix(
                    train_loss="n/a" if train_loss is None else f"{train_loss:.4f}",
                    val_loss=f"{running_loss / sample_count:.4f}",
                )
                progress.update(1)

    return running_loss / sample_count, dice_metric.compute().item(), iou_metric.compute().item()


def train_and_validate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    split,
    device,
    out_dir,
):
    """Train the model, validate each epoch, and save the lowest-loss checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    checkpoint_path = os.path.join(out_dir, "best_model.pth")
    progress = tqdm(
        total=epochs * (len(train_loader) + len(val_loader)),
        desc="Training",
        leave=True,
        position=0,
    )

    for epoch in range(1, epochs + 1):
        progress.set_description(f"Epoch {epoch}/{epochs}")
        last_val_loss = history["val_loss"][-1] if history["val_loss"] else None

        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:
            inputs, labels = _get_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

            progress.set_postfix(
                train_loss=f"{running_loss / sample_count:.4f}",
                val_loss="n/a" if last_val_loss is None else f"{last_val_loss:.4f}",
            )
            progress.update(1)

        train_loss = running_loss / sample_count
        val_loss, val_dice, val_iou = validate_model(
            model, val_loader, criterion, device, progress, train_loss
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    progress.close()

    return {
        "split": split,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
        "history": history,
    }