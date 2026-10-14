from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
) -> float:
    """Train the model for one epoch.

    Returns:
        Average training loss per valid pixel.
    """
    model.train()

    total_loss = 0.0
    total_pixels = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        valid_pixels = (
                masks != criterion.ignore_index
        ).sum().item()

        total_loss += loss.item() * valid_pixels
        total_pixels += valid_pixels

    if total_pixels == 0:
        raise ValueError("No valid training pixels found")

    return total_loss / total_pixels


@torch.inference_mode()
def validate_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> float:
    """Evaluate the average validation loss."""
    model.eval()

    total_loss = 0.0
    total_pixels = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        valid_pixels = (
                masks != criterion.ignore_index
        ).sum().item()

        total_loss += loss.item() * valid_pixels
        total_pixels += valid_pixels

    if total_pixels == 0:
        raise ValueError("No valid validation pixels found")

    return total_loss / total_pixels


def save_checkpoint(
        model: nn.Module,
        path: str | Path,
) -> None:
    """Save model weights (state_dict only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)
