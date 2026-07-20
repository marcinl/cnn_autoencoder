"""Training and validation loops."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """
    One full training epoch.

    Args:
        model:     CNNAutoencoder instance.
        loader:    DataLoader yielding (images, *) batches, images in [0,1].
        optimizer: Any torch optimiser (Adam recommended).
        device:    cuda / mps / cpu.
        loss_fn:   Defaults to MSELoss.  BCELoss also works well with Sigmoid output.
        scaler:    Optional AMP GradScaler for mixed-precision training.

    Returns:
        Mean loss over all batches.
    """
    model.train()
    loss_fn = loss_fn or nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                recon, _ = model(imgs)
                loss = loss_fn(recon, imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon, _ = model(imgs)
            loss = loss_fn(recon, imgs)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> float:
    """Validation epoch, no gradient computation."""
    model.eval()
    loss_fn = loss_fn or nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        recon, _ = model(imgs)
        total_loss += loss_fn(recon, imgs).item()

    return total_loss / len(loader)
