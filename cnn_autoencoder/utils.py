"""Checkpoint helpers and visualisation utilities."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def reconstruct_grid(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run a batch through the autoencoder and return a side-by-side grid
    (original | reconstruction) as a single tensor of shape [C, H, N*2*W].

    Useful for quick visual inspection with torchvision.utils.save_image.
    """
    model.eval()
    with torch.no_grad():
        recon, _ = model(images.to(device))
    recon = recon.cpu()

    pairs = torch.cat([images, recon], dim=3)   # stack width-wise per image
    # flatten batch into a single wide strip
    return torch.cat([pairs[i] for i in range(pairs.size(0))], dim=2)
