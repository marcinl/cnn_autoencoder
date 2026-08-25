"""
Training and validation loops.

`train_epoch` / `val_epoch`      — plain autoencoder (reconstruction only).
`train_epoch_multitask` / `val_epoch_multitask`
                                 — MultiTaskAutoencoder (reconstruction + labels).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from .metrics import ConfusionMatrix


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


# ---------------------------------------------------------------------------
# Multi-task loops (reconstruction + single-label classification)
# ---------------------------------------------------------------------------
#
# Loss:  total = (1 - alpha) * MSE(recon, x)  +  alpha * CE(logits, y)
#
# Caveat on alpha: the two terms are not on the same scale. Pixel MSE on [0, 1]
# images settles around 0.01-0.05, while cross-entropy over 11 classes starts at
# ln(11) = 2.40. At alpha=0.5 the classification term therefore dominates the
# gradient by roughly 50x. That is usually the right bias if labels are the goal,
# but if you want reconstruction to actually improve, alpha needs to go well
# below 0.5 (try 0.05-0.2) rather than the 0.5 a naive reading would suggest.
# Watch the two reported components, not just the total.


def _multitask_losses(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    recon_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, recon_loss, cls_loss, logits)."""
    recon, _, logits = model(imgs)
    loss_recon = recon_loss_fn(recon, imgs)
    loss_cls = cls_loss_fn(logits, labels)
    total = (1.0 - alpha) * loss_recon + alpha * loss_cls
    return total, loss_recon, loss_cls, logits


def train_epoch_multitask(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,
    recon_loss_fn: Optional[nn.Module] = None,
    cls_loss_fn: Optional[nn.Module] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    progress: Optional[object] = None,
) -> dict[str, float]:
    """
    One training epoch for a MultiTaskAutoencoder.

    Args:
        model:         MultiTaskAutoencoder, forward returns (recon, latent, logits).
        loader:        Yields (images in [0,1], integer class labels).
        alpha:         Balance between the two losses — see the note above.
        recon_loss_fn: Defaults to MSELoss.
        cls_loss_fn:   Defaults to CrossEntropyLoss. Pass one constructed with
                       `weight=train_ds.class_weights()` to correct the 13.8:1
                       class imbalance.
        scaler:        Optional AMP GradScaler (CUDA only).
        progress:      Optional tqdm instance to update per batch.

    Returns:
        Mean loss components and running top-1 accuracy for the epoch.
    """
    model.train()
    recon_loss_fn = recon_loss_fn or nn.MSELoss()
    cls_loss_fn = cls_loss_fn or nn.CrossEntropyLoss()

    totals = {"loss": 0.0, "recon": 0.0, "cls": 0.0}
    correct = 0
    seen = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                loss, loss_recon, loss_cls, logits = _multitask_losses(
                    model, imgs, labels, recon_loss_fn, cls_loss_fn, alpha
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_recon, loss_cls, logits = _multitask_losses(
                model, imgs, labels, recon_loss_fn, cls_loss_fn, alpha
            )
            loss.backward()
            optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += loss_recon.item()
        totals["cls"] += loss_cls.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

        if progress is not None:
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / seen:.3f}")

    batches = max(len(loader), 1)
    return {
        "loss": totals["loss"] / batches,
        "recon": totals["recon"] / batches,
        "cls": totals["cls"] / batches,
        "acc": correct / max(seen, 1),
    }


@torch.no_grad()
def val_epoch_multitask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float = 0.5,
    recon_loss_fn: Optional[nn.Module] = None,
    cls_loss_fn: Optional[nn.Module] = None,
    confusion: Optional[ConfusionMatrix] = None,
) -> dict[str, float]:
    """
    Validation epoch for a MultiTaskAutoencoder.

    Args:
        confusion: Optional ConfusionMatrix, updated in place so the caller can
            print a per-class breakdown afterwards.
    """
    model.eval()
    recon_loss_fn = recon_loss_fn or nn.MSELoss()
    cls_loss_fn = cls_loss_fn or nn.CrossEntropyLoss()

    totals = {"loss": 0.0, "recon": 0.0, "cls": 0.0}
    correct = 0
    seen = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss, loss_recon, loss_cls, logits = _multitask_losses(
            model, imgs, labels, recon_loss_fn, cls_loss_fn, alpha
        )

        totals["loss"] += loss.item()
        totals["recon"] += loss_recon.item()
        totals["cls"] += loss_cls.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

        if confusion is not None:
            confusion.update(logits, labels)

    batches = max(len(loader), 1)
    return {
        "loss": totals["loss"] / batches,
        "recon": totals["recon"] / batches,
        "cls": totals["cls"] / batches,
        "acc": correct / max(seen, 1),
    }
