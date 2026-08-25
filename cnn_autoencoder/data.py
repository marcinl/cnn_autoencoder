"""
Dataset and loaders for the ./128x128 image corpus.

Layout expected:

    128x128/
        apparel/     image0000.jpg ...
        artwork/     ...
        ...
        train.csv    (ignored)

The folder name is the label. `train.csv` is deliberately ignored: it carries no
information the tree does not already have, and its `image_name` column is not
unique across folders (48 814 distinct names for 132 528 rows), so it cannot be
joined back to a file path unambiguously.

Pixel range: images are loaded to [0, 1] via ToTensor and are NOT mean/std
normalised. The decoder ends in a Sigmoid, so its output lives in [0, 1]; the
reconstruction target has to live in the same range for MSE to make sense.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

Sample = tuple[Path, int]


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_class_folders(root: str | Path) -> tuple[list[str], list[Sample]]:
    """
    Walk `root` and return (class_names, samples).

    Class names are sorted for a stable label ordering across runs — the index a
    class maps to must not depend on filesystem iteration order, or a checkpoint
    trained today will mispredict when reloaded tomorrow.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not class_names:
        raise FileNotFoundError(f"No class sub-directories under {root}")

    samples: list[Sample] = []
    for idx, name in enumerate(class_names):
        for path in sorted((root / name).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, idx))

    if not samples:
        raise FileNotFoundError(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} under {root}")

    return class_names, samples


def stratified_split(
    samples: Sequence[Sample],
    num_classes: int,
    val_fraction: float = 0.1,
    seed: int = 42,
    per_class_cap: Optional[int] = None,
) -> tuple[list[Sample], list[Sample]]:
    """
    Split per class so both sides keep the full class distribution.

    A plain random split would be adequate for the large classes but can starve
    the small ones (toys has only 2 402 images against landmark's 33 063).

    Args:
        per_class_cap: Keep at most this many images per class before splitting.
            Used for fast end-to-end smoke tests on a stratified subset.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_class: list[list[Sample]] = [[] for _ in range(num_classes)]
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []

    for class_samples in by_class:
        shuffled = sorted(class_samples)          # stable input order
        rng.shuffle(shuffled)                     # deterministic given seed
        if per_class_cap is not None:
            shuffled = shuffled[:per_class_cap]

        n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageClassFolder(Dataset):
    """
    Yields (image_tensor, class_index) with image_tensor in [0, 1].

    Corrupt files are tolerated rather than fatal: at this corpus size a single
    unreadable JPEG should not kill an epoch that is hours in. A failed decode
    substitutes the next readable sample and records the path in `self.failed`.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        class_names: Sequence[str],
        transform: Optional[Callable] = None,
    ):
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform or transforms.ToTensor()
        self.failed: set[Path] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        for offset in range(len(self.samples)):
            path, label = self.samples[(index + offset) % len(self.samples)]
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
            except (OSError, UnidentifiedImageError, ValueError):
                self.failed.add(path)
                continue
            return self.transform(image), label

        raise RuntimeError("No readable image found in the dataset")

    def class_counts(self) -> list[int]:
        counts = Counter(label for _, label in self.samples)
        return [counts[i] for i in range(len(self.class_names))]

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for CrossEntropyLoss, normalised to mean 1.

        The corpus is imbalanced 13.8:1 (landmark 33 063 vs toys 2 402); without
        weighting the model can reach ~25% accuracy by always guessing landmark.
        """
        counts = torch.tensor(self.class_counts(), dtype=torch.float)
        weights = counts.sum() / counts.clamp(min=1)
        return weights / weights.mean()

    def sample_weights(self) -> torch.Tensor:
        """Per-sample weights for a WeightedRandomSampler (balanced batches)."""
        weights = self.class_weights()
        return torch.tensor([weights[label] for _, label in self.samples])


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(augment: bool = True) -> tuple[Callable, Callable]:
    """
    Returns (train_transform, eval_transform).

    Augmentation is limited to a horizontal flip. Colour jitter and crops are
    left out on purpose: the reconstruction target is the *augmented* tensor, so
    any augmentation that destroys information (heavy crops) or shifts colour
    also moves the target the decoder is chasing.
    """
    eval_tf = transforms.ToTensor()
    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------

def build_datasets(
    root: str | Path = "128x128",
    val_fraction: float = 0.1,
    seed: int = 42,
    per_class_cap: Optional[int] = None,
    augment: bool = True,
) -> tuple[ImageClassFolder, ImageClassFolder, list[str]]:
    """Returns (train_dataset, val_dataset, class_names)."""
    class_names, samples = scan_class_folders(root)
    train_samples, val_samples = stratified_split(
        samples, len(class_names), val_fraction, seed, per_class_cap
    )
    train_tf, eval_tf = build_transforms(augment)
    return (
        ImageClassFolder(train_samples, class_names, train_tf),
        ImageClassFolder(val_samples, class_names, eval_tf),
        class_names,
    )


def build_loaders(
    train_ds: ImageClassFolder,
    val_ds: ImageClassFolder,
    batch_size: int = 64,
    num_workers: int = 4,
    balanced_sampling: bool = False,
    pin_memory: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).

    Args:
        balanced_sampling: Draw class-balanced batches with replacement instead
            of weighting the loss. Use one or the other, not both, or the
            imbalance correction is applied twice.
    """
    sampler = None
    if balanced_sampling:
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            train_ds.sample_weights(),
            num_samples=len(train_ds),
            replacement=True,
            generator=generator,
        )

    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,          # keeps BatchNorm from seeing a size-1 tail batch
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader


def verify_images(root: str | Path = "128x128") -> list[Path]:
    """
    Decode every image once and return the paths that fail.

    Slow (a full pass over the corpus) but worth running once before a long
    training job so failures surface up front instead of mid-epoch.
    """
    _, samples = scan_class_folders(root)
    bad: list[Path] = []
    for path, _ in samples:
        try:
            with Image.open(path) as img:
                img.convert("RGB").load()
        except (OSError, UnidentifiedImageError, ValueError):
            bad.append(path)
    return bad
