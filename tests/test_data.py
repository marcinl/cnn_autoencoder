"""Dataset scanning, splitting and imbalance-handling tests."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from cnn_autoencoder.data import (
    ImageClassFolder,
    build_datasets,
    scan_class_folders,
    stratified_split,
    verify_images,
)

# Mirrors the real corpus: imbalanced classes, mixed extensions, a stray CSV.
CLASS_SIZES = {"apparel": 20, "landmark": 40, "toys": 6}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "128x128"
    for class_name, count in CLASS_SIZES.items():
        folder = root / class_name
        folder.mkdir(parents=True)
        for i in range(count):
            suffix = [".jpg", ".jpeg", ".png"][i % 3]
            Image.new("RGB", (128, 128), color=(i, i, i)).save(folder / f"image{i:04d}{suffix}")
    (root / "train.csv").write_text("image_name,label\n")
    return root


def test_scan_finds_all_images_and_ignores_csv(corpus):
    class_names, samples = scan_class_folders(corpus)
    assert class_names == sorted(CLASS_SIZES)          # sorted → stable label indices
    assert len(samples) == sum(CLASS_SIZES.values())
    assert not any(path.suffix == ".csv" for path, _ in samples)


def test_scan_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_class_folders(tmp_path / "does-not-exist")


def test_split_is_stratified_and_leak_free(corpus):
    class_names, samples = scan_class_folders(corpus)
    train, val = stratified_split(samples, len(class_names), val_fraction=0.25, seed=1)

    assert len(train) + len(val) == len(samples)
    assert set(train).isdisjoint(val)
    assert {p for p, _ in train}.isdisjoint({p for p, _ in val})

    for index, name in enumerate(class_names):
        expected_val = round(CLASS_SIZES[name] * 0.25)
        assert sum(1 for _, y in val if y == index) == expected_val
        assert sum(1 for _, y in train if y == index) == CLASS_SIZES[name] - expected_val


def test_split_is_deterministic_for_a_seed(corpus):
    class_names, samples = scan_class_folders(corpus)
    first = stratified_split(samples, len(class_names), seed=7)
    second = stratified_split(samples, len(class_names), seed=7)
    third = stratified_split(samples, len(class_names), seed=8)

    assert first == second
    assert first != third


def test_smallest_class_always_gets_a_val_sample(corpus):
    """Rounding must not leave a rare class unrepresented in validation."""
    class_names, samples = scan_class_folders(corpus)
    _, val = stratified_split(samples, len(class_names), val_fraction=0.01, seed=1)
    toys_index = sorted(CLASS_SIZES).index("toys")
    assert sum(1 for _, y in val if y == toys_index) >= 1


def test_per_class_cap_subsets_every_class(corpus):
    class_names, samples = scan_class_folders(corpus)
    train, val = stratified_split(samples, len(class_names), val_fraction=0.25,
                                  seed=1, per_class_cap=5)
    assert len(train) + len(val) == 5 * len(class_names)


def test_dataset_yields_unit_range_tensors(corpus):
    train_ds, _, class_names = build_datasets(corpus, seed=3)
    image, label = train_ds[0]

    assert image.shape == (3, 128, 128)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0   # decoder ends in Sigmoid
    assert 0 <= label < len(class_names)


def test_class_weights_favour_the_rare_class(corpus):
    train_ds, _, class_names = build_datasets(corpus, seed=3)
    weights = train_ds.class_weights()
    order = {name: i for i, name in enumerate(class_names)}

    assert weights[order["toys"]] > weights[order["apparel"]] > weights[order["landmark"]]
    assert weights.mean() == pytest.approx(1.0, abs=1e-5)
    assert len(train_ds.sample_weights()) == len(train_ds)


def test_corrupt_image_is_skipped_not_fatal(corpus):
    broken = corpus / "toys" / "image0000.jpg"
    broken.write_bytes(b"not an image")

    _, samples = scan_class_folders(corpus)
    dataset = ImageClassFolder(samples, sorted(CLASS_SIZES))
    index = next(i for i, (path, _) in enumerate(samples) if path == broken)

    image, _ = dataset[index]           # substitutes the next readable sample
    assert image.shape == (3, 128, 128)
    assert broken in dataset.failed
    assert verify_images(corpus) == [broken]
