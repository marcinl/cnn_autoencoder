"""Shape, compression and gradient-flow tests for the model."""

import pytest
import torch

from cnn_autoencoder.model import (
    INPUT_ELEMENTS,
    SPATIAL_LATENT_ELEMENTS,
    CNNAutoencoder,
    ClassifierHead,
    MultiTaskAutoencoder,
)

BATCH = 2
NUM_CLASSES = 11


@pytest.fixture
def images() -> torch.Tensor:
    return torch.rand(BATCH, 3, 128, 128)


# ---------------------------------------------------------------------------
# Plain autoencoder — unchanged public behaviour
# ---------------------------------------------------------------------------

def test_spatial_autoencoder_roundtrip(images):
    model = CNNAutoencoder()
    recon, latent = model(images)
    assert latent.shape == (BATCH, 512, 7, 7)
    assert recon.shape == images.shape
    assert model.compression_ratio() == pytest.approx(INPUT_ELEMENTS / SPATIAL_LATENT_ELEMENTS)


def test_flat_autoencoder_roundtrip(images):
    model = CNNAutoencoder(latent_dim=256)
    recon, latent = model(images)
    assert latent.shape == (BATCH, 256)
    assert recon.shape == images.shape
    assert model.compression_ratio() == pytest.approx(INPUT_ELEMENTS / 256)


def test_sigmoid_output_is_in_unit_range(images):
    recon, _ = CNNAutoencoder(latent_dim=64)(images)
    assert recon.min() >= 0.0 and recon.max() <= 1.0


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("latent_dim", [None, 128, 256])
def test_multitask_forward_shapes(images, latent_dim):
    model = MultiTaskAutoencoder(num_classes=NUM_CLASSES, latent_dim=latent_dim)
    recon, latent, logits = model(images)

    assert recon.shape == images.shape
    assert logits.shape == (BATCH, NUM_CLASSES)
    expected = (BATCH, 512, 7, 7) if latent_dim is None else (BATCH, latent_dim)
    assert latent.shape == expected


def test_predict_returns_normalised_probabilities(images):
    model = MultiTaskAutoencoder(num_classes=NUM_CLASSES).eval()
    preds, probs = model.predict(images)

    assert preds.shape == (BATCH,)
    assert probs.shape == (BATCH, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH), atol=1e-5)
    assert torch.equal(preds, probs.argmax(dim=1))


def test_classifier_head_pools_spatial_latent():
    """A spatial latent must be pooled, not flattened, to keep the head small."""
    head = ClassifierHead(NUM_CLASSES, latent_dim=None)
    logits = head(torch.rand(BATCH, 512, 7, 7))
    assert logits.shape == (BATCH, NUM_CLASSES)
    assert sum(p.numel() for p in head.parameters()) < 200_000


def test_gradients_reach_encoder_from_both_heads(images):
    """
    Both losses must actually train the shared encoder — a detached branch would
    silently reduce this to two independent models.
    """
    model = MultiTaskAutoencoder(num_classes=NUM_CLASSES, latent_dim=64)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    first_encoder_weight = model.encoder.stage1[0][0].weight

    recon, _, logits = model(images)
    torch.nn.functional.mse_loss(recon, images).backward(retain_graph=True)
    recon_grad = first_encoder_weight.grad.clone()
    assert recon_grad.abs().sum() > 0

    model.zero_grad()
    _, _, logits = model(images)
    torch.nn.functional.cross_entropy(logits, labels).backward()
    assert first_encoder_weight.grad.abs().sum() > 0


def test_encode_decode_are_separable(images):
    """The latent must be usable on its own (retrieval, clustering, storage)."""
    model = MultiTaskAutoencoder(num_classes=NUM_CLASSES, latent_dim=32).eval()
    with torch.no_grad():
        z = model.encode(images)
        assert model.decode(z).shape == images.shape
        assert model.classify(z).shape == (BATCH, NUM_CLASSES)


def test_parameter_counts_sum_to_total():
    counts = MultiTaskAutoencoder(num_classes=NUM_CLASSES).count_parameters()
    assert counts["encoder"] + counts["decoder"] + counts["classifier"] == counts["total"]
