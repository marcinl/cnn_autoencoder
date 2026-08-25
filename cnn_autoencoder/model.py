"""
CNN Autoencoder: 128x128x3  →  7x7x512  →  128x128x3
With an optional flat latent bottleneck and an optional classification head
(see `MultiTaskAutoencoder`): 128x128x3 → 7x7x512 → z ∈ R^latent_dim → both a
reconstruction and class logits.

Encoder architecture (recommended hidden design):
  Each stage = Conv2d → BatchNorm2d → ReLU → (optional) ResBlock

  Stage 1: [B,   3, 128,128] → [B,  64,  64, 64]  stride-2 conv
  Stage 2: [B,  64,  64, 64] → [B, 128,  32, 32]  stride-2 conv
  Stage 3: [B, 128,  32, 32] → [B, 256,  16, 16]  stride-2 conv
  Stage 4: [B, 256,  16, 16] → [B, 512,   8,  8]  stride-2 conv
  Bottleneck: [B, 512, 8, 8] → [B, 512,   7,  7]  kernel-2, stride-1, no pad

Decoder is the exact mirror using ConvTranspose2d.

Rationale for hidden design:
  - Progressive channel doubling (64→128→256→512) increases representational
    capacity while spatial resolution halves — standard for feature hierarchy.
  - BatchNorm2d after every conv stabilises training and allows higher LR.
  - ReLU (minimum requirement) after BN; Leaky ReLU is available as an option
    via `leaky` flag for better gradient flow in the decoder.
  - ResBlocks at each spatial level are optional (use_res=True) and help
    preserve fine detail without adding parameters to the spatial reduction path.
"""

import torch
import torch.nn as nn


SPATIAL_LATENT_SHAPE = (512, 7, 7)
SPATIAL_LATENT_ELEMENTS = 512 * 7 * 7   # 25 088
INPUT_ELEMENTS = 3 * 128 * 128          # 49 152


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnAct(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU (or LeakyReLU)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        leaky: bool = False,
    ):
        act = nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
        )


class ConvTBnAct(nn.Sequential):
    """ConvTranspose2d → BatchNorm2d → ReLU (or LeakyReLU)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        leaky: bool = False,
    ):
        act = nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
        )


class ResBlock(nn.Module):
    """Two-conv residual block at fixed spatial size (no downsampling)."""

    def __init__(self, channels: int, leaky: bool = False):
        super().__init__()
        act = nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            act,
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    128x128x3  →  7x7x512  (→ optional flat latent vector)

    Spatial reduction:
      stride-2 ×4: 128→64→32→16→8
      kernel-2     : 8→7

    Args:
        latent_dim: If given, the 512x7x7 map is projected to a flat vector of
            this size. This is what turns the model into a genuine compression
            bottleneck: 512*7*7 = 25 088 floats is only 1.96x smaller than the
            49 152-float input, whereas a 256-d vector is 192x smaller. The flat
            vector is also a far better feature for the classifier head and for
            downstream retrieval than a spatial map.
    """

    def __init__(
        self,
        use_res: bool = True,
        leaky: bool = False,
        latent_dim: int | None = None,
    ):
        super().__init__()
        kw = dict(leaky=leaky)

        self.stage1 = nn.Sequential(
            ConvBnAct(  3,  64, 3, stride=2, padding=1, **kw),   # 128→64
            *(ResBlock( 64, **kw),) * (1 if use_res else 0),
        )
        self.stage2 = nn.Sequential(
            ConvBnAct( 64, 128, 3, stride=2, padding=1, **kw),   # 64→32
            *(ResBlock(128, **kw),) * (1 if use_res else 0),
        )
        self.stage3 = nn.Sequential(
            ConvBnAct(128, 256, 3, stride=2, padding=1, **kw),   # 32→16
            *(ResBlock(256, **kw),) * (1 if use_res else 0),
        )
        self.stage4 = nn.Sequential(
            ConvBnAct(256, 512, 3, stride=2, padding=1, **kw),   # 16→8
            *(ResBlock(512, **kw),) * (1 if use_res else 0),
        )
        # 8→7: kernel=2, stride=1, padding=0  →  (8-1)*1 - 0 + 2 = 9? No.
        # output = floor((in + 2p - k)/s + 1) = (8 + 0 - 2)/1 + 1 = 7  ✓
        self.bottleneck = ConvBnAct(512, 512, kernel=2, stride=1, padding=0, **kw)

        self.latent_dim = latent_dim
        self.project = (
            nn.Sequential(nn.Flatten(), nn.Linear(SPATIAL_LATENT_ELEMENTS, latent_dim))
            if latent_dim is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bottleneck(x)   # [B, 512, 7, 7]
        if self.project is not None:
            x = self.project(x)  # [B, latent_dim]
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    7x7x512  →  128x128x3

    Spatial expansion (mirror of encoder):
      ConvTranspose kernel-2: 7→8
      ConvTranspose stride-2 ×4: 8→16→32→64→128

    Args:
        latent_dim: Must match the encoder. If given, the decoder first expands
            the flat latent vector back to a 512x7x7 map.
    """

    def __init__(
        self,
        use_res: bool = True,
        leaky: bool = False,
        latent_dim: int | None = None,
    ):
        super().__init__()
        kw = dict(leaky=leaky)

        self.latent_dim = latent_dim
        self.unproject = (
            nn.Linear(latent_dim, SPATIAL_LATENT_ELEMENTS)
            if latent_dim is not None
            else None
        )

        # 7→8: output = (7-1)*1 + 2 = 8  ✓
        self.unbottleneck = ConvTBnAct(512, 512, kernel=2, stride=1, padding=0, **kw)

        self.stage4 = nn.Sequential(
            *(ResBlock(512, **kw),) * (1 if use_res else 0),
            ConvTBnAct(512, 256, kernel=4, stride=2, padding=1, **kw),  # 8→16
        )
        self.stage3 = nn.Sequential(
            *(ResBlock(256, **kw),) * (1 if use_res else 0),
            ConvTBnAct(256, 128, kernel=4, stride=2, padding=1, **kw),  # 16→32
        )
        self.stage2 = nn.Sequential(
            *(ResBlock(128, **kw),) * (1 if use_res else 0),
            ConvTBnAct(128, 64, kernel=4, stride=2, padding=1, **kw),   # 32→64
        )
        self.stage1 = nn.Sequential(
            *(ResBlock(64, **kw),) * (1 if use_res else 0),
            # Final layer: no BN, Sigmoid to map to [0, 1]
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 64→128
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.unproject is not None:
            z = self.unproject(z).view(-1, *SPATIAL_LATENT_SHAPE)
        x = self.unbottleneck(z)
        x = self.stage4(x)
        x = self.stage3(x)
        x = self.stage2(x)
        x = self.stage1(x)   # [B, 3, 128, 128]
        return x


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------

class CNNAutoencoder(nn.Module):
    """
    End-to-end autoencoder.

    Args:
        use_res:    Add one ResBlock per spatial stage for better gradient flow.
        leaky:      Use LeakyReLU(0.2) instead of ReLU throughout hidden layers.
        latent_dim: Flat bottleneck width. None keeps the 512x7x7 spatial latent.
    """

    def __init__(
        self,
        use_res: bool = True,
        leaky: bool = False,
        latent_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(use_res=use_res, leaky=leaky, latent_dim=latent_dim)
        self.decoder = Decoder(use_res=use_res, leaky=leaky, latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latent tensor: [B, latent_dim] if latent_dim is set, else [B, 512, 7, 7]."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs image from latent tensor."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        return self.decode(z), z

    def latent_elements(self) -> int:
        return self.latent_dim if self.latent_dim is not None else SPATIAL_LATENT_ELEMENTS

    def compression_ratio(self) -> float:
        """Input elements / latent elements (both float32)."""
        return INPUT_ELEMENTS / self.latent_elements()

    def count_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """
    Maps the encoder latent to class logits.

    Accepts either latent form. A spatial [B, 512, 7, 7] latent is global-average
    pooled to [B, 512] first — pooling rather than flattening keeps the head from
    dwarfing the rest of the network (25 088 -> 256 would be a 6.4M-parameter
    layer) and makes the head robust to where in the frame the object sits.
    """

    def __init__(
        self,
        num_classes: int,
        latent_dim: int | None = None,
        hidden: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.spatial = latent_dim is None
        in_features = SPATIAL_LATENT_SHAPE[0] if self.spatial else latent_dim

        self.pool = nn.AdaptiveAvgPool2d(1) if self.spatial else None
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),   # raw logits
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            z = self.pool(z).flatten(1)
        return self.mlp(z)


# ---------------------------------------------------------------------------
# Multi-task autoencoder + classifier
# ---------------------------------------------------------------------------

class MultiTaskAutoencoder(nn.Module):
    """
    Shared encoder feeding two heads: a decoder (reconstruction) and a
    classifier (single-label, softmax over `num_classes`).

    The ./128x128 dataset places every image in exactly one class folder, so the
    classification head emits logits for CrossEntropyLoss. Use
    `torch.softmax(logits, dim=1).argmax(1)` at inference, not a per-class
    sigmoid threshold.

    Args:
        num_classes: Number of mutually exclusive classes (11 for ./128x128).
        latent_dim:  Flat bottleneck width. None keeps the 512x7x7 spatial latent.
        use_res:     Add one ResBlock per spatial stage.
        leaky:       Use LeakyReLU(0.2) instead of ReLU in hidden layers.
        hidden:      Width of the classifier's hidden layer.
        dropout:     Classifier dropout probability.
    """

    def __init__(
        self,
        num_classes: int,
        latent_dim: int | None = 256,
        use_res: bool = True,
        leaky: bool = False,
        hidden: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder = Encoder(use_res=use_res, leaky=leaky, latent_dim=latent_dim)
        self.decoder = Decoder(use_res=use_res, leaky=leaky, latent_dim=latent_dim)
        self.classifier = ClassifierHead(
            num_classes, latent_dim=latent_dim, hidden=hidden, dropout=dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latent tensor: [B, latent_dim] if latent_dim is set, else [B, 512, 7, 7]."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Class logits from an existing latent — no re-encoding."""
        return self.classifier(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent, class_logits)."""
        z = self.encode(x)
        return self.decode(z), z, self.classify(z)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted class indices, class probabilities)."""
        probs = torch.softmax(self.classify(self.encode(x)), dim=1)
        return probs.argmax(dim=1), probs

    def latent_elements(self) -> int:
        return self.latent_dim if self.latent_dim is not None else SPATIAL_LATENT_ELEMENTS

    def compression_ratio(self) -> float:
        """Input elements / latent elements (both float32)."""
        return INPUT_ELEMENTS / self.latent_elements()

    def count_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        cls = sum(p.numel() for p in self.classifier.parameters())
        return {
            "encoder": enc,
            "decoder": dec,
            "classifier": cls,
            "total": enc + dec + cls,
        }
