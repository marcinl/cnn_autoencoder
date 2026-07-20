"""
CNN Autoencoder: 128x128x3  →  7x7x512  →  128x128x3

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
    128x128x3  →  7x7x512

    Spatial reduction:
      stride-2 ×4: 128→64→32→16→8
      kernel-2     : 8→7
    """

    def __init__(self, use_res: bool = True, leaky: bool = False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bottleneck(x)   # [B, 512, 7, 7]
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
    """

    def __init__(self, use_res: bool = True, leaky: bool = False):
        super().__init__()
        kw = dict(leaky=leaky)

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
        use_res:  Add one ResBlock per spatial stage for better gradient flow.
        leaky:    Use LeakyReLU(0.2) instead of ReLU throughout hidden layers.
    """

    def __init__(self, use_res: bool = True, leaky: bool = False):
        super().__init__()
        self.encoder = Encoder(use_res=use_res, leaky=leaky)
        self.decoder = Decoder(use_res=use_res, leaky=leaky)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns latent tensor of shape [B, 512, 7, 7]."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs image from latent tensor."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        return self.decode(z), z

    def compression_ratio(self) -> float:
        """Bits-in / bits-out assuming float32."""
        input_elements  = 3 * 128 * 128          # 49 152
        latent_elements = 512 * 7 * 7            # 25 088
        return input_elements / latent_elements   # ≈ 1.96×

    def count_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}
