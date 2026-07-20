"""
Example: train CNNAutoencoder on random data, then inspect the latent space.

Install first:
    pip install -e .

Or run directly if torch is already available (no install needed):
    python example.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_autoencoder import CNNAutoencoder, train_epoch, val_epoch, save_checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else
                         "mps"  if torch.backends.mps.is_available() else "cpu")
EPOCHS    = 5
BATCH     = 8
LR        = 1e-3
USE_RES   = True   # ResBlocks inside each stage
USE_LEAKY = False  # swap ReLU → LeakyReLU(0.2) everywhere


# ---------------------------------------------------------------------------
# Synthetic dataset  (replace with your real ImageFolder / custom Dataset)
# ---------------------------------------------------------------------------
def make_fake_loaders(n_train: int = 64, n_val: int = 16):
    x_train = torch.rand(n_train, 3, 128, 128)
    x_val   = torch.rand(n_val,   3, 128, 128)
    train_loader = DataLoader(TensorDataset(x_train), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_val),   batch_size=BATCH)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")

    # 1. Build model
    model = CNNAutoencoder(use_res=USE_RES, leaky=USE_LEAKY).to(DEVICE)

    params = model.count_parameters()
    print(f"Parameters  — encoder: {params['encoder']:,}  "
          f"decoder: {params['decoder']:,}  "
          f"total: {params['total']:,}")
    print(f"Compression ratio (spatial elements): {model.compression_ratio():.2f}×")

    # 2. Verify tensor shapes
    dummy = torch.rand(2, 3, 128, 128, device=DEVICE)
    recon, latent = model(dummy)
    assert latent.shape == (2, 512, 7, 7), f"Unexpected latent shape: {latent.shape}"
    assert recon.shape  == (2,   3, 128, 128)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Latent: {tuple(latent.shape)}  ← 7×7×512 ✓")
    print(f"Output: {tuple(recon.shape)}")

    # 3. Train
    train_loader, val_loader = make_fake_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Optional: mixed-precision on CUDA
    scaler = torch.amp.GradScaler() if DEVICE.type == "cuda" else None

    loss_fn = nn.MSELoss()

    print("\nTraining:")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE,
                                 loss_fn=loss_fn, scaler=scaler)
        val_loss   = val_epoch(model, val_loader, DEVICE, loss_fn=loss_fn)
        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

    # 4. Save checkpoint
    save_checkpoint(model, optimizer, EPOCHS, val_loss, "autoencoder.pt")
    print("\nCheckpoint saved → autoencoder.pt")

    # 5. Encode / decode a single image
    sample = torch.rand(1, 3, 128, 128, device=DEVICE)
    z      = model.encode(sample)
    out    = model.decode(z)
    print(f"\nEncode only: {tuple(sample.shape)} → {tuple(z.shape)}")
    print(f"Decode only: {tuple(z.shape)} → {tuple(out.shape)}")


if __name__ == "__main__":
    main()
