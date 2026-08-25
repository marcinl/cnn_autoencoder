"""
Train the multi-task CNN autoencoder (reconstruction + 11-class labels) on ./128x128.

Smoke test the whole pipeline on a stratified subset first (~2 min on MPS):

    ./bin/python train_multitask.py --per-class-cap 200 --epochs 2

Full run:

    ./bin/python train_multitask.py --epochs 30 --batch-size 128 --alpha 0.15

Resume:

    ./bin/python train_multitask.py --resume checkpoints/best.pt --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from cnn_autoencoder import (
    ConfusionMatrix,
    MultiTaskAutoencoder,
    build_datasets,
    build_loaders,
    load_checkpoint,
    save_checkpoint,
    train_epoch_multitask,
    val_epoch_multitask,
)
from cnn_autoencoder.metrics import psnr

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a multi-task CNN autoencoder + classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--data-root", type=Path, default=Path("128x128"))
    data.add_argument("--val-fraction", type=float, default=0.1)
    data.add_argument("--per-class-cap", type=int, default=None,
                      help="Cap images per class before splitting (smoke tests)")
    data.add_argument("--no-augment", action="store_true",
                      help="Disable the random horizontal flip")
    data.add_argument("--num-workers", type=int, default=4)

    model = p.add_argument_group("model")
    model.add_argument("--latent-dim", type=int, default=256,
                       help="Flat bottleneck width; 0 keeps the 512x7x7 spatial latent")
    model.add_argument("--no-res", action="store_true", help="Drop the ResBlocks")
    model.add_argument("--leaky", action="store_true", help="LeakyReLU instead of ReLU")
    model.add_argument("--dropout", type=float, default=0.5)

    optim = p.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=30)
    optim.add_argument("--batch-size", type=int, default=64)
    optim.add_argument("--lr", type=float, default=1e-3)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--alpha", type=float, default=0.5,
                       help="Loss balance: (1-a)*MSE + a*CE. See note in train.py — "
                            "CE outweighs MSE ~50x at a=0.5; try 0.05-0.2 to favour "
                            "reconstruction")
    optim.add_argument("--imbalance", choices=["weighted-loss", "balanced-sampler", "none"],
                       default="weighted-loss",
                       help="How to handle the 13.8:1 class imbalance")
    optim.add_argument("--seed", type=int, default=42)
    optim.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected)")
    optim.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    out.add_argument("--resume", type=Path, default=None)
    out.add_argument("--save-every", type=int, default=0,
                     help="Also save every N epochs (0 = only best/last)")
    out.add_argument("--no-progress", action="store_true",
                     help="Disable the per-batch progress bar (use when redirecting to a log)")

    return p.parse_args()


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    latent_dim = args.latent_dim if args.latent_dim > 0 else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -- data ---------------------------------------------------------------
    print(f"Scanning {args.data_root} ...")
    train_ds, val_ds, class_names = build_datasets(
        root=args.data_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
        per_class_cap=args.per_class_cap,
        augment=not args.no_augment,
    )
    train_loader, val_loader = build_loaders(
        train_ds,
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampling=args.imbalance == "balanced-sampler",
        pin_memory=device.type == "cuda",
        seed=args.seed,
    )
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Train: {len(train_ds):,} images   Val: {len(val_ds):,} images")
    print(f"Per-class train counts: {dict(zip(class_names, train_ds.class_counts()))}")

    # -- model --------------------------------------------------------------
    model = MultiTaskAutoencoder(
        num_classes=len(class_names),
        latent_dim=latent_dim,
        use_res=not args.no_res,
        leaky=args.leaky,
        dropout=args.dropout,
    ).to(device)

    params = model.count_parameters()
    print(f"\nDevice: {device}")
    print(f"Latent: {latent_dim or '512x7x7 (spatial)'}   "
          f"compression {model.compression_ratio():.2f}x")
    print(f"Params — encoder {params['encoder']:,}  decoder {params['decoder']:,}  "
          f"classifier {params['classifier']:,}  total {params['total']:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    recon_loss_fn = nn.MSELoss()
    if args.imbalance == "weighted-loss":
        weights = train_ds.class_weights().to(device)
        cls_loss_fn = nn.CrossEntropyLoss(weight=weights)
        print(f"Class weights: {dict(zip(class_names, [round(w, 3) for w in weights.tolist()]))}")
    else:
        cls_loss_fn = nn.CrossEntropyLoss()

    start_epoch = 1
    best_macro_recall = 0.0
    if args.resume:
        ckpt = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_macro_recall = ckpt.get("macro_recall", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # -- train --------------------------------------------------------------
    history: list[dict] = []
    print(f"\nTraining for {args.epochs} epochs (alpha={args.alpha})\n")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        show_progress = tqdm is not None and not args.no_progress
        progress = (
            tqdm(total=len(train_loader), desc=f"epoch {epoch}/{args.epochs}", leave=False)
            if show_progress else None
        )

        train_metrics = train_epoch_multitask(
            model, train_loader, optimizer, device,
            alpha=args.alpha,
            recon_loss_fn=recon_loss_fn,
            cls_loss_fn=cls_loss_fn,
            scaler=scaler,
            progress=progress,
        )
        if progress is not None:
            progress.close()

        confusion = ConfusionMatrix(len(class_names), class_names)
        val_metrics = val_epoch_multitask(
            model, val_loader, device,
            alpha=args.alpha,
            recon_loss_fn=recon_loss_fn,
            cls_loss_fn=cls_loss_fn,
            confusion=confusion,
        )
        scheduler.step()

        macro_recall = confusion.macro_recall()
        elapsed = time.time() - started
        print(
            f"epoch {epoch:>3}/{args.epochs}  {elapsed:>6.1f}s  "
            f"train loss {train_metrics['loss']:.4f} (recon {train_metrics['recon']:.4f} "
            f"cls {train_metrics['cls']:.4f}) acc {train_metrics['acc']:.3f}  |  "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f} "
            f"macro-recall {macro_recall:.3f} psnr {psnr(val_metrics['recon']):.1f}dB"
        )

        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(elapsed, 1),
            "train": train_metrics,
            "val": {**val_metrics, "macro_recall": macro_recall,
                    "macro_f1": confusion.macro_f1()},
        }
        history.append(record)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

        extra = {
            "class_names": class_names,
            "macro_recall": macro_recall,
            "config": {
                "latent_dim": latent_dim,
                "use_res": not args.no_res,
                "leaky": args.leaky,
                "dropout": args.dropout,
                "alpha": args.alpha,
            },
        }
        save_checkpoint(model, optimizer, epoch, val_metrics["loss"],
                        args.out_dir / "last.pt", extra=extra)

        # Select on macro recall, not top-1: top-1 is dominated by the two
        # classes that make up half the corpus.
        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            save_checkpoint(model, optimizer, epoch, val_metrics["loss"],
                            args.out_dir / "best.pt", extra=extra)
            print(f"  ↳ new best macro-recall {macro_recall:.4f} → {args.out_dir / 'best.pt'}")
            print(confusion.report())

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics["loss"],
                            args.out_dir / f"epoch{epoch:03d}.pt", extra=extra)

    skipped = train_ds.failed | val_ds.failed
    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable image(s):")
        for path in sorted(skipped)[:20]:
            print(f"  {path}")

    print(f"\nDone. Best macro-recall {best_macro_recall:.4f}. "
          f"Checkpoints in {args.out_dir}/")


if __name__ == "__main__":
    main()
