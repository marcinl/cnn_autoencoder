# Architecture

How an image travels from a JPEG on disk, through the compression bottleneck,
back to a restored image — and why each piece is shaped the way it is.

Everything below is implemented in `cnn_autoencoder/model.py`,
`cnn_autoencoder/data.py` and `cnn_autoencoder/train.py`.

---

## 1. The path, end to end

```
128x128/apparel/image0000.jpg        file on disk
        |
        |  PIL.Image.open -> convert("RGB")          data.py: ImageClassFolder
        v
   PIL image, 128x128, 3 channels
        |
        |  RandomHorizontalFlip (train only) -> ToTensor
        v
   x : [3, 128, 128] float32 in [0, 1]               NOT mean/std normalised
        |
        |  DataLoader collate
        v
   x : [B, 3, 128, 128]
        |
        |  ===================== ENCODER =====================
        |  stage1  Conv s2 3->64    + BN + ReLU (+ ResBlock)   128 -> 64
        |  stage2  Conv s2 64->128  + BN + ReLU (+ ResBlock)    64 -> 32
        |  stage3  Conv s2 128->256 + BN + ReLU (+ ResBlock)    32 -> 16
        |  stage4  Conv s2 256->512 + BN + ReLU (+ ResBlock)    16 ->  8
        |  bottleneck  Conv k2 s1 p0 512->512 + BN + ReLU        8 ->  7
        v
   [B, 512, 7, 7]   spatial latent  (25,088 floats)
        |
        |  Flatten -> Linear(25088, latent_dim)      only if latent_dim is set
        v
   z : [B, latent_dim]              THE BOTTLENECK  (256 floats by default)
        |                                    |
        |                                    +----> ClassifierHead -> [B, 11] logits
        |
        |  Linear(latent_dim, 25088) -> view(-1, 512, 7, 7)
        v
   [B, 512, 7, 7]
        |
        |  ===================== DECODER =====================
        |  unbottleneck ConvT k2 s1 p0 512->512 + BN + ReLU      7 ->  8
        |  stage4  (ResBlock +) ConvT s2 512->256 + BN + ReLU     8 -> 16
        |  stage3  (ResBlock +) ConvT s2 256->128 + BN + ReLU    16 -> 32
        |  stage2  (ResBlock +) ConvT s2 128->64  + BN + ReLU    32 -> 64
        |  stage1  (ResBlock +) ConvT s2 64->3    + Sigmoid      64 -> 128
        v
   x_hat : [B, 3, 128, 128] in [0, 1]     restored image
```

The decoder is an exact mirror of the encoder. Every encoder `Conv2d(stride=2)`
has a matching `ConvTranspose2d(kernel=4, stride=2, padding=1)`, and the
kernel-2 bottleneck conv has a matching kernel-2 transposed conv.

### Why 8 -> 7 and not straight to 7

Four stride-2 convolutions take 128 down to 8, not 7. A fifth stride-2 stage
would overshoot to 4. The extra `kernel=2, stride=1, padding=0` convolution
trims 8 to 7 exactly:

```
out = floor((in + 2p - k) / s) + 1 = (8 + 0 - 2) / 1 + 1 = 7
```

and the transposed form reverses it exactly:

```
out = (in - 1) * s - 2p + k = (7 - 1) * 1 - 0 + 2 = 8
```

Both directions are exact, so no `output_padding` fudge is needed anywhere and
the shapes round-trip without loss.

---

## 2. The bottleneck, and why `latent_dim` matters

This is the single most consequential design choice in the model.

| Configuration | Latent | Floats | Compression | Params |
|---|---|---|---|---|
| `latent_dim=None` | `[512, 7, 7]` spatial | 25,088 | **1.96x** | 19.1M |
| `latent_dim=256` (default) | `[256]` flat | 256 | **192x** | 31.9M |

The input is `3 * 128 * 128 = 49,152` floats. The native 512x7x7 feature map is
25,088 floats — barely half the input. Calling that a compression bottleneck is
generous; the network can pass most of the image through almost verbatim and
still score well on reconstruction, which means it never has to learn a
*compact* representation of anything.

Projecting to a flat 256-d vector is what forces genuine compression. It also
produces a far better feature for the classifier head and for any downstream
retrieval or nearest-neighbour work, since a single vector per image is directly
comparable across images in a way a 512x7x7 map is not.

The cost is parameters, and it is not small: the two `Linear(25088, 256)` /
`Linear(256, 25088)` projections add 12.9M parameters, taking the model from
19.1M to 31.9M. Note the direction of the trade — the *more* compressive model
is the *larger* one. Those projection matrices are where the compression is
learned.

Both models are provided because they answer different questions. Use
`latent_dim=None` if you want the best possible reconstruction and treat the
autoencoder as a denoiser or feature extractor. Use `latent_dim=256` if
compression, a comparable embedding, or classification accuracy is the point.

---

## 3. Activation and layer choices

### ReLU by default, LeakyReLU(0.2) available (`--leaky`)

Both are provided; `ReLU` is the default and `LeakyReLU(0.2)` is a flag rather
than a hard choice, because which one wins here is empirical, not settled.

`ReLU` zeroes all negative pre-activations. That is cheap and works well in the
encoder, where BatchNorm keeps activations centred and dead units are rare.

`LeakyReLU(0.2)` passes negatives through at 0.2x instead of clamping them to
zero. Its advantage is in the **decoder**: transposed convolutions upsampling
from a heavily compressed latent are prone to units that receive only negative
input early in training and, under plain ReLU, get exactly zero gradient and
never recover. Leaky units always have a gradient path, so the decoder is less
likely to lock in dead regions and produce patchy reconstructions. If you see
reconstructions with flat dead patches that never improve, try `--leaky` first.

The final decoder layer is deliberately different: **no BatchNorm, Sigmoid
activation**. Sigmoid bounds the output to `[0, 1]`, matching the input range
exactly, so MSE against the original image is well-posed and the model cannot
waste capacity predicting impossible pixel values. BatchNorm is omitted there
because normalising the output layer would fight the Sigmoid for control of the
output distribution.

### BatchNorm after every hidden convolution

`Conv -> BatchNorm -> activation` throughout. BatchNorm stabilises training
enough to run at `lr=1e-3` with AdamW without divergence. It is also why
`build_loaders` sets `drop_last=True` on the training loader — a trailing batch
of size 1 makes BatchNorm's variance undefined and crashes the epoch.

### Progressive channel doubling: 64 -> 128 -> 256 -> 512

Each stage halves spatial resolution and doubles channel count. Spatial
information is progressively traded for representational depth, which is the
standard feature-hierarchy pattern: early layers see fine local detail at low
channel count, later layers see coarse structure at high channel count.

### Residual blocks per stage (`use_res=True`, disable with `--no-res`)

One `ResBlock` at each spatial level: two 3x3 convolutions at fixed resolution
with an identity skip. They add refinement capacity *without* touching the
downsampling path, so fine detail survives the trip through the encoder. The
skip connection also shortens the gradient path, which matters in a network this
deep with a hard bottleneck in the middle. They are optional because they cost
compute and the model trains without them.

### Classifier head: pooling, not flattening

`ClassifierHead` accepts either latent form. Given a spatial `[B, 512, 7, 7]`
latent it applies `AdaptiveAvgPool2d(1)` to get `[B, 512]` before the MLP.
Flattening instead would mean a `Linear(25088, 256)` layer of 6.4M parameters —
a classifier head dwarfing the encoder it hangs off. Pooling also makes the head
translation-invariant, which is what you want when the object of interest may
sit anywhere in frame.

The head is `Linear -> ReLU -> Dropout(0.5) -> Linear`, emitting **raw logits**.
No softmax — `nn.CrossEntropyLoss` applies log-softmax internally, and applying
it twice quietly flattens the gradients.

### Softmax, not per-class sigmoid

Each image in this corpus sits in exactly one class folder, so the classes are
mutually exclusive and the task is single-label. Inference is
`torch.softmax(logits, dim=1).argmax(1)`, not a per-class sigmoid threshold.

(The older `CNN_Auto_Enc+Multilabel_Classification.py` prototype in this repo
uses sigmoid multi-label output. That was the right choice for the multi-label
problem it was written for, and the wrong one for this dataset.)

---

## 4. Losses and metrics

### Reconstruction: MSE, reported as PSNR

`nn.MSELoss` between `x_hat` and `x`, both in `[0, 1]`. MSE is the natural
partner for a Sigmoid output layer on normalised pixels. `BCELoss` also works
and is a reasonable thing to try.

Raw MSE is hard to read, so `metrics.psnr()` converts it to decibels:

```
PSNR = 10 * log10(MAX^2 / MSE)     with MAX = 1.0
```

Rules of thumb at this resolution: **~20 dB is visibly blurry, ~30 dB is a
decent reconstruction**. PSNR is a proxy, not ground truth — it correlates
imperfectly with perceived quality — so look at actual reconstructions too.
`utils.reconstruct_grid()` builds an original-vs-reconstruction strip for
exactly that.

### Classification: weighted cross-entropy

`nn.CrossEntropyLoss`, by default with inverse-frequency class weights
normalised to mean 1 (`ImageClassFolder.class_weights()`).

### The combined loss, and the alpha trap

```
total = (1 - alpha) * MSE(x_hat, x) + alpha * CE(logits, y)
```

**The two terms are not on the same scale, and the default is misleading.**
Pixel MSE on `[0, 1]` images settles around 0.01-0.05. Cross-entropy over 11
classes *starts* at `ln(11) = 2.40`. At `alpha=0.5`, which reads like "weight
both equally", the classification term dominates the gradient by roughly **50x**.

That is often the right bias if labels are what you care about. But if you want
reconstruction to measurably improve, alpha has to go well below 0.5 — try
**0.05-0.2**. Always read the two reported loss components separately; the total
alone will not tell you which task is actually being optimised.

### Why macro-recall is the headline metric

The corpus is imbalanced 13.8:1. Landmark (33,063) and apparel (32,226) are
roughly half the data between them, so a model that never once predicts toys
(2,402) can still post a respectable top-1 accuracy. Top-1 measures the class
prior as much as the model.

`ConfusionMatrix.macro_recall()` — unweighted mean of per-class recall — gives
every class equal say, so ignoring a small class is penalised in full.
**`train_multitask.py` selects `best.pt` on macro-recall, not top-1**, for that
reason. Read `confusion.report()` (per-class precision/recall/F1/support) rather
than any single number.

The confusion matrix accumulates running counts via `bincount`, so it never
holds a full validation set's predictions in memory.

---

## 5. Data pipeline details that affect correctness

**Labels come from the directory tree, never `train.csv`.** The CSV's
`image_name` column has 48,814 distinct values across 132,528 rows, so a name
does not identify a file and cannot be joined back to a path unambiguously. The
folder tree is unambiguous by construction.

**Class ordering is sorted, not filesystem order.** `scan_class_folders` sorts
class names so index -> class stays stable across runs and machines. If this
depended on directory iteration order, a checkpoint trained today would
mispredict when reloaded tomorrow. `save_checkpoint` persists `class_names`
alongside the weights for the same reason.

**The split is stratified, not random.** `stratified_split` splits per class, so
both sides keep the full class distribution. A plain random split is fine for
landmark and can starve toys.

**Pixels stay in `[0, 1]` — no mean/std normalisation.** The decoder ends in a
Sigmoid, whose output is `[0, 1]` by construction. The reconstruction target has
to live in the same range or MSE is comparing incompatible scales. This is why
the usual ImageNet normalisation is absent.

**Augmentation is a horizontal flip and nothing else.** The reconstruction
target *is* the augmented tensor. Colour jitter or aggressive crops would move
the target the decoder is chasing, or destroy information it is being asked to
reproduce. A horizontal flip is safe because it is information-preserving and
the flipped image remains its own valid target.

**Corrupt files are skipped, not fatal.** A failed decode substitutes the next
readable sample and records the path in `dataset.failed`, printed at the end of
training. At 132k images, one bad JPEG should not kill an epoch that is hours
in. Run `verify_images()` beforehand to surface them up front.

**Imbalance is corrected once, not twice.** Either weight the loss
(`--imbalance weighted-loss`, the default) or draw balanced batches
(`--imbalance balanced-sampler`). Doing both applies the correction twice and
over-corrects toward the rare classes.

---

## 6. Training considerations

### Optimiser and schedule

AdamW at `lr=1e-3`, `weight_decay=1e-4`, with `CosineAnnealingLR` over the full
epoch budget. Note that the schedule is built with `T_max=args.epochs`, so
**resuming with a different `--epochs` gives a different LR curve** than an
uninterrupted run — the cosine is recomputed against the new horizon.

### Checkpointing

`best.pt` (best macro-recall so far), `last.pt` (every epoch), and optional
`epoch{N}.pt` via `--save-every`. Each carries `class_names` and the model
config as well as the weights, so a checkpoint can be reloaded without guessing
which label index meant "toys" or whether `latent_dim` was set. `history.json`
is rewritten every epoch, so a killed run still leaves its metrics behind.

### Device selection

`cuda` -> `mps` -> `cpu`, auto-detected, overridable with `--device`. Mixed
precision (`--amp`) is CUDA-only; `GradScaler` is simply not constructed on
other backends, so passing it on a Mac is silently a no-op.

### Practical cost

Measured on an Apple M4 / 16 GB, torch 2.13, `latent_dim=256`, forward+backward
only (no JPEG decode):

| Device | Throughput | Full epoch (119k imgs) | Capped epoch (2.2k) |
|---|---|---|---|
| MPS | ~100 img/s | ~20 min | ~22 s |
| CPU | ~21 img/s | ~93 min | ~103 s |

These are compute-only floors — a real epoch also decodes 119k JPEGs. On MPS the
GPU frequently waits on the data loader, so `--num-workers` is often the binding
constraint rather than the model itself.

Memory: 31.9M parameters is 127.6 MB in fp32, and AdamW holds two further copies
(momentum and variance). With activations, batch 128 fits comfortably in 16 GB;
batch 256 does not. Drop to 32 or 64 if the machine starts swapping.

### A sane order of work on a laptop

1. `./bin/python -m pytest tests/ -q` — 26 tests, ~2 s, no dataset needed.
2. `./bin/python example.py` — shapes and training loop on synthetic data.
3. `--per-class-cap 200 --epochs 2` — the whole real pipeline in about a minute.
4. `--per-class-cap 2000` — ~22k stratified images, a genuine signal check.
5. Full corpus overnight, `--no-progress`, redirected to a log.

Scale the cap before scaling epochs. A model that learns nothing on 22k
stratified images will not be rescued by the remaining 110k.
