cnn_autoencoder
===============

A PyTorch convolutional autoencoder for 128x128 RGB images, with an optional
classification head sharing the same encoder.

    128x128x3  ->  encoder  ->  bottleneck  ->  decoder  ->  128x128x3
                                    |
                                    +------->  classifier -> 11 class logits

Two models are provided:

- `CNNAutoencoder` -- reconstruction only. Returns `(reconstruction, latent)`.
- `MultiTaskAutoencoder` -- shared encoder feeding a decoder *and* a classifier.
  Returns `(reconstruction, latent, logits)`.

The bottleneck is configurable. Keeping the native 512x7x7 spatial map is only
1.96x smaller than the input, so it is a weak compressor; projecting to a flat
`latent_dim` vector is what makes it a real one (256 dims = 192x compression).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the layer-by-layer design, the path from JPEG file to
restored image, the activation and metric choices, and training considerations.


Dataset
-------

The Google Universal Image Embeddings corpus: 132,528 JPEGs, 128x128, in 11
mutually exclusive classes. The folder name is the label.

    128x128/
        apparel/  artwork/  cars/  dishes/  furniture/  illustrations/
        landmark/ meme/     packaged/ storefronts/ toys/
        train.csv          (present but deliberately unused)

Class sizes are imbalanced 13.8:1 -- landmark has 33,063 images, toys 2,402.
`train.csv` is ignored on purpose: its `image_name` column is not unique across
folders, so it cannot be joined back to a path unambiguously. The directory tree
is the only source of labels.

The images are not in git. `Google-Universal_Image_Embeddings.zip` (~1.4 GB) is
gitignored, so a fresh clone has neither the archive nor the extracted tree --
obtain the archive separately, place it at the repo root and extract it:

    unzip -q Google-Universal_Image_Embeddings.zip     # creates 128x128/<class>/

The extracted images are gitignored (`128x128/**/*.jpg`), so they will not show
up as untracked files. `128x128/train.csv` stays tracked.


Install
-------

Requires Python >= 3.10, torch >= 2.0, torchvision >= 0.15.

    python -m venv .
    ./bin/pip install -r requirements.txt
    ./bin/pip install -e .          # optional, for `import cnn_autoencoder` anywhere

`tqdm` is optional; the progress bar is skipped if it is not installed.

Run the tests (26 tests, no dataset required, ~2 s):

    ./bin/python -m pytest tests/ -q


Quick start: does it run at all?
--------------------------------

`example.py` trains on random tensors, so it needs no dataset. It verifies the
shapes end to end and saves a checkpoint. Seconds on any machine:

    ./bin/python example.py

Expected output: latent shape `(2, 512, 7, 7)`, reconstruction back to
`(2, 3, 128, 128)`, 5 epochs of falling loss, `autoencoder.pt` written.


Training on the real dataset
----------------------------

Smoke test first -- 200 images per class, 2 epochs. This exercises the whole
pipeline (scan, stratified split, both losses, confusion matrix, checkpointing)
in about a minute on an Apple-silicon Mac:

    ./bin/python train_multitask.py --per-class-cap 200 --epochs 2

Full run:

    ./bin/python train_multitask.py --epochs 30 --batch-size 128 --alpha 0.15

Resume from a checkpoint:

    ./bin/python train_multitask.py --resume checkpoints/best.pt --epochs 40

Reconstruction only -- keep the spatial latent and zero out the classification
term (the head is still built, it just receives no gradient):

    ./bin/python train_multitask.py --latent-dim 0 --alpha 0.0 --epochs 20

Useful flags:

    --latent-dim N     Flat bottleneck width (default 256; 0 = 512x7x7 spatial)
    --alpha A          Loss balance, (1-A)*MSE + A*CrossEntropy (default 0.5)
    --leaky            LeakyReLU(0.2) instead of ReLU in hidden layers
    --no-res           Drop the per-stage residual blocks
    --imbalance ...    weighted-loss (default) | balanced-sampler | none
    --per-class-cap N  Cap images per class before splitting -- for fast runs
    --device cpu|mps|cuda    Auto-detected if omitted
    --amp              Mixed precision (CUDA only)
    --num-workers N    DataLoader workers (default 4)
    --no-progress      Disable the progress bar when redirecting to a log

Set `--alpha` deliberately. Pixel MSE on [0,1] images settles around 0.01-0.05
while cross-entropy over 11 classes starts at ln(11) = 2.40, so at the default
alpha=0.5 the classification term dominates the gradient by roughly 50x. If you
want reconstruction quality to actually improve, use 0.05-0.2.

Outputs land in `checkpoints/`: `best.pt` (selected on macro-recall, not top-1),
`last.pt`, and `history.json` with per-epoch metrics.


Training on a regular PC or Mac
-------------------------------

The full corpus is 119k training images per epoch, which is more than a laptop
should be asked to chew through casually. Measured on an Apple M4 / 16 GB with
torch 2.13, `latent_dim=256`, forward+backward only (no JPEG decode):

| Device | Throughput | Full epoch (119k) | Capped epoch (2.2k) |
|---|---|---|---|
| MPS (M4) | ~100 img/s | ~20 min | ~22 s |
| CPU (M4) | ~21 img/s | ~93 min | ~103 s |

Those are compute-only floors. Real runs also decode 119k JPEGs per epoch, so
budget meaningfully more wall-clock unless your data loader keeps up.

Practical guidance for a laptop:

- **Always start capped.** `--per-class-cap 200 --epochs 2` finishes in about a
  minute and catches almost every pipeline bug -- wrong paths, unreadable files,
  a misconfigured latent dim -- before you commit hours.

- **Scale the cap, not the epochs.** `--per-class-cap 2000` gives ~22k
  stratified images and a still-balanced problem, at roughly a fifth of
  full-corpus cost. A model that learns nothing on 22k images will not be
  rescued by 132k.

- **Use MPS on Apple silicon.** It is auto-detected; roughly 5x CPU here. Do not
  pass `--amp` -- it is CUDA-only and is ignored elsewhere.

- **Watch memory, not just time.** The `latent_dim=256` model is 31.9M
  parameters (127.6 MB fp32), and AdamW keeps two more copies of every
  parameter. With activations, batch 128 at 128x128 is comfortable on 16 GB but
  batch 256 is not. Drop to `--batch-size 32` or `--batch-size 64` if you are
  swapping.

- **Tune `--num-workers` to your cores.** The default of 4 is right for a 4-8
  core laptop. On MPS the GPU is often waiting on JPEG decode, so too few
  workers, not the model, is the usual bottleneck.

- **Run `verify_images()` once before a long job.** A full decode pass surfaces
  corrupt JPEGs up front rather than mid-epoch:

      ./bin/python -c "from cnn_autoencoder import verify_images; print(verify_images('128x128'))"

- **Overnight beats interactive.** For a full 30-epoch run, redirect to a log
  and disable the progress bar:

      ./bin/python train_multitask.py --epochs 30 --alpha 0.15 --no-progress > train.log 2>&1


Reading the results
-------------------

Per epoch the trainer prints both loss components, top-1 accuracy, macro-recall
and reconstruction PSNR. Read macro-recall as the headline: landmark and apparel
are half the corpus between them, so top-1 accuracy flatters a model that never
predicts toys. For reconstruction, ~20 dB PSNR is visibly blurry and ~30 dB is a
decent result at this resolution.


Repository layout
-----------------

    cnn_autoencoder/model.py     Encoder, Decoder, CNNAutoencoder,
                                 ClassifierHead, MultiTaskAutoencoder
    cnn_autoencoder/data.py      Folder scan, stratified split, dataset,
                                 transforms, loaders, image verification
    cnn_autoencoder/train.py     Single-task and multi-task epoch loops
    cnn_autoencoder/metrics.py   ConfusionMatrix, top-k accuracy, PSNR
    cnn_autoencoder/utils.py     Checkpoint save/load, reconstruction grids
    train_multitask.py           Main training CLI
    example.py                   Synthetic-data smoke test
    tests/                       26 unit tests

`CNN_Auto_Enc+Multilabel_Classification.py` and its README are an earlier
reference prototype (64x64, multi-label sigmoid) kept for context. They are not
part of the package and are not what `train_multitask.py` runs.


References
----------

- https://www.youtube.com/watch?v=1BkzNb3ejK4
- https://machinelearningmastery.com/mastering-digital-art-with-stable-diffusion/
- https://use.ai/
- https://machinelearningmastery.com/brief-introduction-to-diffusion-models-for-image-generation/
