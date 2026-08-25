"""
Classification metrics, in pure torch (no scikit-learn dependency).

Top-1 accuracy alone is misleading on this corpus: landmark and apparel are half
the data between them, so a model that never predicts toys can still look good.
Always read `macro_recall` and the per-class table alongside it.
"""

from __future__ import annotations

import torch


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> dict[int, float]:
    """Fraction of samples whose true class is within the top-k logits."""
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(targets.view(-1, 1))

    results: dict[int, float] = {}
    for k in topk:
        k = min(k, maxk)
        results[k] = correct[:, :k].any(dim=1).float().mean().item()
    return results


class ConfusionMatrix:
    """
    Accumulates a [num_classes, num_classes] count matrix, rows = true class.

    Kept as running counts so it can be updated batch by batch without holding
    every prediction of a 13k-image validation set in memory.
    """

    def __init__(self, num_classes: int, class_names: list[str] | None = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def reset(self) -> None:
        self.matrix.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=1).cpu()
        targets = targets.cpu()
        # bincount over flattened (true, pred) pairs is far faster than a loop
        indices = targets * self.num_classes + preds
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    @property
    def support(self) -> torch.Tensor:
        return self.matrix.sum(dim=1)

    def accuracy(self) -> float:
        total = self.matrix.sum()
        return (self.matrix.diag().sum() / total).item() if total else 0.0

    def per_class_recall(self) -> torch.Tensor:
        return self.matrix.diag().float() / self.support.clamp(min=1).float()

    def per_class_precision(self) -> torch.Tensor:
        predicted = self.matrix.sum(dim=0)
        return self.matrix.diag().float() / predicted.clamp(min=1).float()

    def per_class_f1(self) -> torch.Tensor:
        precision, recall = self.per_class_precision(), self.per_class_recall()
        return 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    def macro_recall(self) -> float:
        """Unweighted mean recall — the imbalance-aware headline number."""
        return self.per_class_recall().mean().item()

    def macro_f1(self) -> float:
        return self.per_class_f1().mean().item()

    def report(self) -> str:
        """Per-class precision/recall/F1/support table."""
        precision = self.per_class_precision()
        recall = self.per_class_recall()
        f1 = self.per_class_f1()
        support = self.support

        width = max((len(n) for n in self.class_names), default=5)
        lines = [f"{'class':<{width}}  {'prec':>6}  {'recall':>6}  {'f1':>6}  {'n':>7}"]
        lines.append("-" * len(lines[0]))
        for i, name in enumerate(self.class_names):
            lines.append(
                f"{name:<{width}}  {precision[i]:>6.3f}  {recall[i]:>6.3f}  "
                f"{f1[i]:>6.3f}  {support[i]:>7d}"
            )
        lines.append("-" * len(lines[0]))
        lines.append(
            f"{'macro':<{width}}  {precision.mean():>6.3f}  {recall.mean():>6.3f}  "
            f"{f1.mean():>6.3f}  {support.sum():>7d}"
        )
        lines.append(f"top-1 accuracy: {self.accuracy():.4f}")
        return "\n".join(lines)

    def matrix_report(self) -> str:
        """Raw confusion matrix, rows = true class, columns = predicted."""
        width = max((len(n) for n in self.class_names), default=5)
        header = " " * width + "".join(f"{n[:6]:>7}" for n in self.class_names)
        lines = [header]
        for i, name in enumerate(self.class_names):
            row = "".join(f"{int(v):>7}" for v in self.matrix[i])
            lines.append(f"{name:<{width}}{row}")
        return "\n".join(lines)


def psnr(mse: float, max_value: float = 1.0) -> float:
    """
    Peak signal-to-noise ratio in dB from a mean squared error.

    More readable than raw MSE for tracking reconstruction quality: ~20 dB is
    visibly blurry, ~30 dB is a decent reconstruction at this resolution.
    """
    if mse <= 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(max_value ** 2 / mse)).item()
