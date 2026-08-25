"""Metric correctness tests, including the imbalance trap."""

import pytest
import torch

from cnn_autoencoder.metrics import ConfusionMatrix, psnr, topk_accuracy


def test_topk_accuracy():
    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [1.0, 2.0, 3.0]])
    targets = torch.tensor([0, 2, 0])          # correct, 2nd choice, 3rd choice

    scores = topk_accuracy(logits, targets, topk=(1, 2, 3))
    assert scores[1] == pytest.approx(1 / 3)
    assert scores[2] == pytest.approx(2 / 3)
    assert scores[3] == pytest.approx(1.0)


def test_confusion_matrix_counts_and_accuracy():
    cm = ConfusionMatrix(3, ["a", "b", "c"])
    logits = torch.eye(3)[[0, 1, 1]]           # predicts a, b, b
    cm.update(logits, torch.tensor([0, 1, 2]))  # truth      a, b, c

    assert cm.matrix[0, 0] == 1
    assert cm.matrix[2, 1] == 1                 # a 'c' was called 'b'
    assert cm.accuracy() == pytest.approx(2 / 3)
    assert cm.support.tolist() == [1, 1, 1]


def test_macro_recall_exposes_a_majority_class_collapse():
    """
    Top-1 looks fine while the model ignores the rare class; macro recall does not.
    90 of class 0, 10 of class 1, everything predicted as class 0.
    """
    cm = ConfusionMatrix(2, ["landmark", "toys"])
    always_class_zero = torch.tensor([[1.0, 0.0]]).repeat(100, 1)
    targets = torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    cm.update(always_class_zero, targets)

    assert cm.accuracy() == pytest.approx(0.90)
    assert cm.macro_recall() == pytest.approx(0.50)
    assert cm.per_class_recall()[1] == pytest.approx(0.0)


def test_confusion_matrix_accumulates_across_batches():
    cm = ConfusionMatrix(2)
    for _ in range(4):
        cm.update(torch.eye(2), torch.tensor([0, 1]))
    assert cm.matrix.sum() == 8
    assert cm.accuracy() == pytest.approx(1.0)

    cm.reset()
    assert cm.matrix.sum() == 0


def test_reports_render_every_class():
    cm = ConfusionMatrix(3, ["apparel", "landmark", "toys"])
    cm.update(torch.eye(3), torch.tensor([0, 1, 2]))

    report = cm.report()
    assert all(name in report for name in ["apparel", "landmark", "toys", "macro"])
    assert "apparel" in cm.matrix_report()


def test_psnr_matches_known_values():
    assert psnr(0.01) == pytest.approx(20.0)
    assert psnr(0.0) == float("inf")
    assert psnr(0.001) > psnr(0.01)            # lower MSE → higher PSNR
