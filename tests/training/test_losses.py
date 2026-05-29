"""Tests for training losses (minichatbot/training/losses/*.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from minichatbot.model.base import ModelOutput
from minichatbot.training.losses.grpo import GRPOLoss
from minichatbot.training.losses.pretrain import PretrainLoss


def test_pretrain_loss_matches_manual_nll() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, 0], [4, 3, 2]])
    got = PretrainLoss()(ModelOutput(logits=logits), {"labels": labels})

    logp = F.log_softmax(logits, dim=-1)
    manual = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()
    assert torch.allclose(got, manual)


def test_pretrain_loss_ignores_masked_positions() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 2, 5)
    labels = torch.tensor([[1, 2]])
    base = PretrainLoss()(ModelOutput(logits=logits), {"labels": labels})

    # Append a position whose label is ignored: loss must be unchanged.
    logits_ext = torch.cat([logits, torch.randn(1, 1, 5)], dim=1)
    labels_ext = torch.tensor([[1, 2, -100]])
    masked = PretrainLoss()(ModelOutput(logits=logits_ext), {"labels": labels_ext})
    assert torch.allclose(base, masked)


def test_grpo_loss_hand_computed() -> None:
    # B=1, T=3. GRPO aligns logits[:, :-1] with input_ids[:, 1:].
    logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # (1,3,2)
    input_ids = torch.tensor([[0, 1, 0]])
    loss_mask = torch.tensor([[0.0, 1.0, 1.0]])  # only the two completion tokens
    advantages = torch.tensor([0.5])
    batch = {"input_ids": input_ids, "loss_mask": loss_mask, "advantages": advantages}

    got = GRPOLoss()(ModelOutput(logits=logits), batch)

    # Targets are input_ids[1:] = [1, 0]; predicted from logits[:-1] = positions 0,1.
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1,2,2)
    neg_logp = -logp[0].gather(-1, torch.tensor([[1], [0]])).squeeze(-1)  # (2,)
    mask = loss_mask[0, 1:]  # [1, 1]
    expected = (0.5 * neg_logp * mask).sum() / mask.sum().clamp(min=1.0)
    assert torch.allclose(got, expected)


def test_grpo_loss_empty_mask_is_zero_not_nan() -> None:
    logits = torch.randn(1, 3, 4)
    batch = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "loss_mask": torch.zeros(1, 3),  # nothing to learn from
        "advantages": torch.tensor([1.0]),
    }
    got = GRPOLoss()(ModelOutput(logits=logits), batch)
    assert torch.isfinite(got)
    assert got == 0.0
