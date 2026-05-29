"""Tests for the supervised Trainer (minichatbot/training/trainer.py).

Runs a real (tiny) Transformer for a handful of CPU steps in fp32 — fast,
deterministic, and enough to verify the training loop, checkpoint round-trip,
and the vocab-size guard.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from minichatbot.config import Config, DataConfig, ModelConfig, OptimConfig, TrainerConfig
from minichatbot.data.collators.pretrain import PretrainCollator
from minichatbot.model.base import LanguageModel
from minichatbot.model.transformer.model import Transformer
from minichatbot.training.losses.pretrain import PretrainLoss
from minichatbot.training.optim import build_optimizer, build_scheduler
from minichatbot.training.trainer import Trainer

SEQ_LEN = 8
BATCH = 4
VOCAB = 32


def _build_trainer(tmp_path: Path, *, max_steps: int, tokenizer=None, seed: int = 0):
    torch.manual_seed(seed)
    mcfg = ModelConfig(
        vocab_size=VOCAB,
        max_seq_len=32,
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    model = Transformer(mcfg)

    g = torch.Generator().manual_seed(seed)
    # Exactly one batch worth of fixed samples -> the cycling loader repeats it,
    # so the model overfits a single batch (loss must fall).
    samples = [torch.randint(0, VOCAB, (SEQ_LEN + 1,), generator=g) for _ in range(BATCH)]
    loader = DataLoader(samples, batch_size=BATCH, collate_fn=PretrainCollator(), shuffle=False)

    ocfg = OptimConfig(lr=1e-2, weight_decay=0.0, warmup_steps=0, lr_schedule="constant")
    optimizer = build_optimizer(model, ocfg)
    scheduler = build_scheduler(optimizer, ocfg, max_steps)
    tcfg = TrainerConfig(
        max_steps=max_steps,
        batch_size=BATCH,
        grad_accum_steps=1,
        grad_clip=1.0,
        precision="fp32",
        compile=False,
    )
    full = Config(
        run_name="t",
        data=DataConfig(train_path="x", seq_len=SEQ_LEN, num_workers=0),
        output_dir=str(tmp_path),
        model=mcfg,
        optim=ocfg,
        trainer=tcfg,
    )
    trainer = Trainer(
        config=tcfg,
        full_config=full,
        model=model,
        loss=PretrainLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        callbacks=[],
        run_dir=tmp_path,
        device=torch.device("cpu"),
        tokenizer=tokenizer,
    )
    return trainer, model, loader


def _batch_loss(model, loader) -> float:
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch["input_ids"])
        return float(PretrainLoss()(out, batch).item())


def test_overfits_single_batch(tmp_path: Path) -> None:
    trainer, model, loader = _build_trainer(tmp_path, max_steps=40)
    before = _batch_loss(model, loader)
    trainer.fit()
    after = _batch_loss(model, loader)
    assert after < before * 0.5  # 40 steps on one batch should drive loss down


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    trainer_a, model_a, _ = _build_trainer(tmp_path, max_steps=3, seed=0)
    trainer_a.fit()
    ckpt = tmp_path / "ckpt.pt"
    trainer_a.save_checkpoint(ckpt)

    # Fresh trainer with a *different* random init; loading must overwrite it.
    trainer_b, model_b, _ = _build_trainer(tmp_path, max_steps=3, seed=999)
    assert not torch.equal(model_a.tok_embed.weight, model_b.tok_embed.weight)
    trainer_b.load_checkpoint(ckpt)

    assert trainer_b.step == 3
    assert torch.equal(model_a.tok_embed.weight, model_b.tok_embed.weight)
    assert trainer_b.optimizer.state_dict()["state"]  # optimizer momentum restored

    # The checkpoint is also a valid standalone model file.
    ids = torch.randint(0, VOCAB, (1, 5))
    loaded = LanguageModel.load(ckpt)
    loaded.eval()
    model_a.eval()
    with torch.no_grad():
        assert torch.allclose(loaded(ids).logits, model_a(ids).logits, atol=1e-6)


def test_vocab_size_mismatch_raises(tmp_path: Path) -> None:
    bad_tokenizer = types.SimpleNamespace(vocab_size=VOCAB + 1)
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        _build_trainer(tmp_path, max_steps=1, tokenizer=bad_tokenizer)
