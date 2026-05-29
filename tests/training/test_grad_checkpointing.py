"""Tests for activation (gradient) checkpointing on the Transformer.

The load-bearing guarantee: checkpointing only trades memory for recompute, so
gradients must be numerically identical to a plain forward/backward. The rest
verify the wiring — it engages only on the training path, never during eval or
KV-cache decode, and the config flag flows through build_model.
"""

from __future__ import annotations

import torch

from minichatbot.config import ModelConfig, TrainerConfig
from minichatbot.model.transformer.model import Transformer


def test_trainer_config_defaults_grad_checkpointing_off() -> None:
    assert TrainerConfig().grad_checkpointing is False


def test_gradients_match_plain_forward(tiny_model_config: ModelConfig) -> None:
    # Same inputs + same init; gradients with checkpointing must match without.
    ids = torch.randint(
        0,
        tiny_model_config.vocab_size,
        (2, 8),
        generator=torch.Generator().manual_seed(7),
    )

    def run(enabled: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        torch.manual_seed(0)  # identical weight init each run
        model = Transformer(tiny_model_config)
        model.set_gradient_checkpointing(enabled)
        loss = model(ids).logits.float().pow(2).mean()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        return loss.detach(), grads

    loss_plain, grads_plain = run(False)
    loss_ckpt, grads_ckpt = run(True)

    assert torch.allclose(loss_plain, loss_ckpt, atol=1e-6)
    assert grads_plain.keys() == grads_ckpt.keys()
    for name in grads_plain:
        assert torch.allclose(grads_plain[name], grads_ckpt[name], atol=1e-5), name


def test_checkpoint_engages_only_on_training_path(
    tiny_model_config: ModelConfig, monkeypatch
) -> None:
    import minichatbot.model.transformer.model as model_mod

    calls = {"n": 0}
    real = model_mod.checkpoint

    def spy(fn, *args, **kwargs):
        calls["n"] += 1
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(model_mod, "checkpoint", spy)

    model = Transformer(tiny_model_config)
    model.set_gradient_checkpointing(True)
    ids = torch.randint(0, tiny_model_config.vocab_size, (2, 6))

    # Training step: grad enabled, no cache -> one checkpoint call per block.
    calls["n"] = 0
    model(ids).logits.sum().backward()
    assert calls["n"] == tiny_model_config.n_layers

    # Under no_grad (eval): nothing to recompute -> skipped.
    calls["n"] = 0
    with torch.no_grad():
        model(ids)
    assert calls["n"] == 0

    # KV-cache decode (state provided): skipped even with grad enabled.
    calls["n"] = 0
    state = model.init_state(batch_size=2, device=torch.device("cpu"))
    model(ids, state=state)
    assert calls["n"] == 0

    # Flag off: never checkpoints.
    model.set_gradient_checkpointing(False)
    calls["n"] = 0
    model(ids).logits.sum().backward()
    assert calls["n"] == 0


def test_build_model_wires_the_flag(tiny_model_config: ModelConfig) -> None:
    from minichatbot.training.builders import build_model

    cpu = torch.device("cpu")
    common = dict(
        device=cpu,
        compile=False,
        pretrained_ckpt=None,
        incoming_state=None,
        weights_label="x",
    )
    on = build_model(tiny_model_config, grad_checkpointing=True, **common)
    off = build_model(tiny_model_config, **common)
    assert isinstance(on, Transformer) and on.grad_checkpointing is True
    assert isinstance(off, Transformer) and off.grad_checkpointing is False
