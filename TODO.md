# TODO

- [x] Add benchmarks to evaluate model performance
      (in-house per-phase benchmark prompt sets + EleutherAI lm-evaluation-harness integration)
- [ ] Switch virtual environment to anaconda
- [ ] Upgrade the model to support 4x H200 GPU computation
- [ ] Add tests (see "Unit test suite" below — do this on its own branch)

# OPTIMIZATION FOR H200
- [ ] FP16
- [ ] Padding with correct mask
- [ ] Activation checkpoint

# Unit test suite

Pytest is already configured (`[tool.pytest.ini_options]`, `testpaths = ["tests"]`)
but `tests/` is empty. Build the suite on a dedicated branch. Prioritize
pure-logic modules with no GPU/model dependency so the suite stays fast and
runnable in CI on CPU. Use `pytest.mark.parametrize` for case tables and a few
small fixtures (a tiny trained `BPETokenizer`, a stub `Tokenizer`, a stub model).

## Priority 1 — dependency-free (no torch model needed)

- [ ] `tests/test_gsm8k_reward.py` — `minichatbot/rl/rewards/gsm8k.py`
  - `_normalize_number`: "1,000" / "1000" / "1000.00" all normalize equal;
    negatives; decimals kept; non-numeric input returned as-is.
  - `extract_final_answer`: prefers number after `####`; falls back to the
    last number when no marker; returns None when no number present.
  - `GSM8KReward.__call__`: 1.0 on match, 0.0 on mismatch, 0.0 when the
    reference has no extractable number.
- [ ] `tests/test_registry.py` — `minichatbot/utils/registry.py`
  - Duplicate `register(key)` raises ValueError; missing `__getitem__` raises
    KeyError with the "Available:" listing; `__contains__`; `keys()` sorted;
    `repr`/`str` formatting (empty and non-empty).

## Priority 2 — core of the lm-eval branch

- [ ] `tests/test_eval_scoring.py` — `minichatbot/eval/scoring.py`
  - `_prepare` (pure): empty context prepends `prefix_id`; over-length input is
    left-truncated keeping the continuation; `n_cont` capped at len-1; n_cont
    floored at 0.
  - `score_batch` with a STUB model returning fixed logits (no real transformer):
    empty continuation -> (0.0, True); `is_greedy` argmax detection true/false;
    summed logprob matches a hand-computed `log_softmax`; batch order preserved;
    ragged batch (right-pad) gives same result as scoring each pair alone.

## Priority 3 — silent-bug guards

- [ ] `tests/test_chat_template.py` — `minichatbot/chat/template.py`
  - `render_messages`: only assistant content + its closing `<|im_end|>` are
    learned (everything else -100); `input_ids`/`labels` equal length and
    pre-shifted (labels[i] predicts input_ids -> tokens[i+1]); unknown role
    raises; empty conversation raises; multi-turn inter-turn newline masked.
  - `render_prompt_for_completion`: ends with `<|im_start|>assistant\n` header;
    unknown role raises.
  - Tokenizer missing chat tokens raises ValueError.

## Priority 4 — lighter wins

- [ ] `tests/test_config.py` — `minichatbot/config.py`
  - `load_config`/`save_config` round-trip; `_from_dict` nested type coercion;
    `validate` rejects bad constraints (enumerate the validated invariants).
- [ ] `tests/test_sampling.py` — `minichatbot/inference/strategies/*.py`
  - Greedy == argmax; `TemperatureSampling(temp=0)` falls back to greedy;
    output shape == (batch,); top-k/top-p mask out filtered tokens. Seed RNG.
- [ ] `tests/test_bpe_tokenizer.py` — `minichatbot/tokenizer/bpe.py`
  - Train on a tiny corpus; encode/decode round-trip; special-token round-trip;
    `encode_batch` matches per-item `encode`; `special_token_id` lookups.

## Infra

- [ ] `tests/conftest.py` — shared fixtures: tiny trained `BPETokenizer`, stub
  `Tokenizer` (implements the base interface), stub model returning fixed logits.
- [ ] `.github/workflows/ci.yml` — run `pytest` + `ruff check` on push/PR
  (CPU-only; install with the `[dev]` and `[eval]` extras). 