# MiniChatBot

A small ChatGPT-style language model trainable end-to-end on a personal PC. Built around a single decoder-only Transformer that progresses through pretraining, supervised fine-tuning, and reinforcement learning (GRPO) — all three stages are implemented.

> **Inspiration**: this project is an attempt to reproduce — and learn from — Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat). The goal is to internalize the full pretrain → SFT → RL pipeline by re-deriving it from scratch in a slightly more modular layout, not to outperform it. If you want a cleaner, faster reference, go read nanochat directly.

> **Built with**: vibecoded with [Claude Code](https://claude.com/claude-code) (Claude Opus 4.7) under the intense supervision of [@HkHK2Prod](https://github.com/HkHK2Prod). Every line was reviewed, questioned, and (often) rewritten — Claude proposed, the human disposed.

The project favors **clear, modular code over framework magic**: every component (model, tokenizer, dataset, loss, callback, sampling strategy) is a small registered class chosen from a typed YAML config. No hidden globals, no implicit state, no behavior that isn't visible in source.

## Requirements

- **Python**: 3.10 or newer
- **OS**: Windows 10/11, Linux, or macOS
- **GPU**: NVIDIA with CUDA 11.8+ recommended (RTX 4080-class hardware is the design target). CPU-only is supported for tiny debug runs.
- **Disk**: a few hundred MB for code + a small corpus; tens of GB if you train on FineWeb-Edu.
- **RAM**: 16 GB+ recommended for medium runs, 8 GB enough for the debug config.

## Installation

The setup scripts handle venv creation, the right torch wheel for your hardware, and optional extras in one shot.

### Windows (PowerShell)

```powershell
git clone <repo-url> MiniChatBot
cd MiniChatBot
.\scripts\setup.ps1                # default: CUDA cu126, full extras
.\scripts\setup.ps1 -Cpu           # CPU-only fallback
.\scripts\setup.ps1 -Cuda cu128    # different CUDA version
.\scripts\setup.ps1 -Force         # swap an existing CPU venv to CUDA
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
git clone <repo-url> MiniChatBot
cd MiniChatBot
bash scripts/setup.sh              # default: CUDA cu126
USE_CPU=1 bash scripts/setup.sh    # CPU-only fallback
FORCE=1 bash scripts/setup.sh      # swap CPU venv to CUDA
source .venv/bin/activate
```

### Manual (any platform)

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\Activate.ps1 on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or omit --index-url for CPU
pip install -e ".[dev,tensorboard,data]"
```

The data extras (`datasets`, `huggingface_hub`) are needed for any source other than `tiny_shakespeare` (which uses plain `urllib`).

## Quick start — minimal end-to-end run

This runs the full pipeline on Tiny Shakespeare (~1 MB corpus, ~1 M-param model, finishes in seconds on GPU, ~1–3 min on CPU).

```bash
# 1. Download corpus → JSONL (one file per line)
python scripts/data/download_corpus.py --source tiny_shakespeare --output data/shakespeare/corpus.jsonl

# 2. Train byte-level BPE tokenizer
python scripts/data/train_tokenizer.py \
    --corpus data/shakespeare/corpus.jsonl --jsonl-key text \
    --output data/shakespeare/tokenizer.json --vocab-size 2048

# 3. Tokenize + pack into uint16 .bin files
python scripts/data/prepare_data.py \
    --corpus data/shakespeare/corpus.jsonl --jsonl-key text \
    --tokenizer data/shakespeare/tokenizer.json \
    --output data/shakespeare/ --val-frac 0.05

# 4. Train
python scripts/train/pretrain.py --config configs/debug_shakespeare.yaml
```

Outputs land in `runs/<timestamp>_<run_name>/` — checkpoints, JSONL metrics, samples, full config snapshot, and a teed log of stdout/stderr.

VS Code users: the four matching launch configurations are in [.vscode/launch.json](.vscode/launch.json) — pick "Download corpus", "Train tokenizer", "Prepare data", then "Pretrain: debug_shakespeare". `F5` starts whichever is selected under the debugger.

## Scaling up

Drop in a larger source and a beefier model config:

```bash
# TinyStories (~2 GB, ~470K stories) — good first real run
python scripts/data/download_corpus.py --source tiny_stories --output data/tinystories/corpus.jsonl

# FineWeb-Edu (multi-TB) — cap with --max-docs for testing
python scripts/data/download_corpus.py --source fineweb_edu --output data/fineweb/corpus.jsonl \
    --subset sample-10BT --max-docs 1000000
```

Then point [configs/pretrain_small.yaml](configs/pretrain_small.yaml) (or your own copy) at the resulting `train.bin` / `val.bin` and adjust `model.*`, `trainer.batch_size`, `trainer.precision`, etc.

## Data sources & references

All corpora are downloaded from their official upstreams; this repo distributes none of them. Cite the underlying dataset / paper when reporting results.

| Source key | Corpus | Size | Reference |
|---|---|---|---|
| `tiny_shakespeare` | Karpathy's Tiny Shakespeare | ~1 MB, ~1.1 M chars | [karpathy/char-rnn](https://github.com/karpathy/char-rnn) (MIT) |
| `tiny_stories` | TinyStories (Eldan & Li, 2023) | ~2 GB, ~470 K stories | Paper: [arXiv:2305.07759](https://arxiv.org/abs/2305.07759) — Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| `fineweb_edu` | FineWeb-Edu (HuggingFaceFW) | up to multi-TB | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) ([ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/)) |
| `hf_dataset` | Any HuggingFace dataset with a text field | varies | Per-dataset (set on HF Hub) |

**Tokenizer**: built on [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) (Apache-2.0). Byte-level BPE follows the GPT-2 design (Radford et al., 2019).

**Architecture**: standard decoder-only Transformer with [RoPE](https://arxiv.org/abs/2104.09864) (Su et al., 2021), [RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019), [SwiGLU](https://arxiv.org/abs/2002.05202) (Shazeer, 2020), and a KV cache for autoregressive decoding — same shape as Llama / Mistral / Qwen.

If your run produces something publishable, attribute the corpus authors and (optionally) link this repo for the training pipeline.

## Configuration

All runtime behavior is set in YAML; see [configs/](configs/) for examples. The schema lives in [minichatbot/config.py](minichatbot/config.py) and is parsed into typed dataclasses. Top-level sections:

| Section | Purpose |
|---|---|
| `run_name`, `output_dir`, `seed`, `device` | Run identity and infra |
| `model` | Architecture (`type`, `n_layers`, `d_model`, `vocab_size`, `max_seq_len`, …) |
| `tokenizer` | `type` + path to a trained `tokenizer.json` |
| `data` | `type` (`pretrain` / `sft` / `rl`), `train_path`, `val_path`, `seq_len`, `system_prompt` (RL) |
| `rl` | GRPO knobs (RL only): `group_size`, `max_new_tokens`, `temperature`, `top_p`/`top_k`, `reward` |
| `optim` | `lr`, `betas`, `warmup_steps`, `lr_schedule`, `weight_decay` |
| `trainer` | `max_steps`, `batch_size`, `grad_accum_steps`, `precision`, `compile` |
| `callbacks` | Ordered list — order matters; `logfile` first so its tee captures everything |

Each component (model, tokenizer, dataset, loss, sampling strategy, callback) is built from its registry by `type` key — see the registry definitions under [minichatbot/](minichatbot/) for the available implementations.

## Project layout

```
minichatbot/
  config.py              # typed YAML schema
  data/
    base.py              # BaseDataset (registry-dispatched factory)
    pretrain.py          # PretrainDataset (random-access over packed .bin)
    collators/           # batch-level transforms (input_ids vs labels split, etc.)
    sources/             # CorpusSource implementations (tiny_shakespeare, tiny_stories, fineweb_edu, hf_dataset)
  model/
    base.py              # LanguageModel + ModelOutput
    transformer/         # decoder-only transformer with RoPE, RMSNorm, SwiGLU, KV cache
  tokenizer/
    base.py              # Tokenizer interface (registry-dispatched factory)
    bpe.py               # byte-level BPE (HF tokenizers backend)
  inference/
    generator.py         # token-level sampling loop with KV cache
    text_generator.py    # text-in / text-out wrapper
    strategies/          # greedy, temperature, top_k, top_p
  rl/
    rewards/             # Reward implementations (REWARD_REGISTRY) — gsm8k, ...
    rollout.py           # sample completions -> reward -> advantage -> training batch
  training/
    trainer.py           # step-based loop with callback events
    rl_trainer.py        # GRPOTrainer — Trainer subclass with the sample/reward/PG step
    runner.py            # build-and-train for pretrain + SFT
    rl_runner.py         # build-and-train for the RL stage
    losses/              # cross-entropy (pretrain/sft), grpo (RL policy-gradient surrogate)
    callbacks/           # logfile, console, jsonl, tensorboard, wandb, checkpoint, eval, sample
    optim.py             # optimizer + LR scheduler builders
  utils/                 # registry, atomic IO, eval-mode context manager

scripts/
  setup.ps1 / setup.sh        # venv + torch + project install
  data/download_corpus.py     # source registry   -> JSONL
  data/train_tokenizer.py     # JSONL/text        -> tokenizer.json
  data/prepare_data.py        # JSONL + tokenizer -> packed uint16 .bin
  data/download_sft_data.py   # HF dataset        -> chat-format JSONL (SFT)
  data/download_rl_data.py    # HF dataset        -> {question, answer} JSONL (RL)
  train/pretrain.py           # YAML config       -> pretraining run
  train/sft.py                # YAML config       -> SFT run (--from-pretrained)
  train/rl.py                 # YAML config       -> GRPO run (--from-pretrained)

configs/
  debug_shakespeare.yaml # ~1M params, fp32, debugger-friendly
  pretrain_small.yaml    # ~25M params, bf16, real small run
  sft_fineweb.yaml       # SFT on top of the 110M fineweb pretrain
  rl_gsm8k.yaml          # GRPO on GSM8K on top of the 110M fineweb SFT model
```

## Development

```bash
pip install -e ".[dev]"     # installs pytest, ruff, mypy
ruff check .                 # lint
ruff format .                # format
pytest                       # tests (work in progress)
```

The codebase prefers explicit composition over inheritance, dataclasses for config, and small modules over large ones.

## License

MiniChatBot is released under the [MIT License](LICENSE) — free for personal, academic, and commercial use, no warranty.

The licenses of the corpora used for training are independent of this repository — see the **Data sources & references** section above before redistributing any data or trained weights.
