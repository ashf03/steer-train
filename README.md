# Live LR steering (synthetic)

We’re checking if tweaking a training run while it’s still going can make it
finish better than leaving it alone. **No aq.**

## Layout

```
configs/default.json     live-editable hyperparameters
src/train.py             headless trainer (+ --sleep override)
src/tui.py               full-screen live steer TUI
requirements.txt         textual
results/…/metrics.jsonl  run artifacts
docs/protocol.md         frozen claim + intervene rule
docs/interventions.md    intervene log
paper/draft.md           writeup
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
# pilot artifacts live under results/pilot/ (do not overwrite)
# new runs: results/steered/ or results/seed_<k>/

# fast batch baseline
python src/train.py --reset --sleep 0 --metrics results/baseline/metrics.jsonl

# one command: train + auto-steer together
python scripts/run_steered.py --sleep 0.08 --seed 7
python scripts/run_steered.py --sleep 0 --seed 7

# full seed matrix → results/seed_<k>/ + results/matrix_summary.json
python scripts/run_matrix.py --sleep 0

# summarize / compare / plot (defaults = pilot)
python scripts/summarize_run.py results/pilot/baseline/metrics.jsonl results/pilot/steered/metrics.jsonl
python scripts/compare_runs.py
python scripts/plot_runs.py --lr

# TUI
python src/tui.py --mode steered --sleep 0.08 --seed 7
```

### TUI keys

| Key | Action |
|-----|--------|
| `i` | slow down (cut lr → 0.02) |
| `r` | speed up (restore lr → 0.05) |
| `a` | auto on/off |
| `p` | pause |
| `q` | quit |

## Status

Pilot archived under `results/pilot/`. Toy matrix `{7,11,13,17,19}` —
aggregate **FAIL** (`results/matrix_summary.json`).

Scale target locked: `docs/protocol.scale.md` (S1 Qwen2.5-1.5B LoRA · Alpaca).  
Tool MVP: `docs/tool-mvp.md`. Research brief: `log.md`.

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).
