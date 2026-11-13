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
# fast batch (no TUI) — --sleep overrides JSON, leave configs/default.json alone
python src/train.py --reset --sleep 0 --metrics results/baseline/metrics.jsonl

# full TUI — steered mode with protocol auto-steer
python src/tui.py --mode steered --sleep 0.08 --seed 7

# baseline in TUI (auto-steer off)
python src/tui.py --mode baseline --sleep 0.08 --seed 7

# summarize / compare / plot
python scripts/summarize_run.py results/baseline/metrics.jsonl results/steered/metrics.jsonl
python scripts/compare_runs.py
python scripts/plot_runs.py
python scripts/plot_runs.py --lr   # loss + lr panels → results/figures/loss_curves.png
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

Pilot (seed 7) done; claim frozen in `docs/protocol.md`. Matrix seeds still TODO.

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).
