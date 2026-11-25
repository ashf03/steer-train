# Live LR steering (synthetic)

We’re checking if tweaking a training run while it’s still going can make it
finish better than leaving it alone. **No aq.**

## Layout

```
configs/default.json     live-editable hyperparameters
src/st_steer/            installable package (CLI: st)
pyproject.toml           pip install -e . → console script st
results/…/metrics.jsonl  run artifacts
docs/protocol.md         frozen claim + intervene rule
docs/interventions.md    intervene log
paper/draft.md           writeup
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
st doctor
```

## Commands

```bash
# pilot artifacts live under results/pilot/ (do not overwrite)
# new runs: results/steered/ or results/seed_<k>/

st train --reset --sleep 0 --metrics results/baseline/metrics.jsonl
st steer --sleep 0 --seed 7
st matrix --sleep 0

st summarize results/pilot/baseline/metrics.jsonl results/pilot/steered/metrics.jsonl
st compare
st plot --lr

st tui --mode steered --sleep 0.08 --seed 7
st watch --metrics results/steered/metrics.jsonl
st validate results/pilot/baseline
st compare --write   # also writes compare.json
st adapters          # file-tail (toy) · callback (S1) · aq deferred
```

Adapters: `docs/adapters.md`.

## CI

```bash
pip install -e ".[dev]"
pytest                 # includes full toy seed matrix harness
```

GitHub Actions runs doctor → unit tests → `st demo --toy` → full toy matrix.
The matrix job stays **green when the harness works**, even if the research
verdict is FAIL (current toy result).


Legacy shims still work (`python src/train.py`, `python scripts/…`) but prefer `st …`.

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
