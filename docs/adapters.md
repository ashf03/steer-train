# Adapters

How `st` attaches to a trainer for **observe** (metrics) and **intervene** (LR).

| Adapter | Status | Default for | Lag |
|---------|--------|-------------|-----|
| `file-tail` | ready | toy | 0–1 step in-process; poll if cross-process |
| `callback` | ready | S1 | ≤1 step (optimizer param groups) |
| `aq` | deferred | — | — |

```bash
st adapters
```

## file-tail

Trainer re-reads a JSON config each step; outer loop writes `lr`.

```python
from st_steer.adapters import FileTailAdapter

adapter = FileTailAdapter(
    config_path="configs/default.json",
    metrics_path="results/steered/metrics.jsonl",
)
adapter.controller.set_lr(0.02)
adapter.metrics.write({"step": 10, "lr": 0.02, "wall_ms": 1.0, "loss": 0.4})
```

Used by `st watch` / `st steer` / toy `st train`.

## callback (S1)

In-process hook; optional HuggingFace `TrainerCallback` when `st-steer[s1]` is installed.

```python
from st_steer.adapters import CallbackAdapter, get_adapter
from st_steer.rules import S1OursRule

# Pure hook (no HF required) — bind get_lr/set_lr or pass optimizer=
adapter = get_adapter(
    "callback",
    metrics_path="results/scale/.../metrics.jsonl",
    rule="s1",  # or S1OursRule()
    seed=7,
    steps=2000,
)
adapter.hook.on_eval(step=500, val_loss=1.4, train_loss=1.2, lr=2e-4)

# With HuggingFace Trainer:
# callback = adapter.as_transformers_callback()
# trainer.add_callback(callback)
```

Rule: `S1OursRule` (`docs/protocol.scale.md`) — plateau on `val_loss` → half LR.

## aq

Deferred. Not required to install or run toy/S1 paths.
