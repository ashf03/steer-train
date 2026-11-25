# Tool MVP spec

Goal: something a stranger installs once and still uses next month for **live
training intervention** — not a paper-only repo.

Name (working): **st** (steer). Rename later if needed; commands below use `st`.

## Install story (60 seconds)

```bash
pip install -e .              # package st-steer → console script st
# later: pip install st-steer (PyPI)

st doctor                     # python, GPU optional, schema OK
st demo --toy                 # synthetic matrix path; no GPU
st demo --s1-dry-run          # prints S1 plan without training
```

**Success criterion:** new user reaches a finished toy compare + plot without
reading more than the README quickstart.

## What users memorize (small surface)

| Command | Meaning |
|---------|---------|
| `st train …` | run trainer adapter (or wrap existing) |
| `st steer …` | one-command train + rule intervene (`run_steered`) |
| `st watch …` | attach outer loop to a live metrics stream |
| `st summarize PATH` | one-run stats |
| `st compare --a A --b B` | paired verdict (protocol-aware) |
| `st plot …` | curves → `figures/` |
| `st matrix …` | multi-seed / multi-condition driver |
| `st doctor` | environment + schema self-check |
| `st validate PATH` | schema v1 check (metrics / snapshot / interventions / run dir) |
| `st tui …` | live dashboard (optional) |

Flags users actually need: `--sleep`, `--seed`, `--metrics`, `--dry-run`,
`--protocol toy|s1`.

Everything else stays under `st … --help` or docs — not memorized.

## Artifact schema (`schema: 1`)

Stable on-disk contract. Breaking changes bump schema integer.

### `metrics.jsonl` (one object per line)

Required keys:

| Key | Type | Notes |
|-----|------|-------|
| `schema` | int | `1` |
| `step` | int | global step |
| `lr` | float | optimizer LR after this step |
| `wall_ms` | float | ms since run start |
| `train_loss` | float | required for scale; toy may use `loss` as alias |

Scale-additional:

| Key | Type | Notes |
|-----|------|-------|
| `val_loss` | float \| null | present on eval steps (every 50); null on non-eval |
| `epoch` | float \| null | optional |

Toy may emit `loss` instead of `train_loss`; readers accept either.

### `config.snapshot.json`

```json
{
  "schema": 1,
  "setting_id": "S1-qwen15b-alpaca-lora",
  "condition": "OURS-RULE",
  "seed": 7,
  "steps": 2000,
  "lr": 0.0002,
  "started_at": "ISO-8601"
}
```

### `interventions.jsonl`

| Key | Type | Notes |
|-----|------|-------|
| `schema` | int | `1` |
| `timestamp` | string | UTC |
| `rule_id` | string | e.g. `S1-plateau-half-lr` |
| `decision_step` | int | observe step when rule fired |
| `evidence` | object | e.g. `{ "val_losses": [...], "best_val": ... }` |
| `action` | object | e.g. `{ "type": "lr_mul", "factor": 0.5, "lr_from": 2e-4, "lr_to": 1e-4 }` |
| `effect_step` | int \| null | first metrics step showing new lr |
| `effect_delay_steps` | int \| null | effect_step − decision_step |

### `compare.json`

Script output: paths, primary metrics, ε check, per-pair verdict — machine-readable twin of `st compare`.

## Adapters (MVP order)

1. **toy** — current synthetic trainer (CI always green — harness, not research PASS)  
2. **callback** — HF `TrainerCallback` / `SteerHook` writes metrics + accepts LR set  
3. **file-tail** — reloadable JSON config (today’s pattern)  
4. **aq** — deferred (listed by `st adapters`, not implemented)

Default path for S1: **callback** (reliable lag ≤1 step).  
Details: `docs/adapters.md`.

## Demo script (must work)

```bash
st demo --toy
# → writes results/, prints compare verdict, writes figures/loss_curves.png
```

GPU demo (when compute available):

```bash
st matrix --protocol s1 --tier day-one --seeds 7 11 13 17 19
```

## Non-goals for MVP

- Hosted SaaS, multi-tenant auth  
- Replacing WandB  
- AutoML search across dozens of trials  
- Requiring axi to run  

## Definition of “loved”

- Install without a research engineer  
- One command shows value  
- Logs are boring and complete  
- Failures are explicit (dry-run, doctor, inconclusive)  
- Docs fit on one screen for quickstart  
