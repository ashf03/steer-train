# Scale protocol — S1 freeze

Toy harness: `docs/protocol.md`.  
Research brief: `log.md`.  
Tool MVP: `docs/tool-mvp.md`.

## Status

| Piece | State |
|-------|--------|
| Primary setting | **FROZEN** |
| Baselines / conditions | **FROZEN** |
| Success metrics + ε + seeds | **FROZEN** |
| S1 intervene rule (`OURS-RULE`) | **FROZEN** |

Claim-facing GPU runs may begin only under this document + matching code tags.

---

## Primary setting (frozen)

**ID:** `S1-qwen15b-alpaca-lora`

| Field | Value |
|-------|--------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adaptation | LoRA r=16, α=32, dropout=0.05 (attn + MLP projections) |
| Data | Stanford Alpaca 52k; seeded 95/5 train/val split |
| Max steps | 2000 |
| Log / eval every | 50 steps |
| Seq length | 512 |
| Effective batch | 32 |
| Base LR (C0 / C4 start) | 2e-4 AdamW |
| Warmup | ratio 0.03 (unless condition overrides) |
| Hardware class | 1× NVIDIA GPU ≥24GB |
| Precision | bf16 if supported, else fp16 |

**Secondary (later):** `S2-cifar100-resnet18` — see `log.md`.

---

## Baselines & conditions (frozen)

### Must-beat

| ID | Name | S1 spec |
|----|------|---------|
| C0 | `B-UNSTEERED` | Warmup 0.03 then **constant** lr=2e-4; 2000 steps; no live edits |

### Must-match-or-beat (best-of bar)

| ID | Name | S1 spec |
|----|------|---------|
| C1 | `B-COSINE` | Warmup 0.03 then cosine → 0.1× peak over 2000 steps |
| C2 | `B-PLATEAU` | Warmup 0.03 then ReduceLROnPlateau on val loss (factor=0.5, patience=3 evals, threshold=1e-3, cooldown=1, min_lr=1e-6) |
| C3 | `B-EARLYSTOP` | Same LR as C0; early stop on val loss (patience=5 evals, min_delta=1e-3); keep best-val ckpt |

### Ours

| ID | Name | Role |
|----|------|------|
| C4 | `OURS-RULE` | Primary method — rule below |
| C5 | `OURS-HUMAN` | Ablation; 10 min watch cap |
| C6 | `OURS-POLICY` | Optional only |

### Matrix tiers

- **Day-one:** C0, C1, C4  
- **Submission:** C0, C1, C2, C3, C4  
- **Optional:** C5, C6  

---

## Success metrics (frozen)

| Rank | Metric | Definition |
|------|--------|------------|
| **Primary** | `best_val_loss` | min `val_loss` over evals (every 50 steps); **lower better** |
| Efficiency (report) | `steps_to_target` | first step with `val_loss ≤ τ`; τ = median C0 `best_val_loss` across seeds (freeze τ in summary after C0 completes) |
| Secondary | `final_val_loss`, `final_train_loss`, `wall_s`, `gpu_hours` | as in `log.md` |
| Process | `schema_ok` | artifacts validate |

**ε (match):** C4 matches baseline B if  
`best_val_loss(C4) ≤ best_val_loss(B) × 1.02`.

**Beat C0:** `best_val_loss(C4) < best_val_loss(C0)`.

**Seeds:** `{7, 11, 13, 17, 19}`.

**Per-seed verdict**

- **pass** — beat C0 and match best available strong baseline within ε  
  (day-one: best of {C1}; submission: best of {C1,C2,C3})
- **fail** — `best_val_loss(C4) > best_val_loss(C0) × 1.02`, or worse than best strong baseline by >ε
- **inconclusive** — crash, OOM, missing artifacts, or C4 required intervene never fired

**Matrix call (5 seeds):** ≥3 pass → PASS; ≥3 fail → FAIL; else INCONCLUSIVE.

---

## S1 intervene rule — `OURS-RULE` (frozen)

Starts identical to `B-UNSTEERED` (warmup 0.03 → constant 2e-4).  
Outer loop reads **only** streamed metrics (train + val). **No test labels.**

### Observe

Each eval row (every 50 steps) must include at least:

```json
{"step": 500, "train_loss": 1.23, "val_loss": 1.45, "lr": 0.0002, "wall_ms": 12345.0}
```

Maintain `best_val` = minimum `val_loss` seen so far.

### Trigger (all must hold)

1. `step >= 500`
2. At least **3 consecutive evals** with `val_loss > best_val - 1e-3`  
   (no meaningful improvement vs best-so-far; `min_delta = 1e-3`)
3. Fewer than **2** interventions already applied this run

### Action (one shot per firing)

- `lr ← max(lr * 0.5, 1e-6)` via live config / optimizer param group update
- Do **not** change LoRA ranks, data, or max_steps
- Append intervention log row immediately; record `effect_step` when next metric row shows new `lr` (target lag ≤ 1 eval = 50 steps)

### Caps / restore

- Max **2** LR cuts per run
- On process exit, restore start LR in shared config file if using file-tail adapter (callback adapter: no global file mutate required)
- If trigger never fires by step 2000 → seed **inconclusive** for C4

### Human condition `OURS-HUMAN` (ablation)

Same observe stream + log schema; human may cut LR (same action) or no-op; **10 minute** interactive cap; no other actions.

### Explicit non-rules

- Do not intervene before step 500  
- Do not use future evals / test split  
- Do not compare to baseline JSONL during the run (no peek)

---

## Artifact root

```
results/scale/S1-qwen15b-alpaca-lora/seed_<k>/<condition>/
  metrics.jsonl
  config.snapshot.json
  interventions.jsonl   # required non-empty semantics for C4/C5; empty OK for C0–C3
  compare.json
```

Schema version field: `"schema": 1` on every JSON/JSONL object header or first line meta — see `docs/tool-mvp.md`.
