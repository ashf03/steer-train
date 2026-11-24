# Research brief — live outer-loop training intervention

*Living note. Toy harness exists; scale claim not yet locked. Target: ICML / NeurIPS
+ a tool people keep using.*

## Problem

Training runs are long, expensive, and mostly **fire-and-forget**. Practitioners
already intervene by hand (cut LR, stop early, change batch size, kill a bad
seed), but those decisions are:

- **ad hoc** — not predeclared, hard to defend in a paper or postmortem;
- **opaque** — scattered shell history, WandB notes, Slack screenshots;
- **non-reproducible** — the next person cannot replay *when* and *why* the run
  changed;
- **underserved by tools** — schedulers and AutoML optimize *ahead of time* or
  inside the trainer; they rarely expose a small, auditable **outer loop** that
  watches emitted metrics and mutates a live run with a logged decision→effect
  trail.

So we under-use information that appears *during* training, and we cannot
scientifically compare “steered” vs “left alone” because the intervene layer is
not a first-class object.

## Claim (working — freeze before scale)

**A thin, auditable outer-loop intervene layer** — observe stream of training
metrics → apply a predeclared rule or policy → mutate run hyperparameters live
→ log decision time, action, and effect lag — **improves controllable training
outcomes** (reliability, efficiency, and/or human oversight) relative to (i)
unsteered runs and (ii) opaque automatic schedulers, **without requiring the
policy to live inside the trainer**.

**Not claiming (yet):** SOTA accuracy from a magic LR cut; replacing optimizers;
that our *toy* synthetic FAIL transfers to real models.

Toy protocol (`docs/protocol.md`) only validates the harness. The scale claim
and metrics will be re-frozen before GPU matrix scoring.

## Why NeurIPS / ICML should care

1. **Training as a closed-loop system.** Much of modern ML systems research
   treats the run as an open loop. Making intervention a measurable object
   connects optimization, AutoML, and MLOps with experimental discipline.
2. **Reproducibility crisis, but for *process*.** We obsess over final
   checkpoints; we rarely ship the *decision log* that shaped them. An open
   intervene schema is a concrete reproducibility primitive.
3. **Human–AI control without full AutoML.** Many labs want steerable
   automation they can audit — not a black-box hyperparameter agent. A minimal
   outer loop is the right abstraction for that middle ground.
4. **Negative and positive results both publishable** if pre-registered: when
   live intervene *hurts* (our toy already shows this) is as important as when
   it helps — if the protocol is clean.
5. **Artifact that outlives the paper.** A loved CLI/TUI with stable JSONL
   metrics + intervention logs is something the community can fork policies
   into — higher impact than a one-off training recipe.

## Method sketch

```
trainer ──metrics.jsonl──► intervene layer ──edit config / signal──► trainer
                │
                └── interventions.jsonl (decision step, rule, action, effect lag)
```

- **Trainer-agnostic:** file-tail, callback, or adapter (HF / torch / aq).
- **Policies:** predeclared rules first; learned / LLM policies later as
  ablations, not the only story.
- **Eval:** multi-seed steered vs unsteered vs strong baselines (cosine/warmup
  schedules, ReduceLROnPlateau, early stopping, human-in-the-loop), with
  pre-registered primary metrics and full ablations on trigger/timing/action.

## Tool thesis (same project)

If the science is “intervene should be first-class,” the product is the daily
driver: `pip install`, one-command demo, clear TUI, boring-reliable logs, docs
people finish. Paper without adoption is half the shot; tool without evidence
is a blog post.

## Related work — clusters and gaps

We are not inventing “change LR sometimes.” The gap is making **live outer-loop
intervention** a first-class, comparable, trainer-agnostic object.

### What exists

| Cluster | Examples (illustrative) | What they optimize |
|---------|-------------------------|-------------------|
| **In-trainer schedules** | cosine / warmup / step decay; ReduceLROnPlateau | LR as a function of step or plateau *inside* the optimizer loop |
| **AutoML / HPO** | Hyperband, ASHA, Optuna, PBT, population-based training | Search or evolve hyperparameters across trials; often kill/replace trials |
| **Early stopping & dashboards** | patience rules; WandB / MLflow / TensorBoard | Stop or *observe*; human still owns most live edits |
| **RL / bandit controllers** | learned LR / batch controllers; “learning to optimize” | Policy usually coupled to trainer internals or simulators |
| **MLOps orchestration** | Kubeflow, Ray Train, custom job wrappers | Launch, retry, scale — rarely a logged intervene *policy* mid-step |
| **Agent / LLM ops (nascent)** | chat agents that shell into training boxes | Powerful but not pre-registered; weak decision→effect audit trail |

### Gaps (where we sit)

1. **Outer vs inner.** Schedulers and many L2O methods live *inside* the
   training step. Practitioners’ real interventions are often *outside*
   (edit a config the trainer reloads, SIGUSR, separate controller). That
   outer loop is under-specified scientifically.
2. **Trial search ≠ live mutate.** HPO/PBT compare many runs; they do not
   cleanly answer: “given *this* run, does a predeclared live edit help?”
3. **Observe ≠ intervene.** Experiment trackers log metrics; they do not
   standardize *actions* with decision step, effect lag, and restore semantics.
4. **Reproducible process.** Papers report final metrics; few ship an
   intervention log that another lab can replay against the same metric stream.
5. **Human-compatible automation.** Full AutoML and agentic “just fix my run”
   systems are hard to audit. A minimal rule/policy layer with mandatory logs
   is the missing middle for labs that want control without opacity.
6. **Negative-result hygiene.** When live intervene hurts (our toy already
   fails), literature often under-reports it because the intervene surface was
   never pre-registered.

**Positioning one-liner:** not a new optimizer — a **measurable intervene
interface** + policies + evidence that the interface is useful (or harmful)
under a frozen protocol.

## Eval protocol (detailed — freeze before scale)

Toy protocol (`docs/protocol.md`) remains the **harness test**. Scale scoring
uses the protocol below (to be copied into `docs/protocol.scale.md` when
setting/metrics are locked).

### 0. Pre-registration (before any GPU seed used for claims)

Freeze in writing:

- primary setting (model, data, max steps/epochs, hardware class);
- primary metric(s) and direction;
- baseline list;
- intervene action space;
- seed list;
- stop rules (budget, early abort).

No claim-facing seed may run before this freeze. Exploratory runs go under
`results/explore/` and cannot enter the main table without a protocol amendment
logged in this file.

### 1. Primary scale setting — **LOCKED: S1**

**Headline setting ID:** `S1-qwen15b-alpaca-lora`

| Field | Choice | Rationale |
|-------|--------|-----------|
| **Family** | S1 — LM LoRA instruction fine-tune | Where humans already babysit runs; max NeurIPS/ICML relevance |
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` | Strong small instruct model; Apache-2.0-friendly weights; fits 1×24GB with LoRA |
| **Method** | LoRA (r=16, α=32, dropout=0.05) on attn + MLP proj | Standard, cheap to multi-seed; not full FT |
| **Data** | [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 52k; **95/5** seeded split → train / val | Universally recognized; public; intervene can watch **val loss** |
| **Optim** | AdamW, lr=2e-4, weight_decay=0.0, warmup_ratio=0.03 (C0/C1 diverge on schedule) | Common LoRA defaults |
| **Budget** | **max_steps = 2000**; eval/log every **50** steps; seq len 512; effective batch **32** (e.g. micro 4 × accum 8) | Long enough for mid-run intervene; short enough for 5–10 seeds on one GPU |
| **Hardware class** | **1× NVIDIA GPU ≥24GB** (A10 / L40 / RTX 4090 / A100-40) | Single-GPU reproducibility; no multi-node confounds in v1 |
| **Wall-clock (est.)** | ~1–3 h / seed / condition on 24GB class | Matrix of 5 seeds × 5 conditions ≈ days, not weeks |

**Secondary (generality — not headline, run after S1 works):** `S2-cifar100-resnet18` — CIFAR-100, ResNet-18, 100 epochs, 1× GPU ≥8GB. Proves the outer loop is not LLM-only.

**Deferred:** S3 (RL / diffusion) until S1+S2 exist.

**Amendment rule:** changing model/data/steps/hardware class requires a dated note in this file and a new `results/scale_<tag>/` prefix — do not mix with prior scale artifacts.

See also `docs/protocol.scale.md` (setting freeze; metrics still to finalize).

### 2. Baselines & conditions — **LOCKED for S1**

Named baselines we must **beat** or **match** (ε frozen with metrics). Ours is
not a baseline — it is the method under test.

#### Must-beat

| ID | Name | Spec (S1) | Role |
|----|------|-----------|------|
| **C0** | `B-UNSTEERED` | Constant lr=2e-4 after warmup_ratio=0.03 (linear warmup then **flat**); no live edits; full 2000 steps | Sanity floor — if we lose to this, method fails |

#### Must-match-or-beat (strong non-ours; take the best of these as the bar)

| ID | Name | Spec (S1) | Role |
|----|------|-----------|------|
| **C1** | `B-COSINE` | Linear warmup_ratio=0.03 then **cosine decay to 0.1× peak LR** over 2000 steps; no live outer-loop edits | Standard “competent schedule” strawman-killer |
| **C2** | `B-PLATEAU` | Peak lr=2e-4, warmup_ratio=0.03, then **ReduceLROnPlateau** on **val loss** (factor=0.5, patience=3 evals [=150 steps], threshold=1e-3, cooldown=1 eval, min_lr=1e-6) | Classic adaptive in-trainer baseline |
| **C3** | `B-EARLYSTOP` | Same LR as C0; **early stop** on val loss (patience=5 evals [=250 steps], min_delta=1e-3); checkpoint = best val; no other HP edits | “Just stop” control — common practitioner move |

**Match rule (once ε frozen):** C4 mean primary metric ≥ best(C1,C2,C3) − ε  
**Beat rule:** C4 mean primary metric > C0 (+ optional significance test).

#### Ours (method — not baselines)

| ID | Name | Spec | Role |
|----|------|------|------|
| **C4** | `OURS-RULE` | Outer-loop predeclared rule; same start as C0; live mutate + intervention log | **Primary method** |
| **C5** | `OURS-HUMAN` | Same log schema/UI; human intervene only; **10 min wall-clock cap** per run watching; scripted instructions | Human-compatible ablation |
| **C6** | `OURS-POLICY` | Optional learned/LLM policy | Ablation only — **not** required for submission |

#### Claim-facing matrix (minimum)

| Tier | Conditions | When |
|------|------------|------|
| **Day-one / compute-light** | C0, C1, C4 | First GPU matrix |
| **Submission-complete** | C0, C1, C2, C3, C4 | Required before venue |
| **Nice-to-have** | + C5 | If bandwidth |
| **Optional ablation** | + C6 | Only if it adds a story |

Fairness: same init seed, data-order seed, base config, LoRA init across
conditions until first *allowed* diverge (schedule or intervene). No peeking
at held-out **test** labels (val is allowed for C2/C3/C4 triggers).

### 3. Success metrics — **LOCKED for S1**

Lower is better unless noted.

| Rank | Metric | Definition | Use |
|------|--------|------------|-----|
| **Primary** | `best_val_loss` | Minimum `val_loss` among eval points (every 50 steps) | Score beat/match |
| **Primary-efficiency (report always)** | `steps_to_target` | First step where `val_loss ≤ τ`, τ = median C0 best_val_loss across seeds in this matrix (computed once after all C0 seeds finish, then frozen in `matrix_summary`) | Efficiency story; not used for pass/fail until secondary analysis |
| **Secondary** | `final_val_loss` | Val loss at last completed eval | Stability |
| **Secondary** | `final_train_loss` | Train loss at step 1999 (or last step) | Overfit check |
| **Secondary** | `wall_s`, `gpu_hours` | Wall clock; GPU-hours ≈ wall_s/3600 on 1 GPU | Cost |
| **Secondary** | intervene stats | count, decision steps, mean decision→effect lag | Method process |
| **Process (tool)** | schema_ok | 1 iff metrics + snapshot + interventions (C4/C5) validate | Adoption claim |
| **Optional later** | `alpaca_eval_lite` | Fixed 100-prompt preference vs C0 ref (only if budget) | Not required day-one |

**Direction for comparisons:** better = lower `best_val_loss`.

**Margin ε (match vs best of C1–C3):**  
`best_val_loss(C4) ≤ best_val_loss(best_baseline) × (1 + 0.02)`  
i.e. within **+2% relative** of the best strong baseline (lower is better, so we may be at most 2% worse and still “match”).

**Beat C0:** `best_val_loss(C4) < best_val_loss(C0)` (strict).

### 4. Seeds, stats, and matrix call — **LOCKED**

- **Seeds:** `{7, 11, 13, 17, 19}` (same as toy). If budget allows, add `{23, 29, 31, 37, 41}` as extended set — pre-register before running.
- Report mean ± 95% bootstrap CI across seeds; per-seed table always.
- **Per-seed verdict:**
  - **pass** if beat C0 **and** match best(C1–C3) within ε (on day-one, best of available strong baselines — at least C1);
  - **fail** if `best_val_loss(C4) > best_val_loss(C0) × 1.02` (worse than C0 by >2% rel) **or** clearly worse than best strong baseline by >ε;
  - **inconclusive** if no intervene fired when required, run crashed, or missing artifacts.
- **Matrix call (5 seeds):** pass if ≥3/5 seed-pass; fail if ≥3/5 seed-fail; else inconclusive.
- Day-one strong baseline set = {C1}; submission = {C1,C2,C3}.

### 5. Ablations (minimum set)

| Ablation | Question |
|----------|----------|
| A1 Trigger | Different observe signals (loss vs val vs grad norm) |
| A2 Timing | Early vs late intervene windows |
| A3 Action | LR cut vs freeze vs early stop vs batch tweak |
| A4 Automation | C4 rule vs C5 human vs C6 policy |
| A5 Coupling | File-tail outer loop vs in-callback adapter |
| A6 Budget | Same wall-clock vs same step count |

### 6. Artifacts (must reproduce the paper table)

Per run directory:

```
metrics.jsonl           # step, loss, lr, wall_ms, + scale fields
config.snapshot.json    # start HPs
interventions.jsonl     # decision_step, rule_id, action, effect_step, lag
compare.json            # vs paired unsteered (script output)
```

One command regenerates tables/figures from artifacts (no notebook-only path).

### 7. Threats to validity (state in paper)

- Confounding schedule strength (C1–C2 must be competent, not strawmen).
- Metric delay / logging lag mistaken for method delay.
- Overfitting the intervene rule to one setting (need S2 or ablation).
- Human condition (C5) expertise confounds — timebox and script instructions.
- Compute asymmetry — report GPU-hours per condition.

### 8. Ethics / non-goals

- Not a license to silently mutate production training without audit logs.
- Not claiming safe autonomous training of dangerous models.
- Tool defaults to explicit logs; dry-run available.

## Status

| Layer | State |
|-------|--------|
| Harness (synthetic) | Working; toy matrix **FAIL** |
| Related-work gap story | Drafted |
| Scale setting + baselines + metrics + S1 rule | **LOCKED** in `docs/protocol.scale.md` |
| Tool MVP spec | **LOCKED** in `docs/tool-mvp.md` |
| Venue draft | Not started from scale data |

## Next lock (before GPU)

1. ~~Setting / baselines / metrics / rule / tool MVP spec~~ → done.
2. Implement S1 trainer adapter + wire `OURS-RULE` (needs compute for real runs).
3. Package `pip install -e .` entrypoints per `docs/tool-mvp.md`.
4. Day-one matrix: C0, C1, C4 on seeds `{7,11,13,17,19}`.

---

*Living brief. Update as the claim hardens; do not let the paper drift from
what is written here.*
