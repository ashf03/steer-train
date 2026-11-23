# Intervention Log

## Intervention 1

- **Timestamp:** 2026-10-22T07:12:21Z
- **Run / latest observed step:** steered; step 67 emitted before the edit
- **Observed evidence:** Loss at step 67 was 0.029457 after a noisy low-loss
  stretch (0.041461 at step 65 → 0.052327 at step 66 → drop at 67).
- **Hypothesis:** Once loss is near zero, reducing learning rate should damp
  noise-driven oscillations and test a convergence/stability tradeoff.
- **Exact action:** Edited `configs/default.json` `lr` from `0.05` to `0.02`;
  preserved `steps=100`, `sleep_seconds=0.1`, and `seed=7`.
  (Originally written under the legacy path `runs/config.json`.)
- **Expected effect:** Later metric rows show `lr: 0.02`, with smaller loss
  fluctuations and stable near-zero convergence.
- **Outcome:** Change first appeared at step 87 (`lr=0.02`, loss `0.044701`).
  Steps 87–99 used `lr=0.02`. Final loss `0.031312` (baseline final
  `0.022369`). Post-change loss ranged `0.012672`–`0.057607` — still noisy;
  clear damping was **not** supported. Config restored to `lr=0.05` afterward.

## Intervention (auto protocol)

- **Timestamp:** 2026-11-12T12:07:19Z
- **Mode:** steered
- **Evidence steps:** [67, 68, 69]
- **Action:** `lr` 0.05 → 0.02 via TUI (`configs/default.json`)
- **Metrics:** `results/steered/metrics.jsonl`
- **Outcome:** _(fill after run)_

## Intervention (steer_watch dry-run)

- **Timestamp:** 2026-11-23T14:57:08Z
- **Decision step:** 69
- **Evidence steps:** [67, 68, 69]
- **Action:** `lr` 0.05 → 0.02 (dry-run — config not written)
- **Config:** `/Users/ashm/work/st/configs/default.json`
- **Metrics:** `/Users/ashm/work/st/results/steered/metrics.jsonl`
- **Decision→effect delay (steps):** _(pending — waiting for lr in metrics)_
- **Outcome:** _(fill after run / compare_runs)_

## Intervention (steer_watch)

- **Timestamp:** 2026-11-23T14:57:13Z
- **Decision step:** 69
- **Evidence steps:** [67, 68, 69]
- **Action:** `lr` 0.05 → 0.02
- **Config:** `/Users/ashm/work/st/configs/default.json`
- **Metrics:** `/Users/ashm/work/st/results/steered/metrics.jsonl`
- **Decision→effect delay (steps):** _(pending — waiting for lr in metrics)_
- **Outcome:** _(fill after run / compare_runs)_

## Intervention (steer_watch)

- **Timestamp:** 2026-11-23T14:57:35Z
- **Decision step:** 69
- **Evidence steps:** [67, 68, 69]
- **Action:** `lr` 0.05 → 0.02
- **Config:** `/Users/ashm/work/st/configs/default.json`
- **Metrics:** `/Users/ashm/work/st/results/steered/metrics.jsonl`
- **Decision→effect delay (steps):** _(pending — waiting for lr in metrics)_
- **Outcome:** _(fill after run / compare_runs)_

## Intervention (steer_watch)

- **Timestamp:** 2026-11-23T14:57:55Z
- **Decision step:** 69
- **Evidence steps:** [67, 68, 69]
- **Action:** `lr` 0.05 → 0.02
- **Config:** `/Users/ashm/work/st/configs/default.json`
- **Metrics:** `/Users/ashm/work/st/results/steered/metrics.jsonl`
- **Decision→effect delay (steps):** _(pending — waiting for lr in metrics)_
- **Outcome:** _(fill after run / compare_runs)_
