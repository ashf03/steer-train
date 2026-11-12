# Live-Steering Protocol

## Frozen claim (do not change mid-matrix)

**Claim:** A predeclared late-stage learning-rate cut (`0.05 → 0.02`) reduces
post-change loss volatility versus a constant-`lr=0.05` baseline, without
making final loss more than **10%** worse than baseline.

**Scope:** Synthetic `src/train.py` only. No aq. We (human or our scripts)
own the intervene loop — not axi mission/daemon.

### Metrics

| Metric | Definition |
|--------|------------|
| Final loss | `loss` at last step (step 99) |
| Post-change window | Steps **87–99** inclusive (13 steps), matched on both runs |
| Volatility | Sample standard deviation of `loss` in that window |

Intervene effect is only scored on that window so baseline and steered are
compared on the same steps. (Steered uses `lr=0.02` there; baseline stays `0.05`.)

### Verdict thresholds (per seed, then aggregate)

| Verdict | Rule |
|---------|------|
| **Pass** | `vol_steered < vol_baseline` **and** `final_steered ≤ 1.10 × final_baseline` |
| **Fail** | `final_steered > 1.10 × final_baseline` **or** `vol_steered ≥ vol_baseline` |
| **Inconclusive** | Runs incomplete, intervene missing/late (first `lr=0.02` row after step 88), or seeds disagree with no clear majority |

**Matrix call:** After seeds `{7, 11, 13, 17, 19}` — **pass** if ≥3/5 seeds pass;
**fail** if ≥3/5 fail; else **inconclusive**.

### Predeclared intervene rule (steered only)

1. Start from the same `configs/default.json` as baseline (`lr=0.05`, `steps=100`, `seed=<k>`, `sleep_seconds` as chosen).
2. Watch **only** `results/.../steered` metrics already written. **No peek** at baseline JSONL.
3. **Trigger:** once `loss < 0.05` for **3 consecutive** emitted steps **and** `step ≥ 60`, edit config `lr` → `0.02` (one shot; do not edit again).
4. Leave `steps` / `seed` / `sleep_seconds` alone.
5. After the run ends, restore `lr` to `0.05`.
6. Log the intervene in `docs/interventions.md` (timestamp, trigger steps, action, outcome).

If the trigger never fires by step 99 → that seed is **inconclusive** (no steer).

---

## Layout

| Path | Role |
|------|------|
| `src/train.py` | Headless trainer (`--sleep` overrides JSON delay) |
| `src/tui.py` | Full-screen live steer TUI (auto + manual) |
| `configs/default.json` | Live-editable hyperparameters |
| `results/baseline/metrics.jsonl` | Uninterrupted control run (pilot) |
| `results/steered/metrics.jsonl` | Live-steered run (pilot) |
| `docs/interventions.md` | Every intervene decision + outcome |
| `paper/draft.md` | Methods / Results / Limitations |

Never mix baseline and steered metrics in one file.

## Baseline

1. Ensure `configs/default.json` matches the planned start (`lr=0.05`, seed set, …).
2. Run:

   ```bash
   python3 src/train.py --reset --metrics results/baseline/metrics.jsonl
   ```

3. Do not edit configuration, pause, restart, or otherwise intervene.
4. Record completion status and summary statistics.

## Steered run

1. Restore the same starting `configs/default.json` used by the baseline.
2. Run:

   ```bash
   python3 src/train.py --reset --metrics results/steered/metrics.jsonl
   ```

3. Apply the **predeclared intervene rule** above (only).
4. Do not alter existing metric records.
5. Continue until the planned step budget completes, or document why it stopped.
6. Restore `configs/default.json` to `lr=0.05` after the run.

## Required intervention log

For every intervention, append to `docs/interventions.md`:

- wall-clock timestamp;
- run and latest observed step;
- observed evidence (the 3 trigger losses + steps);
- exact action;
- expected effect;
- later outcome vs frozen thresholds.

## Evidence and reporting

- Preserve raw JSONL under `results/`.
- Calculate paper numbers only from preserved artifacts + the frozen metrics.
- Distinguish observed metrics, derived statistics, and interpretation.
- Draft must include Methods, Results, and Limitations.
