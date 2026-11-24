# Live Learning-Rate Steering in a Synthetic Training Loop

## Abstract

We compared an uninterrupted baseline with a live-steered run of a 100-step
synthetic training process. Both runs used the same seed and an initial
learning rate of 0.05. In the steered run, the learning rate was reduced to
0.02 during a low but noisy loss regime; the change appeared in emitted
metrics for steps 87–99. The intervention was applied successfully without
restarting training, but it did not improve the final result. Steered final
loss was 0.031312 versus 0.022369 for the baseline (~39.98% higher). The short
post-change segment also provided no clear evidence of damped fluctuations.
This experiment demonstrates successful live configuration steering while
leaving its proposed stabilization benefit unsupported.

## Methods

We compared two 100-step runs of `src/train.py` using seed 7, an initial loss
of 2.0, and an initial learning rate of 0.05 (see `configs/default.json`). At
each step, the script updated loss as the previous loss multiplied by
`exp(-lr)`, added Gaussian noise with standard deviation 0.01, and clipped the
result at zero. Metrics were appended as JSON Lines records containing step,
loss, and learning rate.

The baseline ran uninterrupted with learning rate 0.05; records are preserved
in `results/pilot/baseline/metrics.jsonl`. The steered run began from the same
configuration and seed; records are in `results/pilot/steered/metrics.jsonl`.
Decisions used only already-emitted metrics. After observing a low but noisy
loss regime through step 67, the runtime configuration was edited from
learning rate 0.05 to 0.02. Because the trainer reloads the config file before
each step, the change took effect without restarting; it first appeared in the
metric record at step 87 and remained active through step 99. The
intervention, expected effect, and outcome are recorded in
`docs/interventions.md`.

## Results

Both runs completed 100 steps (0 through 99). Their records were identical
through step 86 because they shared the same seed and learning rate before the
live configuration change took effect.

The baseline loss decreased from 1.899900 at step 0 to 0.022369 at step 99
(~98.82% reduction). It first reached zero at step 73 and remained in a low
but noisy regime thereafter.

The steered run used learning rate 0.02 for steps 87–99. Its final loss was
0.031312, versus 0.022369 for the baseline (+0.008944, ~39.98% relative).
Within the 13-step post-change segment, steered loss ranged from 0.012672 to
0.057607. The run remained in a low-loss regime, but this short segment did
not provide clear evidence that reducing the learning rate damped
fluctuations.

These results do not support a claim that this late intervention improved
final loss. They show that a live learning-rate edit was successfully applied
and preserved in the emitted metrics, while its proposed stabilization benefit
remained inconclusive.

## Limitations

This experiment used one synthetic process, one random seed, and one
baseline/steered pair. The intervention took effect only for the final 13
steps, leaving little post-intervention data. The loss floor at zero and
additive noise make simple fluctuation statistics sensitive to the current
loss level. Because the two trajectories diverged only after the intervention,
the baseline is a useful deterministic control for this seed, but one run
cannot establish generality.

The intervention was chosen adaptively from observed metrics rather than from
a fully predeclared numerical threshold. Its rationale was documented before
outcome assessment, but adaptive timing can still introduce researcher
discretion. The delayed appearance of the edited learning rate — decision at
observed step 67, first recorded use at step 87 — also reduces precision about
operational timing. Future work should automate trigger detection and
configuration writes, repeat across seeds, intervene earlier, and compare
post-change volatility over longer matched windows.
