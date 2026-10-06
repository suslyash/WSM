# TASK-010A method specification and source-grounding gate

## Status

**BLOCKED BEFORE IMPLEMENTATION AND PRODUCTION.**

TASK-010A authorizes a post-closure, DEV-only extension only if PCGrad,
CAGrad, GradNorm, and DB-MTL can each be implemented faithfully from supplied
sources. The repository contains the Smooth Tchebycheff paper
`docs/2402.19078v3.pdf` and the gradient-based multi-objective survey
`docs/2501.10945v3.pdf`; it does not contain the primary papers or an accepted
implementation for any of the four requested methods. A repository-wide search
found no supplied `PCGrad`, `CAGrad`, `GradNorm`, `DB-MTL`, or
Dual-Balancing implementation/source artifact.

No source, configuration, pseudo cache, model role, frozen ledger, paper, or
`src/audio` file was changed. No model was instantiated, no dataset loader was
iterated, no checkpoint was loaded, and no training, inference, DEV metric, or
Test operation was run.

## Frozen comparator check

The accepted balancing comparison uses the fixed R4 composition and seeds
42/43/44. The verified frozen three-seed DEV Mean values are Equal
`0.7791726667`, static-STCH `0.7803973333`, Progress `0.7847510000`, and
RA-STCH `0.7725573333`.

Equal, Progress, and RA-STCH are recorded in `docs/STAGE6_CLAIM_LEDGER_EN.md`
and `docs/PROGRESS_EN.md`. Static-STCH is recovered from the accepted selected
DEV rows in `docs/PROGRESS_EN.md`: `0.787281`, `0.776064`, and `0.777847` for
seeds 42, 43, and 44 respectively (arithmetic mean shown above). No historical
method was rerun.

## Source review

### PCGrad

- **Supplied locator:** `docs/2501.10945v3.pdf`, Section 3.2.2, p. 11,
  Equation (22); primary citation [217], Yu et al. (2020), *Gradient Surgery
  for Multi-Task Learning*.
- **Survey equation:** for conflicting task gradients
  `g_hat_i^T g_j < 0`, update
  `g_hat_i <- g_hat_i - (g_hat_i^T g_j / ||g_j||^2) g_j`; then sum corrected
  gradients.
- **Acts on:** actual model-parameter task gradients; it changes gradients,
  not loss weights.
- **Missing for faithful implementation:** the source-faithful ordering/
  permutation semantics and exact aggregation/edge-case algorithm required by
  TASK-010A are not specified by the supplied survey. The primary paper or an
  accepted primary implementation is required; deterministic ordering cannot
  be invented.

### CAGrad

- **Supplied locator:** `docs/2501.10945v3.pdf`, Section 3.2.1, p. 10,
  Equations (14)--(15); primary citation [106], Liu et al. (2021),
  *Conflict-Averse Gradient Descent for Multi-task Learning*.
- **Survey description:** it constrains the direction around the average
  gradient `g0` within `c ||g0||`, and presents a simplex subproblem followed
  by a conflict-averse direction.
- **Acts on:** actual model-parameter task gradients; it is a gradient
  weighting/conflict-averse update, not a loss-weight method.
- **Missing for faithful implementation:** the supplied source does not fix
  the canonical/default conflict-aversion constant `c`, nor provide an
  unambiguous implementation-level subproblem/direction expression sufficient
  for the required deterministic toy-reference test. TASK-010A prohibits a
  substitute or a hyperparameter guess.

### GradNorm

- **Supplied locator:** `docs/2501.10945v3.pdf`, Section 3.2.1, p. 11,
  Equation (21); primary citation [21], Chen et al. (2018), *GradNorm:
  Gradient Normalization for Adaptive Loss Balancing in Deep Multitask
  Networks*.
- **Survey equation:** learn positive task weights by minimizing the sum of
  differences between scaled task gradient norms and `c * r_i^gamma`, where
  `r_i` is the normalized inverse training rate and `c` is a detached average
  scaled gradient norm.
- **Acts on:** dynamic loss weights and therefore scaled task gradients.
- **Missing for faithful implementation:** the canonical/default `gamma`
  (GradNorm alpha), task-weight learning rate/optimizer update, and required
  initialization/renormalization details are absent from supplied sources.
  TASK-010A explicitly forbids selecting them by a sweep or from memory.

### DB-MTL

- **Supplied locator:** `docs/2501.10945v3.pdf`, Section 3.2.1, p. 11;
  primary citation [94], Lin et al. (2023), *Dual-Balancing for Multi-Task
  Learning*, arXiv:2308.12029.
- **Survey equation:** normalize objective gradients using
  `lambda_i = gamma / ||g_i||` with `gamma = max_i ||g_i||`, so all scaled
  objective gradients have the same norm.
- **Acts on:** gradient weighting/normalization according to the survey.
- **Missing for faithful implementation:** the primary algorithm’s complete
  update, canonical constants, two-task/task-specific-parameter handling, and
  AdamW interaction are not supplied. The survey explicitly notes that update
  magnitude affects performance, so choosing a substitute magnitude would be
  an unauthorized tune/approximation.

## Required blocker resolution

Before implementation can start, the manager must provide or explicitly
authorize primary, versioned sources (or a faithful accepted implementation)
that establish:

1. PCGrad ordering and edge-case algorithm;
2. CAGrad's canonical `c` and deterministic solver/direction formula;
3. GradNorm's canonical alpha, task-weight update rule, and learning rate;
4. DB-MTL's full algorithm, constants, task-specific-parameter semantics, and
   optimizer interaction.

Until then, adding a DEV-only wrapper, loss path, optimizer hook, tests, or
configs would risk labeling an approximation as a published method. Therefore
the mandatory pre-production firewall and all 12 production invocations are
not authorized.
