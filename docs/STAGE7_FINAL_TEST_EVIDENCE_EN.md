# Stage 7 Final Test Evidence — TASK-007C

## Result and boundary

STAGE-7 FINAL TEST EVALUATION INCOMPLETE

The mandatory DEV-only preflight stopped before any Test loader or Test
dataset iteration. The exact blocker is the frozen shared-fusion seed43
checkpoint/config reproduction:

- checkpoint:
  logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b/checkpoints/epoch=10_dev_mean_score=0.7713.pt
- checkpoint SHA256:
  ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee
- config:
  configs/wsm_mm_pd_dep_v1/ablations/56_shared_fusion_seed43.yaml
- frozen DEV D/P/Mean:
  0.724496/0.818152/0.771324
- current exact configured shared-mode reproduction:
  0.7229314130/0.8181521375/0.7705417752
- absolute D/Mean mismatches:
  0.0015645870/0.0007822248, exceeding 0.0005.

A narrow DEV-only diagnostic also evaluated the same checkpoint with
task-aware fusion enabled; it produced D/P/Mean
0.7183358212/0.8181521375/0.7682439793 and therefore does not resolve the
mismatch. No source/config/checkpoint/cache change is authorized.

## Firewall status

Evaluator script:
scripts/common/evaluate_stage7_final_test.py

The script exposes --help, contains 15 frozen entries, verifies checkpoint and
config SHA256 values plus the pseudo-cache SHA256, implements raw-logit
thresholding, historical two-class margin/argmax equivalence, sparse masks,
Student-t arithmetic, atomic output, and overwrite refusal. Static and
synthetic checks passed. The all-15 DEV preflight was attempted with the exact
semantic RAMPS DataModule. Audio and R4 entries, and shared seed42, passed
before the shared seed43 mismatch stopped execution.

No firewall commit was eligible to authorize Test because the DEV gate failed.
No standardized Final Test invocation occurred; invocation count is 0.
The external report was not created, so report SHA256 is not applicable.

## Frozen roles and boundaries

Pre-Test roles remain unchanged and were not revised:

1. primary paper candidate: equal-parameter shared fusion;
2. secondary multimodal reference: full R4 trial012;
3. baseline: temporal audio.

No training, Test evaluation, recalibration, threshold fitting, model selection,
role revision, retry, or Test-driven decision occurred. The prior historical
Test-line visibility procedural exception remains separate:
FINAL TEST VALUES NOT USED FOR TASK-007B SELECTION OR RECOVERY.

Stage 7 remains active pending manager review. Final Test remains locked by the
failed pre-Test DEV gate.
