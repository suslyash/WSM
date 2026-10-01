# DEPART-comparison-ready metric table

## 1. WSM final methods — UAR-to-UAR / MF1-to-MF1

All values below are five-seed arithmetic means **derived from frozen recorded per-seed rows** and cross-checked against frozen Score/Mean summaries. They are proportions and descriptive reporting values; they make no comparative decision.

| Method | DEV: D UAR / D MF1 / P UAR / P MF1 | TEST_NONE: D UAR / D MF1 / P UAR / P MF1 | TEST_SOFT: D UAR / D MF1 / P UAR / P MF1 | TEST_HARD: D UAR / D MF1 / P UAR / P MF1 |
|---|---|---|---|---|
| temporal audio | 0.722073 / 0.721367 / 0.805908 / 0.811231 | 0.753833 / 0.752076 / 0.862977 / 0.833483 | 0.776244 / 0.775606 / 0.865434 / 0.834990 | 0.789271 / 0.787308 / 0.879147 / 0.841345 |
| full R4 trial012 | 0.731494 / 0.731007 / 0.839310 / 0.858416 | 0.790379 / 0.791575 / 0.809679 / 0.822376 | 0.802983 / 0.801609 / 0.812814 / 0.827964 | 0.807218 / 0.804870 / 0.835311 / 0.839669 |
| equal-parameter shared fusion | 0.737946 / 0.737435 / 0.834079 / 0.854199 | 0.798528 / 0.799937 / 0.806691 / 0.818858 | 0.813414 / 0.812100 / 0.810377 / 0.825505 | 0.817077 / 0.814755 / 0.834925 / 0.839173 |

The final methods remain separate from the temporal-audio baseline. Full R4 and equal-parameter shared fusion each have `295239` trainable parameters; shared fusion has `task_aware_fusion: false`, while full R4 has task-aware fusion enabled. The standardized Final Test was invoked exactly once; no post-Test role revision occurred.

## 2. Internal video references

| Method | DEV depression UAR / MF1 / Score | DEV Parkinson UAR / MF1 / Score | DEV Mean | TEST_NONE / TEST_SOFT / TEST_HARD Mean |
|---|---|---|---:|---:|
| DEPART-like V1 | 0.662558 / 0.661287 / 0.661923 | 0.737474 / 0.751778 / 0.744626 | 0.703274 | 0.732454 / 0.742796 / 0.751864 |
| DEPART-like V2 | 0.620401 / 0.619801 / 0.620101 | 0.785231 / 0.800854 / 0.793043 | 0.706572 | 0.722250 / 0.721935 / 0.730925 |

For V1/V2 historical Test monitoring, task-level UAR/MF1/Score is **NOT FOUND IN TRACKED EVIDENCE** and is blank in the CSV.

## 3. Published DEPART context

| DEPART setting | Depression UAR / MF1 (%) | Parkinson UAR / MF1 (%) |
|---|---:|---:|
| Multitask, original Test | 82.25 / 81.79 | 66.82 / 59.03 |
| Multitask, manually cleaned Test | 83.67 / 82.85 | 80.71 / 76.19 |
| Best single-task, original Test | 85.55 / 84.83 | 78.58 / 74.42 |
| Best single-task, manually cleaned Test | 88.08 / 86.91 | 77.96 / 73.75 |

Published DEPART is contextual only; it is not a matched WSM leaderboard.

DEPART UAR is not WSM Mean_Score; its formulation/evaluation differs from WSM's independent masked disease heads; manually cleaned DEPART Test is not established as WSM TEST_HARD. A valid side-by-side table keeps UAR-to-UAR and MF1-to-MF1, retains protocol labels/units, and makes no ranking claim.

Sources: final rows are `docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md` and `docs/STAGE7_FINAL_TEST_EVIDENCE_EN.md`; V1/V2 are `docs/archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md`; DEPART is Tables 7--8 (pp. 16, 19) of `docs/BDCC-10-00089.pdf`.
