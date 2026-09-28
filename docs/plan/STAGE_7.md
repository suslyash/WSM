# WSM Stage 7 Detailed Plan

Status: **LOCKED**.

This is a stage-specific detail file extracted from the former monolithic `docs/PLAN.md`.
Read it only when the active task belongs to this stage or when historical plan detail is explicitly needed.
Current project status is authoritative in [../PROGRESS_EN.md](../PROGRESS_EN.md).

### Stage 7 — Final evaluation

After freeze:

1. train final compared methods on at least 5 seeds;
2. choose every checkpoint only by DEV/Mean_Score;
3. run separate TEST_NONE/SOFT/HARD evaluation once;
4. report per-task UAR/MF1/Score and Mean_Score;
5. report mean, standard deviation, confidence interval, and paired comparison where applicable;
6. trace every table entry to config, commit, checkpoint, and MLflow run.

Gate: no Test-driven revision; complete traceability; claims match annotation evidence.
