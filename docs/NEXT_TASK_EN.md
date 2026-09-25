# TASK-005E-BUNDLE-C2: Audit Corrected R3 Checkpoint Identity and Freeze the Promotion Conclusion

## Authority and branch

This is one evidence-only corrective task.

Required branch:

    codex/task-005e

Reuse of the existing branch is explicitly authorized.

Preserve all prior commits, including:

- corrective firewall `bd346482f2cdaa1c5edb0c0e416ef4a681cc6e7f`;
- manager-preservation merge `ed869986731e94fd50bd70b9decb1751360cc836`;
- corrected production evidence `e0482a42f94b4930aa2f21be5f639f420eaf5201`;
- manager review `182dd69c2922dcc9e2ac5050e1d9b682d6d2e8ae`.

Do not reset, rebase away, force-push, or touch main/master.

## Goal

Close the one remaining C1 acceptance gap without retraining:

1. identify the selected corrected R3-B checkpoints for true seeds 42, 43, and 44;
2. compute SHA256 for each selected checkpoint file;
3. prove the three selected checkpoint SHA256 values are pairwise distinct;
4. verify each corrected run's resolved config carries the intended seed and frozen R3-B method settings;
5. append the fixed PLAN promotion conclusion:
   - R3-B is not promoted because the DEV gain does not repeat across at least 3 seeds;
   - Stage 5 remains active;
   - R4 is not started in this task.

No model/source/config changes and no training are authorized.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5 and Section 7 Promotion Rule
5. docs/PROGRESS_EN.md through MANAGER-REVIEW-043
6. docs/NEXT_TASK_EN.md

## Allowed tracked files

Codex may modify only:

- docs/PROGRESS_EN.md

No source or config file may change.

## Forbidden actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify src/fusion.
- Do not modify src/common.
- Do not modify src/chimera_plugin.py.
- Do not modify any YAML.
- Do not run `chimera-ml train`.
- Do not execute an optimizer step.
- Do not launch any new production run.
- Do not recompute or change selected epochs/checkpoints.
- Do not alter R3 architecture/loss/weights.
- Do not use Test metrics for any decision.
- Do not start R4, Stage 6, Stage 7, Text/Description, or Final Test.
- Do not make significance, final-model, missing-label correctness, or comorbidity claims.

## Existing corrected run identities

Use the corrected C1 runs recorded in PROGRESS_EN.md and their MLflow IDs:

- R3-B seed42:
  - MLflow run ID `ac43bc78d6f7468997d7fe9f9345541d`
  - selected DEV epoch: 11
  - selected DEV D/P/Mean: `0.695808/0.893824/0.794816`

- R3-B seed43:
  - MLflow run ID `0febc7ff45774614bb792154bd7fd2cd`
  - selected DEV epoch: 5
  - selected DEV D/P/Mean: `0.694349/0.841176/0.767762`

- R3-B seed44:
  - MLflow run ID `bac2cf209158407ab3880bd065c036ca`
  - selected DEV epoch: 13
  - selected DEV D/P/Mean: `0.698378/0.838989/0.768684`

Do not substitute checkpoints from the earlier invalid exploratory bundle.

## 1. Locate and hash selected checkpoints

For each corrected R3-B run:

1. locate the exact selected checkpoint corresponding to the recorded selected DEV epoch;
2. compute full-file SHA256;
3. record absolute checkpoint path and SHA256;
4. assert all three SHA256 values are pairwise distinct.

If any selected checkpoint cannot be located or two checkpoint SHA256 values are identical, STOP and report blocked.

Also record each checkpoint file size in bytes.

## 2. Verify checkpoint payload identity

Load each selected checkpoint read-only on CPU.

Record:

- payload top-level keys;
- stored epoch if present;
- number of model-state keys;
- deterministic tensor-content digest of `model_state_dict`:
  - sort state keys;
  - hash each key name, dtype, shape, and raw CPU contiguous tensor bytes.

Assert the three tensor-content digests are pairwise distinct.

Do not write or mutate any checkpoint.

## 3. Verify resolved config identity

For each corrected B42/B43/B44 run, locate its resolved config saved in the run directory/snapshot and verify:

- top-level seed is exactly 42 / 43 / 44 respectively;
- model name is `wsm_av_r3_disease_query_model`;
- model params include:
  - audio_feature_dim 768
  - video_feature_dim 512
  - hidden_dim 192
  - gate_hidden_dim 192
  - dropout 0.2
  - num_tasks 2
- loss name is `wsm_r3_aux_agreement_loss`;
- `aux_weight == 0.25`;
- `agreement_weight == 0.10`;
- checkpoint/early-stopping monitor is only `dev/mean_score`, mode=max;
- no pseudo cache/loss/warm-up settings are present.

Record the resolved-config path for each run.

## 4. Freeze the research conclusion

Append the exact research interpretation to PROGRESS_EN.md.

Comparator:

- F2 DEV Mean: `0.774569`
- F2 D: `0.697035`
- F2 P: `0.852104`

Corrected R3-B Mean deltas:

- seed42: `+0.020247`
- seed43: `-0.006807`
- seed44: `-0.005885`

Three-seed mean:

- D `0.696178`
- P `0.857996`
- Mean `0.777087`

Three-seed mean deltas versus F2:

- D `-0.000857`
- P `+0.005892`
- Mean `+0.002518`

Required conclusion:

- R3-B is a valid three-seed ablation result.
- It is NOT promoted as an R-full component because PLAN Section 7 requires the effect to repeat across at least 3 seeds, and only seed42 improves DEV Mean over F2.
- Seeds43/44 also fall below the predeclared Parkinson floor `0.842104`.
- The positive three-seed average is retained descriptively, not treated as proof of a stable gain.
- No significance/final-model claim is made.
- Stage 5 remains active because R4 from PLAN has not been evaluated and the final Stage-5 composition is not frozen.

Do not use Test values in this conclusion.

## Exact verification

At minimum run:

    python3 - <<'PY'
    # Read-only audit:
    # - locate corrected B42/B43/B44 run dirs and selected checkpoints
    # - SHA256 full files
    # - load CPU payloads
    # - deterministic model_state_dict tensor digests
    # - parse resolved configs
    # - assert seeds/method settings
    # - assert pairwise-distinct checkpoint and tensor digests
    PY

Also:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion
    git diff origin/main -- src/common
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Note: historical R3 source/config changes already exist on this branch versus origin/main. For C2 itself, verify the new C2 commit changes only docs/PROGRESS_EN.md.

## Acceptance criteria

C2 passes only if:

- no training/new run occurs;
- all three corrected selected checkpoints are found;
- full-file SHA256 values are recorded and pairwise distinct;
- tensor-content digests are recorded and pairwise distinct;
- resolved configs prove true seeds 42/43/44 and the exact frozen R3-B settings;
- no Test metric affects the conclusion;
- PROGRESS_EN.md records the exact audit command/results;
- PROGRESS_EN.md records the non-promotion conclusion;
- C2 changes only docs/PROGRESS_EN.md after the manager task commit;
- git diff --check passes;
- branch is committed and pushed;
- main/master remains untouched.

Passing C2 closes the R3 bundle evidence contract. It does NOT close Stage 5 and does NOT authorize R4. The manager will decide the next task separately.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005e`;
- C2 evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- selected B42/B43/B44 checkpoint paths;
- full-file SHA256 values;
- tensor-content digests;
- resolved-config paths and exact seeds;
- confirmation pairwise-distinct identities;
- confirmation no training/new run/Test-driven decision;
- frozen promotion conclusion: R3-B not promoted;
- Stage 5 remains active;
- R4 not started;
- src/audio/src/video/fusion/common unchanged by C2.

Stop after C2.
