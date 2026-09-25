# TASK-005E-BUNDLE-C1: Restore the Frozen R3 Architecture and Run True Seeds

## Authority and branch

This is one manager-authorized corrective bounded bundle.

Required branch:

    codex/task-005e

Reuse of this existing branch is explicitly authorized. Preserve all prior manager/Codex commits and exploratory run evidence. Do not reset, rebase away, force-push, or modify main/master.

The prior four runs remain preserved as exploratory/debug evidence only because:
- the implemented model deviated from the frozen R3 architecture;
- every nominal seed-43/44 YAML actually contained `seed: 42`.

Do not delete or rewrite that history.

## Goal

Correct the R3 implementation to the exact frozen TASK-005E architecture, correct the continuation configs to true seeds 43/44, freeze the corrected bundle before training, then rerun the same bounded DEV-only experiment tree.

Maximum NEW production training invocations in C1: 4.

No post-hoc tuning.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, Stage 5
5. docs/PROGRESS_EN.md through MANAGER-REVIEW-042
6. docs/NEXT_TASK_EN.md
7. src/fusion/models/av_r3_disease_query.py
8. src/fusion/loss/r3_aux_agreement_loss.py
9. src/fusion/data/wsm_av_fusion_datamodule.py
10. src/common/loss/wsm_masked_sparse_loss.py
11. src/chimera_plugin.py
12. configs/wsm_mm_pd_dep_v1/fusion/06_ramps_r3_disease_query_contract.yaml
13. configs/wsm_mm_pd_dep_v1/fusion/07_r3_a_sparse_seed42.yaml
14. configs/wsm_mm_pd_dep_v1/fusion/08_r3_b_agreement_seed42.yaml
15. configs/wsm_mm_pd_dep_v1/fusion/09_r3_a_sparse_seed43.yaml
16. configs/wsm_mm_pd_dep_v1/fusion/10_r3_a_sparse_seed44.yaml
17. configs/wsm_mm_pd_dep_v1/fusion/11_r3_b_agreement_seed43.yaml
18. configs/wsm_mm_pd_dep_v1/fusion/12_r3_b_agreement_seed44.yaml

## Allowed tracked files

Codex may modify only:

- src/fusion/models/av_r3_disease_query.py
- configs/wsm_mm_pd_dep_v1/fusion/06_ramps_r3_disease_query_contract.yaml
- configs/wsm_mm_pd_dep_v1/fusion/07_r3_a_sparse_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/08_r3_b_agreement_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/09_r3_a_sparse_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/10_r3_a_sparse_seed44.yaml
- configs/wsm_mm_pd_dep_v1/fusion/11_r3_b_agreement_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/12_r3_b_agreement_seed44.yaml
- docs/PROGRESS_EN.md

Do not modify:
- src/fusion/loss/r3_aux_agreement_loss.py;
- src/chimera_plugin.py;
- any data module;
- any existing model;
- src/audio;
- src/video;
- src/common;
- PROJECT_REQUIREMENTS/PLAN.

If the already-frozen auxiliary loss itself creates a new exact runtime blocker after the architecture correction, STOP and report it rather than broadening scope.

## 1. Exact frozen R3 architecture

Implement exactly this constructor:

    audio_feature_dim: int = 768
    video_feature_dim: int = 512
    hidden_dim: int = 192
    gate_hidden_dim: int = 192
    dropout: float = 0.2
    num_tasks: int = 2

Validate:
- all dimensions > 0;
- num_tasks == 2;
- 0 <= dropout < 1.

### Inputs

Consume only:
- batch.inputs["audio_cls"] [B,768];
- batch.inputs["video"] [B,T,512];
- video_mask [B,T];
- modality_available [B,2].

Require finite floating audio/video inputs, correct shapes, at least one available modality, and at least one valid video position whenever video is available.

Do not consume corpus identity, observed labels/masks, task IDs, pseudo fields, teacher scores, text, or description as model input.

### Projections

Audio:

    LayerNorm(audio_feature_dim)
    Linear(audio_feature_dim, hidden_dim)
    GELU
    Dropout(dropout)

Video:

    LayerNorm(video_feature_dim)
    Linear(video_feature_dim, hidden_dim)
    GELU
    Dropout(dropout)

Pool video by the F2 masked mean before projection.

Zero unavailable raw input before projection AND zero unavailable projected contribution after projection so projection bias cannot leak unavailable modality information.

### Task queries

Use exactly:

    task_queries: nn.Parameter shape [2, hidden_dim]

Initialize normal mean 0, std 0.02.

### Task-conditioned candidates

Keep exactly two candidate norms total:

    task_candidate_norms: ModuleList[LayerNorm(hidden_dim), LayerNorm(hidden_dim)]

For task t, the SAME norm is used for audio and video:

    q = task_queries[t]
    a_t = task_candidate_norms[t](effective_audio_features + q)
    v_t = task_candidate_norms[t](effective_video_features + q)

No separate audio/video candidate norms.

### Shared query-conditioned gate

Use exactly one shared network:

    LayerNorm(2 * hidden_dim)
    Linear(2 * hidden_dim, gate_hidden_dim)
    GELU
    Dropout(dropout)
    Linear(gate_hidden_dim, 1)

Score exactly:

    [q || a_t]
    [q || v_t]

Mask unavailable modality logits before softmax.

Require:
- finite weights;
- [0,1];
- sum to 1 within tolerance;
- unavailable weight exactly 0;
- audio-only = [1,0];
- video-only = [0,1].

### Task fusion

For each task:

    fused_t = task_fusion_norms[t](
        q + w_audio * a_t + w_video * v_t
    )

Use exactly two task_fusion_norms.

### Main heads

Exactly two independent heads:

    LayerNorm(hidden_dim)
    Linear(hidden_dim, hidden_dim)
    GELU
    Dropout(dropout)
    Linear(hidden_dim, 1)

Return preds [B,2].

### Auxiliary heads

For each task and modality use exactly:

    LayerNorm(hidden_dim)
    Linear(hidden_dim, 1)

Compute audio aux from a_t and video aux from v_t.

Keep logits finite even when modality invalid; validity masks control downstream use.

### Required aux keys

Expose exactly/at least:

- features_audio [B,H]
- features_video [B,H]
- task_audio_features [B,2,H]
- task_video_features [B,2,H]
- task_modality_weights [B,2,2]
- task_features [B,2,H]
- audio_aux_logits [B,2]
- video_aux_logits [B,2]
- audio_aux_valid bool [B,2]
- video_aux_valid bool [B,2]
- task_logits mapping with depression and parkinson

All floating outputs finite.

### Exact parameter count

For the exact frozen dimensions above, require:

    R3 trainable parameters = 403079

Frozen F2:

    736004

Delta:

    -332925

If the implementation does not produce exactly 403079 trainable parameters, STOP before training and report the mismatch.

## 2. Correct all config seeds and exact model params

Every R3 config must include:

    gate_hidden_dim: 192

Required top-level seeds:

- 06 contract: 42
- 07 R3-A seed42: 42
- 08 R3-B seed42: 42
- 09 R3-A seed43: 43
- 10 R3-A seed44: 44
- 11 R3-B seed43: 43
- 12 R3-B seed44: 44

Use unique corrected run names ending in `_c1` so the new runs cannot be confused with the invalid exploratory runs.

Examples:

    r3_a_sparse_seed42_c1
    r3_b_agreement_seed42_c1
    r3_b_agreement_seed43_c1

Preserve all other frozen optimizer/data/instrumentation/loss settings.

R3-B remains exactly:

    aux_weight: 0.25
    agreement_weight: 0.10

No pseudo path.

## 3. Mandatory corrected firewall commit before training

Before ANY new production training invocation:

1. finish the exact model correction;
2. correct all seven YAMLs;
3. validate all seven YAMLs;
4. run every contract smoke below;
5. update PROGRESS_EN.md with the corrected frozen contract;
6. commit and push one corrective firewall commit.

No production training may begin before that corrective firewall commit exists on origin.

## 4. Seed identity firewall

Run a deterministic inline audit BEFORE training.

It must parse the YAMLs and assert exact seeds above.

Then for seeds 42/43/44:

- call Chimera `define_seed(seed)`;
- instantiate the exact R3 model;
- hash the complete initial trainable state deterministically;
- assert hashes for 42/43/44 are pairwise distinct;
- instantiate seed 42 twice after resetting with `define_seed(42)` and assert the two seed-42 hashes are identical.

Also demonstrate that the first TRAIN shuffle order differs for at least one of 42/43/44 OR, if direct DataLoader-order capture is awkward, generate and record a deterministic `torch.randperm(6325)` prefix immediately after `define_seed(seed)` for each seed and assert pairwise distinct prefixes.

Record all hashes/prefixes in PROGRESS_EN.md.

This is a seed firewall only; do not use DEV/Test rows.

## 5. Contract smokes

Repeat the exact synthetic availability/query smoke:

- both / audio-only / video-only samples;
- exact modality weights;
- unavailable-input invariance <= 1e-7;
- all-unavailable raises;
- required aux shapes/keys and finite values.

Repeat real TRAIN-only forward/loss/backward with ordinary sparse loss:

- both main heads non-zero finite gradients;
- both projections non-zero finite gradients;
- task_queries non-zero finite gradient;
- shared gate non-zero finite gradients;
- auxiliary head gradients absent under sparse main-only loss.

Repeat R3-B synthetic auxiliary-loss smoke:

- main and aux heads receive gradients;
- unknown labels masked;
- agreement finite;
- zero-valid cases finite;
- no optimizer step.

Do not iterate DEV/Test in firewall smokes.

## 6. Corrected bounded production rerun

The previous four runs are INVALID for gate decisions after this architecture correction. Do not combine them statistically with C1.

Use the same frozen comparator and screen:

F2:
- D 0.697035
- P 0.852104
- Mean 0.774569

Seed-42 corrected variant passes only if:
- Mean > 0.774569;
- D >= 0.687035;
- P >= 0.842104.

Run after firewall:

1. corrected R3-A seed 42;
2. corrected R3-B seed 42;
3. if neither passes: stop;
4. if exactly one passes: run that corrected variant at true seeds 43 and 44;
5. if both pass: choose by higher DEV Mean; tie <=1e-6 by higher minimum task delta vs F2; final tie choose A; then run chosen variant at true seeds 43 and 44.

Maximum new production invocations: 4.

All selection/branching is DEV-only.

TEST_NONE/SOFT/HARD remain monitoring-only.

No post-hoc method change.

## 7. Post-run seed audit

For every executed corrected run record:

- config path;
- top-level seed;
- run directory;
- MLflow run ID/status;
- selected DEV epoch/checkpoint;
- selected DEV D/P/Mean;
- F2 deltas;
- frozen screen status;
- same-epoch Test monitoring only after DEV freeze.

For any continuation variant seeds 42/43/44 additionally record:

- SHA256 of selected checkpoint file for each seed;
- assert the three selected checkpoint SHA256 values are pairwise distinct;
- DEV D/P/Mean mean and sample std across the TRUE three seeds.

If two true seeds happen to yield identical rounded metrics, distinct selected-checkpoint hashes are required before interpreting that as legitimate zero/near-zero metric variance.

## 8. Forbidden

- no modification of auxiliary loss weights/formula;
- no new architecture variant;
- no R2 pseudo path;
- no text/description;
- no R4;
- no Stage 6/7;
- no Final Test;
- no second attempt with altered settings after seeing DEV;
- no Test-driven branching;
- no missing-label/comorbidity/significance/final-model claim.

## Verification

Run at minimum:

    python3 -m py_compile       src/fusion/models/av_r3_disease_query.py       src/fusion/loss/r3_aux_agreement_loss.py       src/chimera_plugin.py

Validate all seven YAMLs.

Run the parameter count assertion:

    R3 == 403079
    F2 == 736004

Run seed identity firewall, synthetic model smoke, real TRAIN-only sparse smoke, and auxiliary-loss smoke.

After all runs:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/common
    git status --short
    git diff --stat origin/main...HEAD
    git log -8 --oneline --decorate

## Acceptance

C1 passes only if:

- exact frozen R3 architecture is implemented;
- exact R3 count is 403079;
- all seven configs include gate_hidden_dim 192;
- seed43/44 YAMLs contain true 43/44 values;
- seed initialization hashes prove distinct initial states;
- corrected firewall commit exists before training;
- all contract smokes pass;
- corrected A42/B42 rerun under frozen rules;
- continuation, if any, uses true seeds 43/44;
- continuation selected checkpoints have distinct SHA256 values;
- Test never affects selection/branching;
- no post-hoc tuning occurs;
- scope remains exact;
- PROGRESS preserves both invalid exploratory evidence and corrected evidence;
- branch is pushed;
- main/master untouched.

Passing C1 allows manager acceptance of the R3 bundle and consideration of the next PLAN step. Do not start it.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly distinguish:
- old exploratory invalid bundle;
- corrected C1 bundle.

Include:
- corrective firewall commit SHA;
- final evidence commit SHA;
- exact R3 parameter count;
- exact config seeds;
- initial-state hashes;
- number of NEW production invocations;
- corrected A42/B42 results;
- continuation rule;
- true seed43/44 results if executed;
- selected checkpoint SHA256 values;
- true three-seed mean/std if available;
- no Test-driven decisions;
- no post-hoc tuning;
- src/audio/src/video unchanged;
- R2 remains negative/unpromoted;
- no R4/text/Stage6/7/Final Test;
- Stage 5 remains active.

Stop after C1.
