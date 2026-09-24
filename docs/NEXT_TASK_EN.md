# TASK-005A4: Add Fixed CLIP Semantic Evidence for the Blocked Depression RAMPS Acceptance Gate

## Role

You are the implementing Codex. Execute only this offline semantic/VLM RAMPS reliability task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-005a4

Do not train a student.
Do not modify src/audio or src/video.
Do not modify accepted models, checkpoints, caches, or the DataModule.
Do not lower precision_target below 0.90.
Do not lower min_support below 10.
Do not use Test rows or Test metrics.
Do not reselect or redesign the already-deployable Parkinson R2 rules.
Do not start RAMPS R3/R4.
Do not start the deferred general Text/Description Stage 3.

## Goal

TASK-005A2 and TASK-005A3 established that depression cannot obtain a stable reliable missing-head acceptance subset from strong-audio confidence alone or from strong-audio plus independent-video reliability.

Parkinson already has accepted R2 deployment rules and must remain frozen.

This task tests one fixed semantic/VLM bridge for the blocked depression head only.

Use:
- frozen strong temporal audio as the pseudo-target source;
- the already selected frozen video V2 only where the fixed semantic family explicitly needs three-way agreement;
- existing label-free cached CLIP image embeddings;
- the matching local CLIP text encoder with one predeclared prompt bank;
- deterministic 5-fold out-of-fold DEV selection;
- unchanged final reliability gate: empirical precision >= 0.90 and support >= 10.

If at least one depression side survives OOF selection and frozen full-DEV confirmation, combine that depression rule with the previously frozen Parkinson R2 rules and only then build a two-head TRAIN missing-target cache.

Pseudo-target values must remain calibrated strong-audio probabilities. Semantic/video signals are acceptance evidence only.

## Explicit Semantic-Embedding Authorization

This task is a manager-authorized semantic label-embedding acceptance ablation under PROJECT_REQUIREMENTS Section 11 and the Stage-5 semantic bridge. It does not start the deferred general Text/Description stage.

The existing CLIP image cache was extracted label-free. The fixed text prompts are global task-level semantic prototypes. No sample label, corpus identity, split, protocol, observed target, missing target, or model prediction may enter prompt construction.

No prompt may be changed after seeing metrics. Negative prompts such as "no depression", "healthy instead of depression", or "not depressed" are prohibited.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 5, 6, 8, 9, 10, 11, 12, 13, 14
3. docs/PLAN.md Stage 5
4. docs/PROGRESS_EN.md through MANAGER-DECISION-033 and MANAGER-CORRECTION-033A
5. docs/NEXT_TASK_EN.md
6. docs/SOTA_REVIEW_EN.md Sections 4.4, 4.5, 7, 8, 9, 10, 11, 13
7. src/fusion/loss/ramps_r1_teacher.py
8. src/fusion/loss/ramps_r2_reliability.py
9. src/fusion/models/frozen_audio_temporal_adapter.py
10. src/video/models/depart_v2.py
11. src/video/features/clip_video_features.py
12. src/fusion/data/wsm_av_fusion_datamodule.py
13. scripts/video/extract_clip_video_features.py

## Allowed Tracked Files

- src/fusion/loss/ramps_r2_semantic.py
- src/fusion/loss/__init__.py
- scripts/common/prepare_ramps_r2_semantic_targets.py
- docs/PROGRESS_EN.md

No other tracked file may be modified. No training config is authorized. No Chimera registration is required because this is an offline artifact-preparation/audit utility.

## Fixed Model/Caches

Strong-audio checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Historical DEV:
- depression Score = 0.7479183895
- Parkinson Score = 0.8277353635
- Mean_Score = 0.7878268765

Independent video checkpoint:

    logs/wsm_mm_pd_dep_v1/depart_v2_prototype_gated_2026-09-24_14-39_wsm_video_depart_v2_model_0a7294f4/checkpoints/epoch=5_dev_mean_score=0.7066.pt

Required SHA256:

    3e39778126db401a2fe17bb472a172191616e8a7b5265e982233c53dcd4af2f

Historical DEV:
- depression Score = 0.6201013364
- Parkinson Score = 0.7930427585
- Mean_Score = 0.7065720475

Strict-load epoch 5 exactly.

Accepted video-cache CLIP identity:

    openai/clip-vit-base-patch32
    revision main
    feature_dim 512

Use matching local transformers.CLIPModel and transformers.CLIPProcessor with local_files_only=true. Do not download a model or use another VLM.

The text model is inference-only, eval mode, requires_grad=false, torch.inference_mode, no optimizer.

## Fixed Prompt Bank

Exactly these prompts are authorized.

Depression prompts:
1. a video of a person with depression
2. a person with depressive symptoms
3. a person showing signs of depression

Neutral prompts:
1. a video of a person
2. a person
3. a video showing a person

No additional prompt, paraphrase, prompt search, negative prompt, or Parkinson semantic prompt search.

Record the exact ordered prompt bank and its canonical JSON SHA256.

## Frozen Parkinson R2 Deployment Contract

Do not search Parkinson semantic rules.

Reproduce and freeze TASK-005A3 R2 Family-A rules.

Positive:
    tau_conf = 0.77

Negative:
    tau_conf = 0.50

Family A definition:
- audio/video predict the same side;
- joint_confidence = min(audio_class_confidence, video_class_confidence);
- accept when joint_confidence >= tau_conf.

Re-fit only the required full-DEV audio/video temperatures on complete observed Parkinson DEV rows, then apply the fixed tau values.

Require exact support reproduction and precision tolerance 1e-6:

Positive:
    support = 21
    precision = 1.0

Negative:
    support = 193
    precision = 0.9585492

If these frozen rules do not reproduce, stop blocked. Do not reselect Parkinson.

## Fixed Data

Use WSMAVFusionDataModule.

Roots:
    data_root = /media/maxim/Databases/WSM_NEW
    audio_feature_cache_root = /media/maxim/Databases/WSM_NEW/features
    video_cache_root = /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Expected:
    train = 6325
    dev = 933
    test_none = 1364
    test_soft = 1208
    test_hard = 1014
    joined_total = 8622
    missing_audio = 0
    missing_video = 0

Only train_dataset and val_dataset may be iterated.

Do not call val_dataloader().
Do not iterate test_none, test_soft, or test_hard.

## Fixed Output Root

Write runtime artifacts outside git:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1

Always write:
- semantic_prompt_bank.json
- semantic_search.json
- audit.json

Write only if the final two-head gate passes:
- train_missing_targets.pt

Refuse overwrite of a non-empty output root unless explicit --overwrite is supplied.

## 1. Reusable Semantic Reliability Utilities

Implement src/fusion/loss/ramps_r2_semantic.py.

Required deterministic utilities:
- normalize cached CLIP frame embeddings;
- masked pooled CLIP image embedding;
- normalize/average prompt embeddings;
- semantic cosine-margin computation;
- positive-monotonic scalar semantic calibration;
- binary calibration metrics;
- deterministic depression 5-fold records;
- side-conditional OOD percentile with no label leakage;
- exactly three semantic reliability-family search/evaluation functions;
- rule application;
- detached reliability calculation.

No new dependency. Reuse R1/R2 utilities where safe.

## 2. CLIP Image Semantic Representation

Input cached video features:
    video [B,T,512]
    video_mask [B,T]

For each valid frame:
    frame_norm = L2_normalize(frame_feature)

Then:
    pooled = sum(frame_norm * mask) / valid_count
    image_embedding = L2_normalize(pooled)

Do not pass cached image embeddings back through the CLIP vision tower. Do not read/re-extract raw video frames.

## 3. CLIP Text Semantic Representation

Using the fixed prompt bank and matching local CLIP text encoder:

1. tokenize all prompts exactly;
2. compute get_text_features;
3. L2-normalize each prompt embedding;
4. average normalized embeddings inside each bank;
5. L2-normalize the bank mean.

Produce:
    e_depression
    e_neutral

Record:
- model identity/revision;
- exact prompts;
- prompt-bank JSON SHA256;
- e_depression tensor SHA256;
- e_neutral tensor SHA256.

Use the same embeddings for every sample and fold.

## 4. Fixed Semantic Margin

For each sample:

    s_dep = cosine(image_embedding, e_depression)
    s_neutral = cosine(image_embedding, e_neutral)
    semantic_margin = s_dep - s_neutral

No alternative score and no prompt weighting.

## 5. Positive-Monotonic Semantic Calibration

Fit observed depression labels only.

Model:

    semantic_logit = scale * semantic_margin + bias
    scale = exp(log_scale) > 0
    p_semantic = sigmoid(semantic_logit)

Fit log_scale and bias by deterministic full-batch BCE with torch LBFGS.

Deployed bounds:
    0.01 <= scale <= 100.0
    -20.0 <= bias <= 20.0

If unconstrained values leave bounds, clamp deployed values and recompute metrics. Record unconstrained/deployed values.

The positive scale is mandatory: calibration may shift/rescale the fixed label margin but may not flip semantic direction.

Record NLL, Brier, and ECE-15 for calibrated semantic probabilities.

## 6. Deterministic Depression Five-Fold OOF Contract

Use only observed depression DEV rows.

Exactly five stratified folds.

Assignment:
1. split rows by binary depression label;
2. inside each class sort by sha256("depression-semantic:" + segment_id);
3. round-robin folds 0..4.

Each held-out row appears exactly once.

For each fold, using the other four folds only, fit:
- strong-audio temperature;
- video temperature;
- semantic monotonic scale/bias;
- normalized audio task-feature centroids for depression class 0/1;
- class-specific audio feature distance reference distributions.

No held-out row may fit its own parameter/statistic. Both classes must occur in each fitting partition.

## 7. Correct Side-Conditional OOD Rule

MANAGER-CORRECTION-033A is mandatory.

When evaluating candidate side c:
- compute row distance to centroid c;
- compare it with the fitting-partition distance distribution for true class c.

Do not use the held-out row's true class to choose its centroid.

For observed held-out rows, true label may be used only afterward to calculate empirical precision/correctness.

For missing TRAIN rows, candidate side alone chooses centroid/reference.

## 8. OOF Evidence Per Depression Row

For every held-out observed depression row compute:
    p_audio
    p_video
    p_semantic
    audio_entropy
    video_entropy
    semantic_entropy

For candidate side c compute:
    audio_conf_c
    video_conf_c
    semantic_conf_c
    ood_percentile_c

Agreement uses predicted side at 0.5.

## 9. Exactly Three Depression Semantic Reliability Families

Final precision_target remains 0.90.
Final min_support remains 10.

### Family S1 — Audio + Semantic Agreement

Require audio and semantic same-side prediction.

    joint_conf = min(audio_conf_c, semantic_conf_c)
    joint_conf >= tau_conf

tau_conf grid:
    0.50, 0.51, ..., 0.99

### Family S2 — Audio + Semantic + Uncertainty + Correct OOD

Require S1 plus:

    uncertainty = max(audio_entropy, semantic_entropy)
    uncertainty <= tau_entropy
    ood_percentile_c <= tau_ood

tau_entropy:
    0.25
    0.50
    0.75

tau_ood:
    0.80
    0.90
    0.95

### Family S3 — Three-Way Audio + Video + Semantic Agreement

Require audio, video, and semantic same-side prediction.

    triple_conf = min(audio_conf_c, video_conf_c, semantic_conf_c)
    uncertainty = max(audio_entropy, video_entropy, semantic_entropy)

Accept when:
    triple_conf >= tau_conf
    uncertainty <= tau_entropy

tau_conf:
    0.50, 0.51, ..., 0.99

tau_entropy:
    0.25
    0.50
    0.75

No fourth family, prompt change, or corpus-aware rule.

## 10. Depression OOF Rule Selection

Search positive and negative sides separately.

For every configuration record:
- family;
- parameters;
- support;
- correct;
- precision;
- class coverage.

Eligible iff:
    support >= 10
    precision >= 0.90

Select deterministically:
1. maximum support;
2. maximum precision;
3. lower complexity S1 before S2 before S3;
4. lower tau_conf;
5. higher tau_entropy where applicable;
6. higher tau_ood where applicable.

If no eligible rule for a side, disable it.

Depression is OOF-recoverable if at least one side is enabled.

## 11. Frozen Full-DEV Confirmation

After OOF selection, fit deployment statistics using all observed depression DEV rows:
- audio temperature;
- video temperature;
- semantic positive-monotonic scale/bias;
- class 0/1 audio centroids;
- class-specific distance reference distributions.

Apply the exact OOF-selected rules only. Do not search again.

Each selected side must again satisfy:
    support >= 10
    precision >= 0.90

A failing side is disabled.

Depression deployable iff at least one side survives.

If depression remains undeployable:
- write semantic_prompt_bank.json;
- write semantic_search.json;
- write audit.json;
- do not infer TRAIN;
- do not create train_missing_targets.pt;
- exit with clear RuntimeError.

## 12. Final Two-Head Deployment Rule

Only if depression becomes deployable.

Depression:
    use confirmed semantic rule(s) from TASK-005A4.

Parkinson:
    use only frozen TASK-005A3 Family-A rules:
      positive tau_conf=0.77
      negative tau_conf=0.50

Do not semantically reselect Parkinson.

## 13. TRAIN Target Construction

Only after complete two-head DEV gate passes.

Infer all 6325 canonical TRAIN rows once.

For missing depression:
- compute audio/video probabilities;
- compute semantic margin/probability;
- apply confirmed semantic rule;
- for S2 use candidate-side OOD centroid/reference.

For missing Parkinson:
- use frozen R2 Family-A audio/video agreement rules.

For every accepted entry:
    pseudo_target = calibrated strong-audio probability

Do not use semantic probability as target.
Do not use video probability as target.
Do not average teachers.
Do not harden to 0/1.

Reliability:

Depression S1:
    min(audio_conf, semantic_conf)

Depression S2:
    min(audio_conf, semantic_conf) * (1 - uncertainty) * (1 - ood_percentile)

Depression S3:
    min(audio_conf, video_conf, semantic_conf) * (1 - uncertainty)

Parkinson frozen Family A:
    min(audio_conf, video_conf)

Clamp numerically to [0,1]. Detach all cache values.

Observed entries:
    pseudo_accept_mask = false
    pseudo_target = NaN
    pseudo_reliability = 0
    pseudo_class = -1

Observed truth always wins.

## 14. train_missing_targets.pt Contract

Only if passed.

Version:
    ramps-r2-semantic-v1

Required fields:
- task_names
- segment_ids
- observed_mask
- observed_targets
- raw_audio_logits
- calibrated_audio_probs
- raw_video_logits
- calibrated_video_probs
- semantic_margin_depression
- calibrated_semantic_prob_depression
- audio_task_features
- pseudo_accept_mask
- pseudo_targets
- pseudo_reliability
- pseudo_class
- depression_selected_rules
- parkinson_frozen_rules
- audio_temperatures
- video_temperatures
- semantic_calibrator
- semantic_prompt_bank
- semantic_prompt_bank_sha256
- clip_model_name
- clip_model_revision
- audio_checkpoint_path
- audio_checkpoint_sha256
- video_checkpoint_path
- video_checkpoint_sha256

Requirements:
- rows=6325, tasks=2;
- unique canonical IDs;
- observed tensors unchanged;
- no pseudo on observed;
- accepted targets finite in [0,1];
- every accepted target equals calibrated AUDIO probability exactly;
- reliability finite in [0,1];
- reliability zero on observed/rejected entries;
- pseudo_class in {-1,0,1};
- accepted depression count > 0;
- accepted Parkinson count > 0.

## 15. Runtime JSON Contracts

semantic_prompt_bank.json always records:
- version;
- CLIP model/revision;
- exact ordered prompt banks;
- canonical JSON SHA256;
- depression embedding SHA256;
- neutral embedding SHA256;
- no sample-specific information statement;
- no negative-prompt statement.

semantic_search.json always records:
- version/timestamp;
- audio/video checkpoint paths + SHA256;
- CLIP model/revision;
- prompt bank + hashes;
- canonical counts;
- no-Test statement;
- historical audio/video DEV reproduction;
- frozen Parkinson R2 reproduction;
- depression fold audit;
- per-fold audio/video temperatures;
- per-fold semantic scale/bias;
- per-fold semantic calibration metrics;
- full-DEV semantic scale/bias and metrics;
- exact S1/S2/S3 definitions;
- complete OOF candidate table;
- selected OOF rules;
- full-DEV confirmation;
- depression deployable;
- overall two-head deployable.

audit.json always records blocked/pass state and no correctness/comorbidity claim. If passed also record per-task missing/accepted/rejected/coverage, accepted pos/neg, target/reliability statistics, and overwrite invariants.

## 16. Production Script

Implement:

    scripts/common/prepare_ramps_r2_semantic_targets.py

Required sequence:

1. deterministic seeds;
2. verify audio SHA;
3. verify video SHA and strict epoch-5 load;
4. build DataModule and canonical counts;
5. build/freeze/eval audio/video teachers;
6. load matching local CLIP text encoder;
7. encode fixed prompt bank once;
8. infer canonical DEV only: audio logits/features, video logits, cached CLIP semantic representation/margin;
9. reproduce historical audio/video DEV metrics;
10. reproduce frozen Parkinson R2 rules exactly;
11. construct depression 5-fold OOF semantic records;
12. search exactly S1/S2/S3;
13. freeze OOF selected depression rules;
14. full-DEV confirmation with no redesign;
15. write semantic_prompt_bank.json + semantic_search.json;
16. if depression blocked: write audit.json, do not infer TRAIN, do not write cache, exit RuntimeError;
17. if depression passes: infer canonical TRAIN only, build two-head cache using semantic depression + frozen Parkinson, validate, write cache + audit.

Do not call dm.val_dataloader().
Do not infer Test.
Do not re-extract raw frames.

## Exact Production Command

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/common/prepare_ramps_r2_semantic_targets.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features \
      --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --audio-checkpoint logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt \
      --video-checkpoint logs/wsm_mm_pd_dep_v1/depart_v2_prototype_gated_2026-09-24_14-39_wsm_video_depart_v2_model_0a7294f4/checkpoints/epoch=5_dev_mean_score=0.7066.pt \
      --clip-model openai/clip-vit-base-patch32 \
      --clip-revision main \
      --output-root /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1 \
      --precision-target 0.90 \
      --min-support 10 \
      --folds 5 \
      --batch-size 32 \
      --num-workers 4 \
      --device cuda

If CUDA unavailable, stop blocked.
If local CLIP text model unavailable, stop blocked.
Do not download it.

## Required Verification

Compile:
    python3 -m py_compile \
      src/fusion/loss/ramps_r2_semantic.py \
      src/fusion/loss/__init__.py \
      scripts/common/prepare_ramps_r2_semantic_targets.py

Synthetic checks:
- fixed prompt-bank hash determinism;
- semantic embedding normalization;
- masked image pooling;
- finite semantic margin;
- positive semantic scale > 0;
- finite NLL/Brier/ECE;
- deterministic folds;
- no held-out calibration leakage;
- side-conditional OOD uses candidate side, not target;
- S1/S2/S3 search/tie-breaking;
- disabled side;
- frozen Parkinson rule reproduction helper;
- observed entry cannot become pseudo;
- accepted target equals audio probability;
- reliability in [0,1].

Then run the exact production command.

Afterward verify:
- audio/video DEV reproduction;
- prompt-bank hashes;
- no teacher/text-encoder gradients;
- no optimizer;
- no Test;
- no raw-frame extraction;
- if blocked, cache absent;
- if passed, complete TRAIN cache invariants.

Finally:
    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/models
    git diff -- src/fusion/data
    git status --short

Tracked diff must contain only the four allowed paths.

## Acceptance Criteria

Pass only if:
- exact audio/video checkpoints verify;
- historical DEV reproduction passes;
- matching local CLIP text encoder loads;
- fixed prompt bank remains exact;
- no negative/sample-specific prompts;
- no raw-frame extraction;
- Parkinson frozen R2 rules reproduce exactly and are not reselected;
- depression OOF uses exactly five deterministic folds;
- semantic calibrator is positive-monotonic;
- OOD implementation has no target leakage;
- exactly S1/S2/S3 are searched;
- precision_target remains 0.90;
- min_support remains 10;
- no Test;
- depression keeps at least one enabled side after OOF + frozen full-DEV confirmation;
- only then TRAIN inference runs;
- depression and Parkinson both have accepted TRAIN missing targets;
- accepted target is always calibrated audio probability;
- observed truth is never overwritten;
- no missing-label correctness claim;
- no student training;
- no R3/R4;
- no general Stage-3 text/description work;
- src/audio unchanged;
- src/video unchanged;
- models/DataModule unchanged;
- git diff --check passes;
- branch codex/task-005a4 committed and pushed;
- main/master untouched.

A blocked result is valid evidence if semantic depression still fails and:
- semantic_prompt_bank.json, semantic_search.json, audit.json exist;
- no train_missing_targets.pt exists;
- no prompt/rule/reliability relaxation;
- no Test/student training.

## Required PROGRESS_EN Update

Append TASK-005A4 evidence with:
- branch;
- implementation/evidence SHA;
- push result;
- audio/video checkpoint verification;
- historical DEV reproduction;
- CLIP model/revision/local-only load;
- exact prompt bank + SHA;
- text embedding hashes;
- frozen Parkinson rule reproduction;
- depression fold audit;
- semantic per-fold/full-DEV scale+bias;
- semantic calibration metrics;
- selected depression OOF rules;
- OOF support/precision/coverage;
- full-DEV confirmation;
- depression deployable state;
- overall two-head gate;
- artifact paths;
- if passed, per-task TRAIN accepted/rejected/coverage, accepted pos/neg, reliability/target stats, overwrite audits;
- if blocked, explicit no cache publication;
- explicit 0.90/min_support=10 unchanged;
- explicit no Test;
- explicit no raw-frame extraction;
- explicit no Candidate B/fusion teacher;
- explicit no student training;
- explicit no missing-label correctness/comorbidity claim;
- R3/R4 not started;
- general text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 5 status;
- recommended next atomic task only.

If TASK-005A4 passes:
    recommended next = TASK-005B wire the accepted semantic+R2 two-head offline cache into an observed+pseudo sparse loss/data contract and prove direct missing-head gradients without full training.

If TASK-005A4 remains blocked:
    recommended next = manager review of whether to pause missing-label recovery until the deferred text/description stage or stop the RAMPS pseudo-target path; do not relax reliability criteria automatically.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:
- branch codex/task-005a4;
- implementation/evidence SHA;
- pushed-to-origin status;
- main/master untouched;
- audio/video checkpoint verification;
- CLIP model/revision;
- prompt-bank SHA;
- frozen Parkinson rule reproduction;
- depression selected OOF rules;
- OOF/full-DEV precision/support;
- depression deployable state;
- whether two-head TRAIN cache was published;
- if published, per-task accepted/rejected/coverage and pos/neg;
- artifact paths;
- unchanged precision_target=0.90/min_support=10;
- no Test use;
- no raw-frame extraction;
- no Candidate B/fusion teacher;
- no student training;
- no missing-label correctness claim;
- R3/R4 not started;
- general text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
