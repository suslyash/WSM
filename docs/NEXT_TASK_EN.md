# TASK-003B: Freeze and Implement the T1 Video-Level Transcript Pipeline

## Authority and branch

This task follows **MANAGER-DECISION-074**.

Required branch:

    codex/task-003b

Start from current manager-updated `origin/main`.

This task implements and freezes T1, but **does not run production training**.

## Frozen scientific/data contract

TASK-003A established:

    SEGMENT TEXT ALIGNMENT NOT ESTABLISHED
    T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT
    T1 TRANSCRIPT DATA CONTRACT READY

Authoritative audit report:

    /media/maxim/Programs/Features/WSM/stage3_text_audit/transcript_source_audit_v1.json

SHA256:

    4254d81281522015d0633d6534e1616e925a2bf0130f67a37b541a2aacc2ed78

Canonical facts:

- 8,622 segments;
- 755 unique `(corpus, split, video_id)` groups;
- 754 plain-text files;
- one missing transcript: TRAIN availability gap;
- zero decode errors;
- no cross-split exact transcript-content hashes;
- three within-TRAIN duplicate hashes;
- no explicit segment timing metadata;
- zero parseable transcript timestamps.

Do not invent segment text.

## Frozen T1 family

There is exactly one T1 encoder/model family. No encoder search is allowed.

Frozen pretrained encoder:

    FacebookAI/xlm-roberta-base

Frozen revision:

    e73636d

Expected model.safetensors SHA256:

    6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb

Use `AutoTokenizer(..., use_fast=True)` and `AutoModel`.

The encoder is **fully frozen** and used only for offline feature extraction.

Do not fine-tune it.

Do not compare it against another text encoder.

## Frozen chunking and feature extraction

For every present video-level transcript:

1. decode UTF-8/UTF-8-SIG using the accepted TASK-003A contract;
2. tokenize the complete transcript with no label/task/diagnosis prompt or metadata injected;
3. do not truncate the document;
4. split token IDs into consecutive, non-overlapping chunks of exactly at most 510 content tokens;
5. add the encoder's normal single-sequence special tokens so every model input is at most 512 tokens;
6. stride/overlap = 0;
7. encoder in eval mode, no gradients;
8. for every chunk, take final hidden states;
9. mean-pool only valid **non-special** token positions;
10. save one float32 768-d vector per chunk, preserving chunk order.

Empty decoded text must be treated as unavailable, not as a fabricated zero feature.

The one missing transcript remains unavailable.

No labels, diagnosis, task ID, pseudo label, or corpus-specific prompt may enter feature extraction.

Corpus/split/video identity may appear only in cache indexing/audit metadata.

## External T1 cache

Build exactly one external cache root:

    /media/maxim/Programs/Features/WSM/text_t1_xlmr_v1

Required index:

    /media/maxim/Programs/Features/WSM/text_t1_xlmr_v1/cache_index.json

Feature artifact layout:

    features/<corpus>/<split>/<video_id>.pt

The cache builder MUST fail rather than overwrite an existing cache root/index.

Every index entry must include at least:

- corpus;
- split;
- video_id;
- transcript source path;
- transcript SHA256;
- availability;
- chunk count;
- feature dimension;
- artifact path/SHA256 when available;
- encoder name;
- requested revision;
- resolved Hugging Face commit SHA;
- model weight SHA256;
- tokenizer identity/fingerprints;
- max model length;
- content tokens per chunk;
- overlap/stride;
- pooling rule;
- feature dtype;
- canonical manifest fingerprint;
- TASK-003A report SHA.

Global cache metadata must include package versions and extractor source version/fingerprint.

Record cache-index SHA256 in PROGRESS.

Do not store raw transcript text or token IDs in the committed repo or cache index.

Feature artifacts may contain only numeric chunk features and reproducibility metadata, not raw text/token IDs.

## Allowed tracked files

Only:

- `src/chimera_plugin.py`
- `src/text/__init__.py`
- `src/text/features/__init__.py`
- `src/text/features/xlmr_video_transcript.py`
- `src/text/data/__init__.py`
- `src/text/data/wsm_text_video_datamodule.py`
- `src/text/models/__init__.py`
- `src/text/models/t1_chunk_transformer.py`
- `scripts/text/build_t1_xlmr_cache.py`
- `configs/wsm_mm_pd_dep_v1/text/00_t1_xlmr_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/text/01_t1_xlmr_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/text/02_t1_xlmr_seed44.yaml`
- `docs/PROGRESS_EN.md`

No `src/audio/*` changes.
No `src/video/*`, `src/fusion/*`, description, or common-source changes.
No dependency changes.

The three YAMLs are seed clones of **one** T1 family and do not count as three architecture families.

## T1 DataModule contract

Register:

    wsm_text_t1_datamodule

Use the canonical manifest plus the existing canonical segment/protocol metadata.

### TRAIN

TRAIN dataset unit must be exactly one available `(corpus, video_id)` video-level transcript.

Do not create one training row per segment.

Expected available TRAIN units from TASK-003A:

- depression: 287;
- Parkinson: 266;
- total: 553.

For each grouped video:

- assert all canonical segments agree on the observed disease label;
- target is two independent binary slots with NaN for the unknown task;
- observed mask is `[true,false]` for depression corpus and `[false,true]` for Parkinson corpus;
- load one cached ordered chunk-feature sequence.

The missing depression TRAIN transcript is excluded from text-only T1 training and counted explicitly as text unavailable.

If identical text exists in both corpus namespaces, keep the two canonical corpus/video training identities separate; do not merge them into synthetic dual annotation.

### DEV / TEST evaluation

DEV and TEST protocol datasets remain **canonical segment rows**.

Each segment row loads its parent video's one cached transcript feature sequence.

This deliberate evaluation broadcast is only to reproduce the project's primary segment-level metric protocol.

It MUST NOT be used for TRAIN.

Expose separately named evaluation streams so every epoch can report:

- `dev`;
- `test_none`;
- `test_soft`;
- `test_hard`.

Test remains monitoring-only and may not drive selection.

No Test lexical/manual inspection is authorized.

### Batch contract

Collate to:

- `inputs["text"]`: float tensor `[B,C,768]`;
- `masks["text_mask"]`: bool `[B,C]`;
- targets: float `[B,2]` with NaN unknowns;
- `masks["observed_mask"]`: bool `[B,2]`;
- metadata with canonical IDs/protocol identity.

Every sample must have at least one valid chunk.

Context must publish text feature dim, task names/count, class names, cache fingerprint/index SHA, and train/eval unit semantics.

## T1 model contract

Register:

    wsm_text_t1_chunk_transformer

Inputs are frozen cached chunk vectors.

Frozen architecture:

- input dim 768;
- hidden dim 192;
- input LayerNorm + Linear(768,192) + GELU + Dropout(0.20);
- deterministic sinusoidal chunk-position encoding generated at runtime;
- one `TransformerEncoderLayer`;
- 4 attention heads;
- feed-forward width = 384;
- dropout = 0.20;
- `batch_first=True`;
- `norm_first=True`;
- one encoder layer only;
- final LayerNorm;
- mask-aware mean pooling over valid chunks;
- two independent disease heads, each:
  - LayerNorm(192);
  - Dropout(0.20);
  - Linear(192,1).

Output:

    ModelOutput.preds shape [B,2]

No task ID enters the model.

Both logits are always produced, so simultaneous positives remain possible.

## Loss and optimizer

Reuse existing:

    wsm_masked_sparse_loss

Unknown target slots remain masked and never become negatives.

Optimizer:

    adamw_optimizer
    lr: 0.0001
    weight_decay: 0.01

Train settings for future production:

- epochs: 30;
- mixed precision: true;
- grad clip: 0.5;
- batch size: 32;
- early stopping patience: 6;
- min_delta: 0.0005.

No hyperparameter sweep.

## Configs

Create the three seed-clone configs now, but do not execute production training.

Required run names:

- seed42: `stage3_t1_xlmr_video_text_seed42`
- seed43: `stage3_t1_xlmr_video_text_seed43`
- seed44: `stage3_t1_xlmr_video_text_seed44`

All config semantics must be identical except seed/run_name.

Every config must include required instrumentation:

- `wsm_segment_metrics_callback`;
- checkpoint callback on `dev/mean_score`, max;
- early stopping on `dev/mean_score`, max;
- snapshot;
- summary;
- console logger;
- MLflow logger.

Every epoch-level evaluation must expose DEV and TEST_NONE/SOFT/HARD metrics, but only DEV may drive checkpoint/early stopping.

## Implementation/cache firewall

Before building the full external cache:

1. implement the encoder wrapper/cache builder/DataModule/model/configs/plugin registration;
2. plugin imports with no project-module warnings;
3. registry keys resolve;
4. configs validate;
5. synthetic transcript tests prove:
   - <510 content tokens => one chunk;
   - >510 => ordered multiple chunks;
   - no overlap;
   - no truncation;
   - special tokens excluded from pooling;
   - raw text/token IDs absent from artifact metadata;
6. verify the requested HF revision resolves;
7. verify downloaded model.safetensors SHA256 matches the frozen value;
8. verify encoder parameters `requires_grad=false` and extraction under no-grad/eval;
9. verify cache builder refuses overwrite;
10. verify no label/task/corpus prompt enters encoder input;
11. verify `git diff --check`;
12. verify `git diff origin/main -- src/audio src/video src/fusion src/description src/common pyproject.toml` empty;
13. append firewall evidence to PROGRESS;
14. commit and PUSH the firewall.

No full cache build before firewall is visible on origin.

## Build the full T1 cache exactly once

After firewall push, run exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/text/build_t1_xlmr_cache.py       --data-root /media/maxim/Databases/WSM_NEW       --audit-report /media/maxim/Programs/Features/WSM/stage3_text_audit/transcript_source_audit_v1.json       --output-root /media/maxim/Programs/Features/WSM/text_t1_xlmr_v1

No model-family alternative or parameter variant is allowed.

A pure runtime bug may be corrected only transparently with a pushed corrective firewall before a replacement cache build.

## Post-cache verification

Before final commit:

1. verify cache index SHA256;
2. verify expected 755 canonical video entries;
3. verify available/missing counts exactly match TASK-003A;
4. verify every present transcript SHA matches the audit report;
5. verify feature artifact SHA and metadata;
6. verify all chunk features finite float32 `[C,768]`, `C>=1`;
7. verify no raw text/token IDs stored;
8. instantiate all three configs from the frozen cache;
9. TRAIN dataset counts exactly D287/P266/total553;
10. verify no duplicate TRAIN `(corpus,video_id)` unit;
11. verify DEV and TEST protocol segment membership matches canonical existing protocol counts;
12. one TRAIN batch forward/loss/backward:
    - finite loss;
    - finite nonzero model gradients;
    - masked unknown slots do not affect loss;
13. one shape-only batch load from each evaluation stream;
14. do NOT compute or inspect DEV/Test performance metrics;
15. verify seed configs differ only seed/run_name;
16. final scope checks.

This task may inspect tensor shapes and IDs from Test loaders, but must not inspect Test text manually or calculate model-performance metrics.

## No production training

TASK-003B MUST NOT:

- run `chimera-ml train` for a production epoch;
- report DEV/Test model performance;
- select a checkpoint;
- tune hidden size/layers/dropout/chunking/encoder;
- compare another encoder.

The next manager task, if this pipeline passes review, will run the frozen T1 family on seeds42/43/44.

## Required PROGRESS evidence

Record:

- branch;
- frozen encoder/revision and resolved commit;
- model weight SHA;
- chunking/pooling contract;
- cache path/index SHA;
- canonical/cache counts;
- TRAIN unique-video counts;
- evaluation segment/protocol counts;
- registered keys;
- model trainable parameter count;
- smoke forward/loss/backward evidence;
- no performance metrics inspected;
- no production training;
- no source changes outside allowed scope;
- Stage 3 active;
- Stage 7/Final Test locked.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion
    git diff origin/main -- src/description
    git diff origin/main -- src/common
    git diff origin/main -- pyproject.toml
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only authorized files may differ.

## Acceptance criteria

TASK-003B passes only if:

- branch exactly `codex/task-003b`;
- one and only one T1 family is implemented;
- video-level TRAIN unit is enforced;
- segment-level evaluation broadcast is explicit and TRAIN-safe;
- frozen encoder/cache fingerprint is reproducible;
- encoder is frozen;
- no labels/prompts enter extraction;
- cache counts/fingerprints pass;
- required Chimera registrations/configs pass;
- smoke backward passes;
- DEV/Test performance is not inspected;
- no production training occurs;
- branch pushed;
- main/master untouched.

Passing TASK-003B authorizes no production run automatically.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, encoder/revision/resolved commit/model SHA, cache path/index SHA, cache counts, TRAIN unique-video counts, eval protocol counts, registry/config validation, trainable parameter count, smoke loss/gradient result, no performance metrics, no production training, Stage 3 active, Stage 7/Final Test locked.

For section 6 write only:

    Manager review of TASK-003B; do not start another task.

Stop.
