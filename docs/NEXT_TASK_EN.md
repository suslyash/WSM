# TASK-003A: Transcript Source, Alignment, and Language Audit

## Authority and branch

This task follows **MANAGER-DECISION-073**.

Required branch:

    codex/task-003a

Start from current manager-updated `origin/main`.

This is the first active Stage-3 task after core Stage-6 completion.

It is **audit-only**. No model training or encoder implementation is authorized.

## Why this task comes first

Stage 1 intentionally left:

    text_available = false

for every canonical row.

The canonical manifest builder found video-level files at the conceptual location:

    <data_root>/<corpus>/<split>_labels/<video_id>/<video_id>.txt

but explicitly recorded:

    segment_text_alignment_established = false

Stage 3 T1 requires an audited-language transcript encoder. Before building it, determine what these transcript files actually contain and what text granularity is scientifically defensible.

Do not invent segment-level transcript labels.

## Required reading

Read exactly:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_3.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/common/data/wsm_manifest.py`
8. `scripts/common/build_wsm_manifest.py`

Do not load closed-stage source/config history unless needed to verify the canonical manifest contract.

## Allowed tracked files

Only:

- `scripts/text/audit_transcript_sources.py`
- `docs/STAGE3_TRANSCRIPT_AUDIT_EN.md`
- `docs/PROGRESS_EN.md`

No `src/*` changes.
No config changes.
No dependency changes.

## Inputs

Data root:

    /media/maxim/Databases/WSM_NEW

Use the current canonical manifest implementation:

    common.data.wsm_manifest.build_manifest

Do not modify dataset files.

Do not write inside the dataset root.

## External machine-readable report

Write exactly one external report:

    /media/maxim/Programs/Features/WSM/stage3_text_audit/transcript_source_audit_v1.json

The script MUST fail rather than overwrite an existing output path.

The report must contain no raw transcript text.

Record its SHA256 in PROGRESS and in the human-readable audit document.

## Canonical mapping

For every canonical manifest row:

- parse `segment_id` exactly as the JSON array `[corpus, video_id, segment_file]`;
- group by canonical `(corpus, split, video_id)`;
- derive the transcript path only as:

      data_root / corpus / f"{split}_labels" / video_id / f"{video_id}.txt"

- do not search arbitrary alternative transcript files;
- missing files are legitimate missing text availability and must be counted, not guessed.

Verify every grouped video maps to exactly one derived transcript path.

## Structural transcript audit

For every unique canonical video, record machine-readably:

- corpus;
- split;
- video_id;
- number of canonical segments;
- derived transcript path;
- exists;
- non-empty;
- byte size;
- SHA256 if present;
- UTF-8/UTF-8-SIG decode success;
- line count;
- nonblank line count;
- Unicode character count;
- whitespace-token count;
- deterministic format classification.

Allowed format classes:

- `plain_text`
- `webvtt`
- `srt_like`
- `timestamped_lines`
- `json_like`
- `unknown_text`
- `missing`
- `decode_error`

Implement deterministic local classification; no network and no LLM.

Record aggregate counts by corpus and split.

## Timestamp and segment-alignment audit

The central scientific question is whether text can be aligned to canonical segment files.

Inspect only canonical source metadata and the derived transcript file.

For every source CSV used by the manifest, record its column names and whether it provides explicit segment timing fields such as start/end/offset/time.

For transcript files, record counts of parseable timestamp spans/lines using deterministic regex/parsers for common SRT/VTT/timestamp forms.

Do NOT infer a segment start time from a suffix like `_005` unless source metadata explicitly establishes the mapping.

Do NOT assume fixed segment duration.

Do NOT use video duration plus ordinal arithmetic as a substitute for explicit segment offsets.

Record exactly one global alignment conclusion:

    SEGMENT TEXT ALIGNMENT ESTABLISHED

only if every text-available canonical segment can be mapped deterministically to a transcript span from explicit metadata.

Otherwise record exactly:

    SEGMENT TEXT ALIGNMENT NOT ESTABLISHED

If alignment is not established, record:

    T1 TEXT UNIT SHOULD BE VIDEO-LEVEL TRANSCRIPT

This means a future T1 must treat one transcript as one video-level text unit; it may later broadcast a frozen video prediction/feature to segment rows for metric compatibility, but training must not silently multiply one transcript by segment count without an explicit weighting policy.

Do not implement that future policy in TASK-003A.

## Exact-duplicate / leakage audit

Using transcript SHA256 only:

- count duplicate transcript contents within each split;
- count identical transcript contents across train/dev;
- count identical transcript contents across train/test;
- count identical transcript contents across dev/test;
- list only video IDs/hashes, never raw text.

Any cross-split exact duplicate must be highlighted as a leakage risk.

Do not use diagnosis labels in duplicate analysis.

## TRAIN/DEV language audit

Language/model-family decisions must not use Test lexical content.

For TRAIN and DEV only:

1. compute aggregate Unicode-script character counts and proportions at least for:
   - Latin;
   - Cyrillic;
   - Greek;
   - Arabic;
   - Han;
   - Hiragana/Katakana;
   - Hangul;
   - digits;
   - other letters;
2. create a deterministic manual-audit sample of up to 24 unique videos:
   - stratify by corpus x split over TRAIN and DEV;
   - use stable SHA256 ordering of `corpus|split|video_id`;
   - target up to 6 videos per stratum;
3. Codex may inspect those local transcript files manually to assign a coarse language label per sampled video.

Do NOT commit transcript snippets or verbatim phrases.

The committed report/document may contain only:

- sampled video IDs;
- coarse language label;
- mixed-language yes/no;
- unreadable/empty status;
- aggregate counts.

No external language-ID service/model download is authorized.

If an already-installed offline language-ID package happens to exist, do not make the task depend on it; manual + Unicode-script audit remains authoritative.

## Test-content firewall

For TEST:

Allowed:
- path existence;
- byte size;
- hash;
- decode success;
- line/character/token counts;
- structural format class;
- timestamp-pattern counts;
- duplicate-hash checks.

Forbidden:
- manual lexical reading;
- language-based model choice;
- vocabulary analysis;
- label association;
- performance analysis.

No Test prediction or metric may be read.

## Coverage and readiness conclusions

Report canonical segment-level and unique-video-level transcript coverage for TRAIN/DEV/TEST and by corpus.

Missing transcripts remain a future availability-mask case; do not fabricate text.

Record exactly one readiness string:

    T1 TRANSCRIPT DATA CONTRACT READY

if all of the following hold:

1. canonical video-to-transcript path derivation is deterministic;
2. existing TRAIN/DEV transcript files are auditable without uncontrolled decode ambiguity;
3. language/granularity evidence is sufficient to specify a T1 input contract;
4. any duplicate/leakage risk is explicitly enumerated.

Otherwise record exactly:

    T1 TRANSCRIPT DATA CONTRACT BLOCKED

and name the concrete blocker.

This readiness result does NOT authorize training automatically.

## Human-readable audit document

Create `docs/STAGE3_TRANSCRIPT_AUDIT_EN.md` containing:

- scope and no-Test-selection boundary;
- canonical counts;
- transcript coverage by split/corpus at video and segment level;
- format/encoding audit;
- timestamp/source-metadata audit;
- exact alignment conclusion;
- duplicate/leakage audit;
- TRAIN/DEV Unicode-script summary;
- deterministic manual sample IDs and coarse language labels;
- exact readiness string;
- recommended **data granularity only** for T1;
- explicit statement that no encoder/model was selected or downloaded.

Do not include raw transcript text.

## Pre-execution firewall

Before running the full audit:

1. implement the script;
2. run `--help`;
3. run synthetic parser tests in a temporary directory for:
   - plain text;
   - SRT-like;
   - WebVTT;
   - timestamped lines;
   - invalid UTF-8;
   - missing file;
4. verify output schema contains no raw text fields;
5. verify script refuses overwrite;
6. `git diff --check`;
7. `git diff origin/main -- src configs pyproject.toml` must be empty;
8. append firewall evidence to PROGRESS;
9. commit and push the firewall.

No full dataset audit before the firewall commit is visible on origin.

## Execute the full audit once

Run exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/text/audit_transcript_sources.py       --data-root /media/maxim/Databases/WSM_NEW       --output /media/maxim/Programs/Features/WSM/stage3_text_audit/transcript_source_audit_v1.json

Do not rerun to change conclusions.

A runtime/parser bug may be corrected only if documented, with a new corrective firewall before a replacement full audit. No data/content-driven tuning.

## Final evidence

After the full audit:

- verify report SHA256;
- create the human-readable audit document;
- append compact final evidence to PROGRESS;
- record whether the report inspected Test lexical content: must be `false`;
- record no Test metrics/predictions inspected;
- record no model/encoder selected;
- commit and push.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git diff origin/main -- configs
    git diff origin/main -- pyproject.toml
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Only the three authorized tracked files may differ.

## Acceptance criteria

TASK-003A passes only if:

- branch exactly `codex/task-003a`;
- no source/config/dependency change;
- canonical path mapping is deterministic and audited;
- full transcript coverage/format/decode audit is complete;
- segment alignment conclusion follows explicit metadata only;
- no ordinal/fixed-duration alignment is invented;
- exact duplicate cross-split audit is complete;
- TRAIN/DEV language audit is deterministic and no raw text is committed;
- Test lexical content is not manually inspected or used for model choice;
- external JSON contains no raw transcript text and has a recorded SHA;
- exact readiness string is recorded;
- firewall precedes the one full audit;
- no training/model download/encoder selection occurs;
- branch pushed;
- main/master untouched.

Passing TASK-003A closes only the Stage-3 text data-contract audit. It authorizes no T1 implementation automatically.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include:

- branch;
- firewall/final commit SHAs;
- pushed status;
- main/master untouched;
- report path/SHA;
- canonical coverage;
- format/decode summary;
- exact alignment string;
- duplicate/leakage summary;
- TRAIN/DEV language/script summary;
- exact readiness string;
- explicit no raw text committed;
- Test lexical inspection = false;
- no Test metrics/predictions;
- no model/encoder selected;
- Stage 3 active;
- Stage 6 complete;
- Stage 7/Final Test locked.

For section 6 write only:

    Manager review of TASK-003A; do not start another task.

Stop.
