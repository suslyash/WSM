# Stage 7 Final DEV Evidence — TASK-007B

## Result and boundary

`STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE INCOMPLETE`

The recovered accepted temporal-audio seed42 checkpoint is present at the
historical path below and its SHA256 was verified. The existing adapter strict-
loaded the epoch-4 payload with the archived graph compatibility path, all
adapter parameters remained frozen, and evaluation mode was retained. However,
the corrected canonical DEV-only reproduction materially failed the accepted
TASK-004H DEV scores, so the unified freeze remains incomplete. No replacement
run or source/config/model change is authorized.

TASK-007B otherwise preserved the frozen method set (temporal audio, full R4
trial012, equal-parameter shared fusion) and seeds 42–46. The six authorized
runs completed once, in order: audio45, R4-45, shared45, audio46, R4-46,
shared46. No retry, sweep, extra seed, config change, source/cache/dependency
change, or metric-driven stopping occurred. Test inspection is explicitly
`false`; no Test loader, Test metric key, raw run output, or Test log was
inspected.

## New run and checkpoint identities

| method | seed | MLflow/run identity | selected epoch | checkpoint path | SHA256 |
|---|---:|---|---:|---|---|
| audio | 45 | `bf1a681a7896499b884257cacee025c5` / `stage7_audio_final_seed45_2026-09-29_15-59_audio_mamba_segment_model_e1210d8f` | 6 | `logs/wsm_mm_pd_dep_v1/stage7_audio_final_seed45_2026-09-29_15-59_audio_mamba_segment_model_e1210d8f/checkpoints/epoch=6_dev_mean_score=0.7482.pt` | `1881bc526e9d17d77f424d2b053b5ea53f7ba5d85b82cb7643176c764da5f370` |
| full R4 | 45 | `14b0d142e78b4887bddac1a7dd286471` / `stage7_r4_trial012_final_seed45_2026-09-29_16-08_wsm_av_r3_disease_query_model_9a555979` | 20 | `logs/wsm_mm_pd_dep_v1/stage7_r4_trial012_final_seed45_2026-09-29_16-08_wsm_av_r3_disease_query_model_9a555979/checkpoints/epoch=20_dev_mean_score=0.7918.pt` | `cd30baf76392ab04e8b54d64ad67b260bae71216313942d968ad564a3422fc37` |
| shared fusion | 45 | `77dc95d7b8a84ed283dc76d22ef54e83` / `stage7_shared_fusion_final_seed45_2026-09-29_16-17_wsm_av_r3_disease_query_model_2eb9d5be` | 14 | `logs/wsm_mm_pd_dep_v1/stage7_shared_fusion_final_seed45_2026-09-29_16-17_wsm_av_r3_disease_query_model_2eb9d5be/checkpoints/epoch=14_dev_mean_score=0.7872.pt` | `ccd80ca0be94d25901eb6ad628671c4ebda24ca745a22af1136f5b6c6c4ea606` |
| audio | 46 | `fa12953b90bf46869fa012a6b42dcc95` / `stage7_audio_final_seed46_2026-09-29_16-25_audio_mamba_segment_model_57104699` | 10 | `logs/wsm_mm_pd_dep_v1/stage7_audio_final_seed46_2026-09-29_16-25_audio_mamba_segment_model_57104699/checkpoints/epoch=10_dev_mean_score=0.7576.pt` | `546ca0929e746692f18106922cecb10ea4781cca2243d31855e68d7d63983263` |
| full R4 | 46 | `7cfc409f33e24713be127c295e79a279` / `stage7_r4_trial012_final_seed46_2026-09-29_16-36_wsm_av_r3_disease_query_model_52637a4b` | 30 | `logs/wsm_mm_pd_dep_v1/stage7_r4_trial012_final_seed46_2026-09-29_16-36_wsm_av_r3_disease_query_model_52637a4b/checkpoints/epoch=30_dev_mean_score=0.7978.pt` | `18ee9c9e298fab13f807404cd625360c3e6c063a73ee4340d6f474207044f5e1` |
| shared fusion | 46 | `15ef29a1583a40a9a9c8fb6f63445574` / `stage7_shared_fusion_final_seed46_2026-09-29_16-47_wsm_av_r3_disease_query_model_8d38837e` | 25 | `logs/wsm_mm_pd_dep_v1/stage7_shared_fusion_final_seed46_2026-09-29_16-47_wsm_av_r3_disease_query_model_8d38837e/checkpoints/epoch=25_dev_mean_score=0.7867.pt` | `b99c153723265641057bf25c0aa7a57c01e5dc8cac638a1bcd40f9c87bb3041b` |

## Existing accepted identities

The following accepted identities were not retrained. Full paths and SHA256
were verified locally for all listed entries, including the recovered audio seed42:

- audio seed42: `logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt`; SHA `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`; historical run `multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77`; MLflow ID `not recorded`.
- audio seed43: `logs/wsm_mm_pd_dep_v1/frozen_audio_wavlm_l9_pool4_seed43_2026-09-28_13-12_audio_mamba_segment_model_08192c1b/checkpoints/epoch=5_dev_mean_score=0.7602.pt`; SHA `1a65f8dd605a6d942bed828f126e79a1f02930dd578176ba31f49c4dc10899dd`; MLflow `4119e9a654ec481ca42606f7f207b969`.
- audio seed44: `logs/wsm_mm_pd_dep_v1/frozen_audio_wavlm_l9_pool4_seed44_2026-09-28_13-21_audio_mamba_segment_model_8720f109/checkpoints/epoch=4_dev_mean_score=0.7719.pt`; SHA `7832d20a5431666365e2214423633f2be87bb27f37406d5a5dda8adda21e9852`; MLflow `ab9a3a73a0e94a2c889a69b25038d141`.
- full R4 seed42: `logs/wsm_mm_pd_dep_v1/optuna_r4_ra_stch_base_seed42_2026-09-28_01-54_wsm_av_r3_disease_query_model_r4-ra-stch-optuna-v1-18c6-012_0ce9c7a4/checkpoints/epoch=11_dev_mean_score=0.8191.pt`; SHA `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`; MLflow ID not recorded in accepted evidence.
- full R4 seed43: `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed43_2026-09-28_12-01_wsm_av_r3_disease_query_model_12038580/checkpoints/epoch=10_dev_mean_score=0.7614.pt`; SHA `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`; MLflow ID not recorded in accepted evidence.
- full R4 seed44: `logs/wsm_mm_pd_dep_v1/r4_ra_stch_optuna_trial012_confirm_seed44_2026-09-28_12-07_wsm_av_r3_disease_query_model_41bc602b/checkpoints/epoch=11_dev_mean_score=0.7801.pt`; SHA `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`; MLflow ID not recorded in accepted evidence.
- shared fusion seed42: `logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed42_2026-09-28_22-44_wsm_av_r3_disease_query_model_5db549e8/checkpoints/epoch=11_dev_mean_score=0.8245.pt`; SHA `21504702976a960ffea02a68778ebc3bf7e6cc15ea8c9866f182ff04cbd30785`; MLflow `2fa7446c90dc44e381e5f7b6aca9720b`.
- shared fusion seed43: `logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed43_2026-09-28_22-51_wsm_av_r3_disease_query_model_454bf80b/checkpoints/epoch=10_dev_mean_score=0.7713.pt`; SHA `ec746e519e551d69d59954dffff0a73968fb4ab555d43020b7722e635cc477ee`; MLflow `f6ef444b8e704065a679bb5294a39248`.
- shared fusion seed44: `logs/wsm_mm_pd_dep_v1/stage6_shared_fusion_trial012_seed44_2026-09-28_22-57_wsm_av_r3_disease_query_model_a398d7d8/checkpoints/epoch=11_dev_mean_score=0.7849.pt`; SHA `215e3a61c5dcb8ff7dd92ed5ca9336d66405c3f1ba6cc045841f08ec990d10a8`; MLflow `93391d46396a4475a69fe48e3ea84098`.

## New DEV evidence

Selected only by maximum `dev/mean_score`; the unified DEV pass used no Test
loader. Observed counts were D/P `621/312`.

| method | seed | D UAR/MF1/Score | P UAR/MF1/Score | Mean |
|---|---:|---|---|---:|
| audio | 45 | 0.697572/0.697224/0.697398 | 0.799034/0.799034/0.799034 | 0.748216 |
| full R4 | 45 | 0.725350/0.724803/0.725077 | 0.849758/0.867178/0.858468 | 0.791772 |
| shared fusion | 45 | 0.739916/0.739549/0.739733 | 0.825880/0.843464/0.834672 | 0.787202 |
| audio | 46 | 0.720495/0.719880/0.720187 | 0.801035/0.788813/0.794924 | 0.757556 |
| full R4 | 46 | 0.725537/0.725273/0.725405 | 0.859420/0.880909/0.870164 | 0.797785 |
| shared fusion | 46 | 0.727451/0.727451/0.727451 | 0.833264/0.858518/0.845891 | 0.786671 |

Five-seed DEV summaries use the frozen multiplier `2.7764451051977987` and
sample standard deviation. Values are ordered D, P, Mean; each tuple is
`mean / std / min / max / range / CI-low / CI-high`.

- audio: `0.721720/0.020065/0.697398/0.747918/0.050520/0.696805/0.746634`; `0.808569/0.012800/0.794924/0.827735/0.032811/0.792676/0.824463`; `0.765145/0.015234/0.748216/0.787827/0.039611/0.746230/0.784059`.
- full R4: `0.731250/0.015928/0.718527/0.759025/0.040498/0.711473/0.751027`; `0.848863/0.030547/0.804363/0.879259/0.074896/0.810934/0.886792`; `0.790057/0.021364/0.761445/0.819142/0.057697/0.763530/0.816583`.
- shared fusion: `0.737690/0.016475/0.724496/0.765324/0.040828/0.717234/0.758147`; `0.844139/0.024284/0.818152/0.883606/0.065454/0.813986/0.874292`; `0.790914/0.019857/0.771324/0.824465/0.053141/0.766258/0.815571`.

Paired Mean deltas, seeds42–46, are R4-audio
`0.031316/0.001210/0.008250/0.043556/0.040229`, shared-audio
`0.036638/0.011089/0.013021/0.038986/0.029115`, and shared-R4
`0.005323/0.009879/0.004771/-0.004570/-0.011114`. Their mean/std/95% CI
are respectively `0.024912/0.019122/[0.001169,0.048655]`,
`0.025769/0.013058/[0.009556,0.041984]`, and
`0.000858/0.008506/[-0.009704,0.011419]`. Mean wins/ties/losses are
R4-audio `5/0/0`, shared-audio `5/0/0`, shared-R4 `3/0/2`.
These are descriptive only and do not change the finalist set.

## Calibration and closure

The corrective audio42 DEV reproduction used observed rows only, sigmoid
probabilities, 15 equal-width bins, no threshold fitting, and no recalibration.
The recovered checkpoint SHA was exact and strict loading passed, but the
reproduced DEV result was D UAR/MF1/Score
`0.5047152194/0.3433026030/0.4240089112`, P UAR/MF1/Score
`0.5000000000/0.3988439306/0.4494219653`, Mean
`0.4367154383`, versus accepted D/P/Mean
`0.7480392157/0.8277353635/0.7878268765`. This exceeds the required absolute
tolerance `0.0005`, so the recovery stopped and the audio42 calibration is
diagnostic-only, not accepted for finalization: observed D/P `621/312`,
Brier D/P `0.4688533750/0.3203929081`, ECE-15 D/P
`0.4694130072/0.3149863942`. Consequently all-15 final verification and the
complete five-seed calibration summaries remain blocked; no new summary is
claimed. The six seed45/46 production runs were not rerun.

The original TASK-007B pass incidentally surfaced pre-existing historical Test
lines through a broad PROGRESS search. This corrective action performed no Test
loader construction or iteration and no Test metric query. The values were not
used for selection, checkpoint choice, statistics, calibration, or recovery:
`FINAL TEST VALUES NOT USED FOR TASK-007B SELECTION OR RECOVERY`.

No method, config, threshold, source, cache, or family changed after TASK-007A.
Stage 7 remains active. Final Test remains locked.
