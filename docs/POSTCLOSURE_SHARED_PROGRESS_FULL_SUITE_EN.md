# POST-CLOSURE DEV-ONLY FULL ABLATION/MTL SUITE — NOT PART OF THE FROZEN STAGE-7/FINAL-TEST EVIDENCE

This is descriptive DEV-only evidence around TASK-011A Shared+Progress trial-28. It does not revise Stage-7, Final-Test, or paper roles.

POST-PRODUCTION REPORTING-ONLY COLLECTION RECOVERY

Collection integrity:
- manifest rows: 52
- scientific production results: 52
- referenced MLflow run IDs: 53
- duplicate-alias rows: 1
- row 34 aliases 44455703f7de44f6b032b4cde21ddd5d and b43e09ce179a49f199993628ad3c063d were equivalent FINISHED aliases for one frozen result.
- canonical row-34 ID: 44455703f7de44f6b032b4cde21ddd5d; alias ID: b43e09ce179a49f199993628ad3c063d.
- equivalence included config SHA 660cac8293e60e28648c4225edf78486dbb06087a9459d730bb49ada157d5be8, selected epoch 5, checkpoint SHA 4215e95623afe66cd42706f42f0f72e2662a679561cff844e89a7fde836e40f1, and identical DEV-only selected metrics D 0.746078/0.745256/0.745667, P 0.863906/0.874235/0.869071, Mean 0.807369.
- canonicalization is bookkeeping only and does not count the alias as another production run.


## Five-seed Shared+Progress parent confirmation

Only DEV metric keys were collected.
seed42 frozen: epoch 7; Mean 0.8336994328; checkpoint SHA ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346
seed43: D 0.7107376283846871/0.709874432650121/0.710306030517404; P 0.7901311249137336/0.80933614357158/0.7997336342426569; Mean 0.7550198323800305; epoch 9; checkpoint SHA ba280153b2a1dca9808f25c05e581fec72ab3404a3602233154ef26c9f62b926
seed44: D 0.7509803921568627/0.7503289473684212/0.750654669762642; P 0.8758454106280193/0.8856399645285249/0.8807426875782721; Mean 0.815698678670457; epoch 5; checkpoint SHA 775826bf903fced08e24eb9bfff267e414c1ea7d5e94201a466bc40642e93100
seed45: D 0.7304388422035482/0.7302703616526003/0.7303546019280742; P 0.8780538302277433/0.8807215097487228/0.879387669988233; Mean 0.8048711359581536; epoch 19; checkpoint SHA e4a6776902ca57885ceb8f1c5444736d16fb8257618181c027b2fc8ad996e3d9
seed46: D 0.7412231559290383/0.7403140485499822/0.7407686022395102; P 0.8543133195307109/0.863505764114691/0.858909541822701; Mean 0.7998390720311056; epoch 3; checkpoint SHA 374efc91f7ecf153f85ce42bee6af92475a229795d1221e6c8dfa4c0bb267478
five-seed Mean mean/std/min/max/range/95% t CI(df=4): 0.801826/0.029208/0.755020/0.833699/0.078680/[0.765559,0.838092]

## Three-seed suite rows

Rows are descriptive same-seed DEV comparisons; balancing and gradient-MTL use the Progress-tuned recipe and are not unbiased global rankings.
no_direct_pseudo: means 0.796265,0.763565,0.831011; mean/std 0.796947/0.033728; paired deltas -0.037435,+0.008545,+0.015312; wins 2/3.
uniform_reliability: means 0.813186,0.754999,0.814913; mean/std 0.794366/0.034104; paired deltas -0.020514,-0.000021,-0.000786; wins 0/3.
no_semantic_depression: means 0.796265,0.754059,0.824716; mean/std 0.791680/0.035551; paired deltas -0.037435,-0.000961,+0.009018; wins 1/3.
no_video_online: means 0.747278,0.756513,0.764815; mean/std 0.756202/0.008773; paired deltas -0.086421,+0.001493,-0.050884; wins 1/3.
no_audio_online: means 0.753836,0.752720,0.732638; mean/std 0.746398/0.011929; paired deltas -0.079864,-0.002300,-0.083060; wins 0/3.
task_aware_r4_progress: means 0.796663,0.758440,0.811574; mean/std 0.788893/0.027406; paired deltas -0.037036,+0.003420,-0.004125; wins 1/3.
shuffled_pseudo: means 0.796265,0.754059,0.818213; mean/std 0.789512/0.032606; paired deltas -0.037435,-0.000961,+0.002514; wins 1/3.
depression_only: means 0.575088,0.595824,0.624718; mean/std 0.598543/0.024926; paired deltas -0.258612,-0.159196,-0.190981; wins 0/3.
parkinson_only: means 0.701249,0.606849,0.713418; mean/std 0.673839/0.058333; paired deltas -0.132450,-0.148170,-0.102281; wins 0/3.
equal: means 0.807009,0.747673,0.807369; mean/std 0.787350/0.034362; paired deltas -0.026690,-0.007347,-0.008330; wins 0/3.
static_stch: means 0.797173,0.754862,0.800989; mean/std 0.784342/0.025601; paired deltas -0.036527,-0.000157,-0.014709; wins 0/3.
ra_stch: means 0.805618,0.753147,0.817926; mean/std 0.792230/0.034402; paired deltas -0.028081,-0.001873,+0.002227; wins 1/3.
gradnorm: means 0.810114,0.743680,0.802312; mean/std 0.785369/0.036314; paired deltas -0.023585,-0.011340,-0.013387; wins 0/3.
pcgrad: means 0.804145,0.744508,0.802795; mean/std 0.783816/0.034048; paired deltas -0.029554,-0.010512,-0.012904; wins 0/3.
cagrad: means 0.780505,0.737108,0.798899; mean/std 0.772171/0.031727; paired deltas -0.053194,-0.017912,-0.016800; wins 0/3.
dbmtl: means 0.787763,0.785387,0.805863; mean/std 0.793004/0.011199; paired deltas -0.045936,+0.030367,-0.009835; wins 1/3.

## Scope and quarantine

- DEV-only post-closure evidence; no Test loader, Test metric, mixed summary, or Final-Test artifact was inspected.
- D-only/P-only are not a deployable two-task model and are not averaged into a joint score.
- Online modality removals retain historical offline pseudo evidence.
- No final-method promotion, significance claim, or paper-role revision is authorized.
