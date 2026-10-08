#!/usr/bin/env python3
"""TASK-011B deterministic DEV-only suite orchestration."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sqlite3, statistics, subprocess, sys
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "configs/wsm_mm_pd_dep_v1/postclosure_shared_progress_suite"
MANIFEST = SUITE / "suite_manifest.json"
LEDGER = REPO / "logs/task011b_shared_progress_suite/execution_ledger.json"
RESULTS = REPO / "logs/task011b_shared_progress_suite/dev_only_results.json"
REPORT = REPO / "docs/POSTCLOSURE_SHARED_PROGRESS_FULL_SUITE_EN.md"
PARENT = REPO / "logs/wsm_mm_pd_dep_v1/postclosure_shared_progress_optuna_seed42_2026-10-08_15-14_wsm_av_r3_disease_query_model_task011a-shared-progress-8a53-029_f5dbef29/task011a-shared-progress-8a53-029.yaml"
PSEUDO = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt")
UNIFORM = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_reliability_ablation/uniform_accepted_reliability_v1.pt")
NO_SEM = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_semantic_ablation/no_semantic_depression_v1.pt")
SHUFFLED = Path("/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt")
PSEUDO_SHA = "17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945"
T4 = 2.7764451051977987

def stop(msg: str) -> None:
    raise SystemExit(msg)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def add(group: str, variant: str, seeds: list[int], intervention: str) -> None:
        for seed in seeds:
            out.append({"group":group, "variant":variant, "seed":seed,
                        "intervention":intervention,
                        "run_name":f"task011b_{variant}_seed{seed}"})
    add("A","shared_progress_parent",[43,44,45,46],"new seed confirmation of frozen trial-28 parent")
    for v,i in [("no_direct_pseudo","warmup final_scale=0.0"),
                ("uniform_reliability","uniform accepted reliability cache"),
                ("no_semantic_depression","no-semantic-depression cache"),
                ("no_video_online","modality_available_override=[true,false]"),
                ("no_audio_online","modality_available_override=[false,true]")]:
        add("B",v,[42,43,44],i)
    add("C","task_aware_r4_progress",[42,43,44],"task_aware_fusion=false -> true")
    add("D","shuffled_pseudo",[42,43,44],"accepted deterministic shuffled pseudo cache")
    add("E","depression_only",[42,43,44],"training_task=depression; selector=dev/depression/score")
    add("E","parkinson_only",[42,43,44],"training_task=parkinson; selector=dev/parkinson/score")
    for v,i in [("equal","loss mode=equal"),("static_stch","loss mode=stch; tau=0.1"),
                ("ra_stch","loss mode=ra_stch; frozen controller constants")]:
        add("F",v,[42,43,44],i)
    for v,i in [("gradnorm","gradient method=gradnorm; alpha=1.5; weight_lr=0.025"),
                ("pcgrad","gradient method=pcgrad; seed-bound"),
                ("cagrad","gradient method=cagrad; calpha=0.5; rescale=1"),
                ("dbmtl","gradient method=dbmtl; beta=0.9; beta_sigma=0.0")]:
        add("G",v,[42,43,44],i)
    if len(out) != 52: stop(f"matrix error: {len(out)}")
    return out

def load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict): stop(f"not a mapping: {path}")
    return value

def sha_cache_contract(path: Path) -> dict[str, Any]:
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    accept, observed = value["pseudo_accept_mask"].bool(), value["observed_mask"].bool()
    classes = value["pseudo_class"].long()
    return {"accepted_counts":[int((accept[:,i] & ~observed[:,i]).sum()) for i in range(2)],
            "positive_counts":[int((accept[:,i] & (classes[:,i] == 1)).sum()) for i in range(2)],
            "negative_counts":[int((accept[:,i] & (classes[:,i] == 0)).sum()) for i in range(2)],
            "missing_counts":[int((~observed[:,i]).sum()) for i in range(2)]}

def replace_balancer(c: dict[str, Any], callback: str, params: dict[str, Any] | None = None) -> None:
    c["callbacks"] = [x for x in c["callbacks"]
                      if x.get("name") not in {"wsm_r4_balance_callback","wsm_gradient_mtl_callback"}]
    c["callbacks"].append({"name":callback,"params":params or {}})

def selector(c: dict[str, Any], key: str) -> None:
    for x in c["callbacks"]:
        if x.get("name") in {"checkpoint_callback","early_stopping_callback","sweep_target_callback"}:
            x.setdefault("params", {})["monitor"] = key
            x["params"]["mode"] = "max"

def materialize(parent: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any],dict[str,Any]]:
    c = deepcopy(parent)
    v, seed = spec["variant"], spec["seed"]
    c["seed"] = seed
    c["experiment_info"]["params"]["run_name"] = spec["run_name"]
    c["data"]["name"] = "wsm_ramps_semantic_dev_only_datamodule"
    dp = c["data"]["params"]
    for k in ("modality_available_override","expected_accepted_counts","expected_positive_counts","expected_negative_counts"):
        dp.pop(k, None)
    cache = PSEUDO
    accepted, positive, negative = [376,1801], [376,212], [0,1589]
    if v == "shared_progress_parent":
        pass
    elif v == "no_direct_pseudo":
        for x in c["callbacks"]:
            if x.get("name") == "wsm_pseudo_scale_warmup_callback": x["params"]["final_scale"] = 0.0
    elif v == "uniform_reliability": cache = UNIFORM
    elif v == "no_semantic_depression": cache, accepted = NO_SEM, [0,1801]
    elif v == "no_video_online": dp["modality_available_override"] = [True,False]
    elif v == "no_audio_online": dp["modality_available_override"] = [False,True]
    elif v == "task_aware_r4_progress": c["model"]["params"]["task_aware_fusion"] = True
    elif v == "shuffled_pseudo": cache = SHUFFLED
    elif v == "depression_only":
        c["loss"]["params"]["training_task"] = "depression"; selector(c,"dev/depression/score")
    elif v == "parkinson_only":
        c["loss"]["params"]["training_task"] = "parkinson"; selector(c,"dev/parkinson/score")
    elif v in {"equal","static_stch","ra_stch"}:
        c["loss"]["params"]["mode"] = {"equal":"equal","static_stch":"stch","ra_stch":"ra_stch"}[v]
        if v in {"static_stch","ra_stch"}: c["loss"]["params"]["tau"] = 0.1
        if v == "ra_stch": c["loss"]["params"].update({"controller_ema":0.8,"grad_ema":0.9,"reliability_ema":0.9,"weight_min":0.2,"weight_max":0.8})
    elif v in {"gradnorm","pcgrad","cagrad","dbmtl"}:
        c["loss"]["name"] = "wsm_gradient_mtl_loss"; lp = c["loss"]["params"]; lp.pop("mode",None)
        lp.update({"method":v,"pseudo_scale":0.0})
        if v == "gradnorm": lp.update({"gradnorm_alpha":1.5,"gradnorm_weight_lr":0.025})
        if v == "pcgrad": lp["pcgrad_seed"] = seed
        if v == "cagrad": lp.update({"calpha":0.5,"rescale":1})
        if v == "dbmtl": lp.update({"db_beta":0.9,"db_beta_sigma":0.0})
        replace_balancer(c,"wsm_gradient_mtl_callback")
    else: stop(f"unknown variant: {v}")
    dp["pseudo_cache_path"] = str(cache)
    if v == "no_semantic_depression":
        dp.update({"expected_accepted_counts":accepted,"expected_positive_counts":[0,212],"expected_negative_counts":[0,1589]})
    meta={"cache_path":str(cache),"cache_sha256":digest(cache),"accepted_counts":accepted,
          "positive_counts":positive,"negative_counts":negative,"missing_counts":[2665,3660],
          "model_mode":"task_aware" if c["model"]["params"].get("task_aware_fusion",True) else "shared",
          "loss_mode":c["loss"]["params"].get("mode"),"loss_method":c["loss"]["params"].get("method"),
          "selector":"dev/depression/score" if v=="depression_only" else "dev/parkinson/score" if v=="parkinson_only" else "dev/mean_score"}
    return c,meta

def prepare() -> None:
    if not PARENT.is_file(): stop(f"parent config missing: {PARENT}")
    for p in (PSEUDO,UNIFORM,NO_SEM,SHUFFLED):
        if not p.is_file(): stop(f"cache missing: {p}")
    parent=load(PARENT)
    if parent.get("seed") != 42 or parent["model"]["params"].get("task_aware_fusion") is not False: stop("parent is not frozen Shared trial-28")
    SUITE.mkdir(parents=True,exist_ok=True)
    for p in SUITE.glob("*.yaml"): p.unlink()
    output=[]
    for n,spec in enumerate(rows(),1):
        c,meta=materialize(parent,spec); path=SUITE/f"{n:02d}_{spec['group']}_{spec['variant']}_seed{spec['seed']}.yaml"
        path.write_text(yaml.safe_dump(c,sort_keys=False))
        item=dict(spec,index=n,config_path=str(path.relative_to(REPO)),parent_config=str(PARENT.relative_to(REPO)),parent_run="TASK-011A Shared+Progress trial-28",**meta)
        item["config_sha256"]=digest(path); output.append(item)
    MANIFEST.write_text(json.dumps({"schema":"task011b-bundle-v1","production_count":52,"test_access":False,"rows":output},indent=2,sort_keys=True)+"\n")
    print(f"PREPARE PASS rows=52 manifest={MANIFEST.relative_to(REPO)}")

def manifest() -> dict[str,Any]:
    if not MANIFEST.is_file(): stop("manifest missing")
    m=json.loads(MANIFEST.read_text())
    if m.get("production_count") != 52 or len(m.get("rows",[])) != 52: stop("manifest is not 52 rows")
    return m

def validate() -> None:
    m,parent=manifest(),load(PARENT); expected=rows()
    if [(x["group"],x["variant"],x["seed"]) for x in m["rows"]] != [(x["group"],x["variant"],x["seed"]) for x in expected]: stop("manifest order mismatch")
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src"}
    for row in m["rows"]:
        path=REPO/row["config_path"]
        if not path.is_file() or digest(path) != row["config_sha256"]: stop(f"config SHA mismatch: {path}")
        c=load(path)
        if c["seed"] != row["seed"] or c["experiment_info"]["params"]["run_name"] != row["run_name"]: stop(f"config identity mismatch: {path}")
        if c["data"]["name"] != "wsm_ramps_semantic_dev_only_datamodule": stop(f"not DEV-only: {path}")
        p=subprocess.run([str(REPO/".venv/bin/chimera-ml"),"validate-config","--config-path",str(path)],cwd=REPO,env=env,capture_output=True,text=True)
        if p.returncode: stop(f"validate-config failed for {path}\n{p.stdout}\n{p.stderr}")
    for p in (PSEUDO,UNIFORM,NO_SEM,SHUFFLED):
        got=sha_cache_contract(p)
        if got["missing_counts"] != [2665,3660]: stop(f"missing counts mismatch: {p}")
        if p == NO_SEM and got["accepted_counts"] != [0,1801]: stop(f"no-sem counts mismatch: {got}")
        if p != NO_SEM and got["accepted_counts"] != [376,1801]: stop(f"accepted counts mismatch: {p} {got}")
    if digest(PSEUDO) != PSEUDO_SHA: stop("canonical pseudo cache SHA mismatch")
    sys.path.insert(0,str(REPO/"src"))
    import chimera_plugin
    import torch
    from fusion.data.wsm_ramps_semantic_datamodule import WSMRampsSemanticDataModule,WSMRampsSemanticDevOnlyDataModule
    common=deepcopy(parent["data"]["params"]); base=WSMRampsSemanticDataModule(**common); dev=WSMRampsSemanticDevOnlyDataModule(**common)
    bt=[x["meta"]["segment_id"] for x in base.train_dataset.base._samples]; dt=[x["meta"]["segment_id"] for x in dev.train_dataset.base._samples]
    bv=[x["meta"]["segment_id"] for x in base.val_dataset.base._samples]; dv=[x["meta"]["segment_id"] for x in dev.val_dataset.base._samples]
    if bt != dt or bv != dv or len(bt) != 6325 or len(bv) != 933: stop("TRAIN/DEV membership mismatch")
    if set(dev.val_dataloader()) != {"dev"}: stop("DEV-only dataloader exposes non-DEV")
    from fusion.models.av_r3_disease_query import WSMAVR3DiseaseQueryModel
    from fusion.loss.gradient_mtl_loss import WSMGradientMTLLoss
    from common.callbacks.wsm_gradient_mtl_callback import WSMGradientMTLCallback
    batch=next(iter(dev.train_dataloader()))
    for aware in (False,True):
        model_params=deepcopy(parent["model"]["params"]); model_params["task_aware_fusion"]=aware
        model=WSMAVR3DiseaseQueryModel(**model_params)
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != 552775: stop("param count mismatch")
        for method in ("pcgrad","cagrad","gradnorm","dbmtl"):
            loss=WSMGradientMTLLoss(method=method,pseudo_scale=0.0,pcgrad_seed=42); loss.bind_model(model)
            wanted={"shared":14,"depression":18,"parkinson":18,"row":1} if aware else {"shared":23,"depression":14,"parkinson":14,"row":0}
            if loss.ownership_counts != wanted: stop(f"ownership mismatch {loss.ownership_counts}")
            model.zero_grad(set_to_none=True); value=loss(model(batch),batch); value.backward()
            if not torch.isfinite(value) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad): stop(f"nonfinite TRAIN smoke {method} {aware}")
        WSMGradientMTLCallback().on_fit_start(type("Trainer",(),{"model":model,"loss_fn":loss})())
    print("VALIDATE PASS rows=52 test_loader_accessed=False train_dev_equivalent=True")

def dry_run() -> None:
    m=manifest()
    for r in m["rows"]: print(f"{r['index']:02d} {r['group']} {r['variant']} seed={r['seed']} selector={r['selector']}")
    print("DRY-RUN PASS rows=52")

def run(resume: bool) -> None:
    m=manifest(); LEDGER.parent.mkdir(parents=True,exist_ok=True)
    ledger=json.loads(LEDGER.read_text()) if LEDGER.is_file() else {"schema":"task011b-execution-v1","rows":[]}; current={x["index"]:x for x in ledger["rows"]}
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":"src"}
    for r in m["rows"]:
        old=current.get(r["index"])
        if old and old.get("status")=="FAILED": stop(f"failed row {r['index']}; no automatic retry")
        if old and old.get("status")=="COMPLETE":
            if old.get("config_sha256") != r["config_sha256"]: stop(f"completed hash mismatch row {r['index']}")
            if not resume: stop(f"row {r['index']} complete; use --resume")
            continue
        if old and old.get("status")=="RUNNING": stop(f"interrupted row {r['index']}; no retry")
        entry={"index":r["index"],"group":r["group"],"variant":r["variant"],"seed":r["seed"],"run_name":r["run_name"],"config_path":r["config_path"],"config_sha256":r["config_sha256"],"status":"RUNNING"}
        current[r["index"]]=entry; LEDGER.write_text(json.dumps({"schema":"task011b-execution-v1","rows":[current[k] for k in sorted(current)]},indent=2)+"\n")
        log=LEDGER.parent/f"{r['index']:02d}_{r['variant']}_seed{r['seed']}.log"
        with log.open("w") as handle:
            p=subprocess.run([str(REPO/".venv/bin/chimera-ml"),"train","--config-path",str(REPO/r["config_path"])],cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
        entry.update({"status":"COMPLETE" if p.returncode==0 else "FAILED","returncode":p.returncode,"log_path":str(log.relative_to(REPO))})
        LEDGER.write_text(json.dumps({"schema":"task011b-execution-v1","rows":[current[k] for k in sorted(current)]},indent=2)+"\n")
        print(f"{entry['status']} {r['index']}/52 {r['variant']} seed={r['seed']}")
        if p.returncode: stop(f"production failed row {r['index']} returncode={p.returncode}; no retry")
    if len([x for x in current.values() if x.get("status")=="COMPLETE"]) != 52: stop("not 52 complete")
    print("RUN PASS rows=52")

def mlflow_dev(run_name: str) -> tuple[str,dict[str,float]]:
    db=sqlite3.connect(f"file:{REPO/'logs/mlflow.db'}?mode=ro",uri=True)
    tag=db.execute("SELECT run_uuid FROM tags WHERE key='mlflow.runName' AND value=? ORDER BY run_uuid DESC LIMIT 1",(run_name,)).fetchone()
    if tag is None: stop(f"MLflow run missing: {run_name}")
    run_id=tag[0]; keys=[x[0] for x in db.execute("SELECT DISTINCT key FROM metrics WHERE run_uuid=? AND key LIKE 'dev/%'",(run_id,))]
    if not keys: stop(f"no DEV metrics for {run_name}")
    values={}
    for key in keys: values[key]=[(int(s),float(v)) for _,v,s in db.execute("SELECT key,value,step FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",(run_id,key))]
    monitor="dev/mean_score" if "dev/mean_score" in values else "dev/depression/score" if "dev/depression/score" in values else "dev/parkinson/score"
    epoch,_=max(values[monitor],key=lambda x:x[1]); selected={key:value for key,entries in values.items() for step,value in entries if step==epoch}; selected["selected_epoch"]=float(epoch)
    db.close(); return run_id,selected

def checkpoint(run_name: str, epoch: int) -> tuple[str|None,str|None,int|None]:
    found=[]
    for root in (REPO/"logs/wsm_mm_pd_dep_v1").glob(f"{run_name}_*"):
        for p in (root/"checkpoints").glob("epoch=*.pt"):
            m=re.search(r"epoch=(\d+)",p.name)
            if m and int(m.group(1))==epoch: found.append(p)
    if not found: return None,None,None
    p=sorted(found)[0]; state=torch.load(p,map_location="cpu",weights_only=False); vals=state.get("model_state_dict",{})
    return str(p.relative_to(REPO)),digest(p),sum(v.numel() for v in vals.values() if hasattr(v,"numel"))

def write_report(result: list[dict[str,Any]]) -> None:
    grouped={}
    for r in result: grouped.setdefault(r["variant"],[]).append(r)
    frozen={"seed":42,"mean":0.8336994328,"epoch":7,"checkpoint_sha256":"ba02ea341800d8d6330ecca32e8a31dde3afaf61e5fb1e1eca4fd144d1ff0346"}
    parents=[frozen]+grouped.get("shared_progress_parent",[]); means=[x["mean"] if "mean" in x else x["metrics"]["dev/mean_score"] for x in parents]
    avg,sd=statistics.mean(means),statistics.stdev(means); ci=T4*sd/math.sqrt(5)
    lines=["# POST-CLOSURE DEV-ONLY FULL ABLATION/MTL SUITE — NOT PART OF THE FROZEN STAGE-7/FINAL-TEST EVIDENCE","",
           "This is descriptive DEV-only evidence around TASK-011A Shared+Progress trial-28. It does not revise Stage-7, Final-Test, or paper roles.","",
           "## Five-seed Shared+Progress parent confirmation","",
           "Only DEV metric keys were collected."]
    lines.append("seed42 frozen: epoch 7; Mean 0.8336994328; checkpoint SHA "+frozen["checkpoint_sha256"])
    for x in parents[1:]:
        m=x["metrics"]; lines.append(f"seed{x['seed']}: D {m.get('dev/depression/uar')}/{m.get('dev/depression/mf1')}/{m.get('dev/depression/score')}; P {m.get('dev/parkinson/uar')}/{m.get('dev/parkinson/mf1')}/{m.get('dev/parkinson/score')}; Mean {m.get('dev/mean_score')}; epoch {x['selected_epoch']}; checkpoint SHA {x.get('checkpoint_sha256')}")
    lines.append(f"five-seed Mean mean/std/min/max/range/95% t CI(df=4): {avg:.6f}/{sd:.6f}/{min(means):.6f}/{max(means):.6f}/{max(means)-min(means):.6f}/[{avg-ci:.6f},{avg+ci:.6f}]")
    lines += ["","## Three-seed suite rows","", "Rows are descriptive same-seed DEV comparisons; balancing and gradient-MTL use the Progress-tuned recipe and are not unbiased global rankings."]
    pmap={x["seed"]:x for x in parents[1:]}; pmap[42]={"metrics":{"dev/mean_score":frozen["mean"]}}
    for v,xs in grouped.items():
        if v=="shared_progress_parent": continue
        vals=[float(x["metrics"]["dev/mean_score"]) for x in xs]; ds=[vals[i]-float(pmap[xs[i]["seed"]]["metrics"]["dev/mean_score"]) for i in range(len(xs))]
        lines.append(f"{v}: means {','.join(f'{v:.6f}' for v in vals)}; mean/std {statistics.mean(vals):.6f}/{statistics.stdev(vals):.6f}; paired deltas {','.join(f'{d:+.6f}' for d in ds)}; wins {sum(d>0 for d in ds)}/3.")
    lines += ["","## Scope and quarantine","", "- DEV-only post-closure evidence; no Test loader, Test metric, mixed summary, or Final-Test artifact was inspected.","- D-only/P-only are not a deployable two-task model and are not averaged into a joint score.","- Online modality removals retain historical offline pseudo evidence.","- No final-method promotion, significance claim, or paper-role revision is authorized."]
    REPORT.write_text("\n".join(lines)+"\n")

def collect() -> None:
    m=manifest(); ledger=json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    if len([x for x in ledger.get("rows",[]) if x.get("status")=="COMPLETE"]) != 52: stop("collect requires 52 COMPLETE rows")
    out=[]
    for r in m["rows"]:
        run_id,metrics=mlflow_dev(r["run_name"]); epoch=int(metrics.pop("selected_epoch")); path,csha,params=checkpoint(r["run_name"],epoch)
        out.append({k:r[k] for k in ("index","group","variant","seed","run_name","selector","config_path","config_sha256")}|{"run_id":run_id,"selected_epoch":epoch,"metrics":metrics,"checkpoint_path":path,"checkpoint_sha256":csha,"parameters":params})
    RESULTS.parent.mkdir(parents=True,exist_ok=True); RESULTS.write_text(json.dumps({"schema":"task011b-dev-only-results-v1","test_access":False,"rows":out},indent=2,sort_keys=True)+"\n")
    write_report(out); print(f"COLLECT PASS rows=52 test_metrics_read=False report={REPORT.relative_to(REPO)}")

def main() -> None:
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    for x in ("prepare","validate","dry-run","run","resume","collect"): g.add_argument("--"+x,action="store_true")
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.validate: validate()
    elif a.dry_run: dry_run()
    elif a.run: run(False)
    elif a.resume: run(True)
    else: collect()
if __name__=="__main__": main()
