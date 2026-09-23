#!/usr/bin/env python3
"""Independently audit a TRAIN+DEV video feature cache without extraction."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
import sys
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common.data.wsm_manifest import build_manifest
from video.features.yolov8_body_roi import weights_sha256

def _version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return "unavailable"

def _git_state():
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT_ROOT,text=True).strip()
        dirty=bool(subprocess.check_output(["git","status","--short"],cwd=PROJECT_ROOT,text=True).strip())
        return {"commit":commit,"dirty":dirty}
    except Exception as exc: return {"commit":None,"dirty":None,"error":str(exc)}

def _source_path(data_root, row):
    corpus, video_id, segment_file=json.loads(row["segment_id"])
    return data_root / corpus / f"{row['split']}_labels" / video_id / "segments" / segment_file

def _load_index(path):
    if not path.is_file(): raise FileNotFoundError(f"cache index is missing: {path}")
    records=[]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): records.append(json.loads(line))
    return records

def _validate_artifact(record, args):
    import torch
    path=Path(record["cache_path"])
    artifact=torch.load(path,map_location="cpu",weights_only=False)
    features, mask=artifact["features"], artifact["valid_mask"]
    if artifact["segment_id"] != record["segment_id"]: raise ValueError("segment_id mismatch")
    if artifact["cache_fingerprint"] != record["cache_fingerprint"]: raise ValueError("fingerprint mismatch")
    if artifact["model_name"] != args.model_name or artifact["model_revision"] != args.model_revision: raise ValueError("CLIP identity mismatch")
    if artifact["detector_weights_sha256"] != args.yolo_sha: raise ValueError("YOLO SHA mismatch")
    if artifact["detection_parameters"] != {"confidence":0.5,"iou":0.5,"imgsz":640}: raise ValueError("detector settings mismatch")
    if not isinstance(features,torch.Tensor) or tuple(features.shape)!=(args.target_frames,512): raise ValueError("feature shape mismatch")
    if not isinstance(mask,torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape)!=(args.target_frames,): raise ValueError("mask mismatch")
    if not bool(torch.isfinite(features[mask]).all()): raise ValueError("non-finite valid feature")
    if not torch.equal(features[~mask],torch.zeros_like(features[~mask])): raise ValueError("non-zero invalid feature")
    return {"temporal_length":int(features.shape[0]),"feature_dim":int(features.shape[1]),
            "valid_detection_count":int(artifact["valid_detection_count"]),
            "detection_coverage":float(artifact["detection_coverage"])}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True); p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--report-output",type=Path,required=True); p.add_argument("--splits",required=True)
    p.add_argument("--model-name",required=True); p.add_argument("--model-revision",required=True)
    p.add_argument("--yolo-weights",type=Path,required=True); p.add_argument("--target-frames",type=int,required=True)
    args=p.parse_args()
    requested=[x.strip() for x in args.splits.split(",") if x.strip()]
    if requested != ["train","dev"]: raise ValueError("TASK-002G audit requires --splits train,dev")
    if args.target_frames != 60: raise ValueError("TASK-002G audit requires target_frames=60")
    if not args.yolo_weights.is_file(): raise FileNotFoundError(args.yolo_weights)
    args.yolo_sha=weights_sha256(args.yolo_weights)
    rows,audit=build_manifest(args.data_root)
    selected=[row for row in rows if row["split"] in requested and row["video_available"]]
    expected_by_split={split:sum(row["split"]==split for row in selected) for split in requested}
    expected_ids={row["segment_id"] for row in selected}
    records=_load_index(args.cache_root/"cache_index.jsonl")
    ids=[record["segment_id"] for record in records]
    duplicate_ids=len(ids)!=len(set(ids))
    test_rows=[record for record in records if record.get("split")=="test"]
    selected_records=[record for record in records if record.get("segment_id") in expected_ids]
    missing=expected_ids-{record.get("segment_id") for record in selected_records}
    fingerprints=[record.get("cache_fingerprint") for record in selected_records if record.get("status") in {"extracted","reused"}]
    artifact_errors=[]; valid_records=[]
    for record in selected_records:
        if record.get("status") in {"extracted","reused"}:
            try: valid_records.append({**record,**_validate_artifact(record,args)})
            except Exception as exc: artifact_errors.append({"segment_id":record["segment_id"],"reason":str(exc)})
    def cov(records):
        vals=[float(x["detection_coverage"]) for x in records if x.get("status") in {"extracted","reused"} and "detection_coverage" in x]
        return {"count":len(vals),"mean":sum(vals)/len(vals) if vals else None,"min":min(vals) if vals else None,"max":max(vals) if vals else None}
    failures=[record for record in selected_records if record.get("status")=="failed"]
    report={"schema_version":"wsm-video-cache-audit-v1","requested_splits":requested,
            "manifest_fingerprint":audit["manifest_fingerprint"]["value"],"expected_by_split":expected_by_split,
            "expected_total":len(selected),"indexed_selected_count":len(selected_records),
            "missing_record_count":len(missing),"missing_segment_ids":sorted(missing),
            "test_rows_indexed":len(test_rows),"test_rows_processed":False,
            "duplicate_segment_id":duplicate_ids,"cache_fingerprints_unique":len(fingerprints)==len(set(fingerprints)),
            "successful_artifacts_valid":not artifact_errors,"artifact_errors":artifact_errors,
            "success_by_split":{split:sum(r.get("status") in {"extracted","reused"} and r.get("split")==split for r in selected_records) for split in requested},
            "failed_by_split":{split:sum(r.get("status")=="failed" and r.get("split")==split for r in selected_records) for split in requested},
            "coverage_by_split":{split:cov([r for r in selected_records if r.get("split")==split]) for split in requested},
            "coverage_overall":cov(selected_records),"failure_categories":{cat:sum(r.get("failure_category")==cat for r in failures) for cat in sorted({r.get("failure_category") for r in failures})},
            "detector_weights_sha256":args.yolo_sha,"model_name":args.model_name,"model_revision":args.model_revision,
            "target_frames":args.target_frames,"git":_git_state(),
            "package_versions":{name:_version(name) for name in ("torch","transformers","ultralytics")},
            "test_usage":{"model_predictions_inspected":False,"performance_metrics_inspected":False,"selection_or_tuning_performed":False},
            "complete_for_requested_splits":not missing and not duplicate_ids and not test_rows and not artifact_errors and
                all(record.get("status") in {"extracted","reused","failed"} for record in selected_records) and len(selected_records)==len(expected_ids)}
    args.report_output.parent.mkdir(parents=True,exist_ok=True)
    args.report_output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"expected_total":len(selected),"missing":len(missing),"complete":report["complete_for_requested_splits"]},sort_keys=True))
    return 0 if report["complete_for_requested_splits"] else 2
if __name__=="__main__": raise SystemExit(main())
