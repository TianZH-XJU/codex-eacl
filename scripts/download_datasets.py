#!/usr/bin/env python3
"""Download exact public split files pinned by the upstream Laya tournament benchmark."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
DATASETS = {
 "banking77":{"repo_id":"mteb/banking77","revision":"18072d2685ea682290f7b8924d94c62acc19c0b2","files":["data/train-00000-of-00001.parquet","data/test-00000-of-00001.parquet"]},
 "clinc150":{"repo_id":"clinc/clinc_oos","revision":"155b9c710419136e17307b80d0a13e68cd46b4ec","files":["plus/train-00000-of-00001.parquet","plus/validation-00000-of-00001.parquet","plus/test-00000-of-00001.parquet"]},
 "massive_en_us":{"repo_id":"SetFit/amazon_massive_intent_en-US","revision":"f7672a018e8ceb37fc0184dcfbb7e665155ffea6","files":["train.jsonl","validation.jsonl","test.jsonl"]}}
def sha256(path):
 digest=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
 return digest.hexdigest()
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--output-dir",default="data/raw"); parser.add_argument("--manifest",default="results/dataset_manifest.json")
 parser.add_argument("--datasets",default="all",help="all or comma-separated dataset keys"); args=parser.parse_args()
 try: from huggingface_hub import hf_hub_download
 except ImportError as exc: raise SystemExit("huggingface_hub missing; install requirements-cpu.txt first") from exc
 selected=list(DATASETS) if args.datasets=="all" else [x.strip() for x in args.datasets.split(",")]
 unknown=sorted(set(selected)-set(DATASETS))
 if unknown: raise SystemExit(f"Unknown dataset keys {unknown}; valid keys: {list(DATASETS)}")
 root=Path(args.output_dir); records=[]
 for key in selected:
  spec=DATASETS[key]; local_dir=root/key; local_dir.mkdir(parents=True,exist_ok=True)
  for filename in spec["files"]:
   path=Path(hf_hub_download(repo_id=spec["repo_id"],filename=filename,repo_type="dataset",revision=spec["revision"],local_dir=str(local_dir)))
   record={"dataset_key":key,"repo_id":spec["repo_id"],"revision":spec["revision"],"filename":filename,"path":str(path),"size_bytes":path.stat().st_size,"sha256":sha256(path),"status":"downloaded"}
   records.append(record); print(f"OK {key} {filename} {path.stat().st_size} bytes sha256={record['sha256']}")
 manifest={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","output_dir":str(root.resolve()),"files":records}
 out=Path(args.manifest); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(manifest,indent=2)+"\n")
 print(f"Wrote manifest: {out}"); return 0
if __name__=="__main__": raise SystemExit(main())
