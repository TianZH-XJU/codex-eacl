#!/usr/bin/env python3
"""Run tiny CPU smoke tests using the upstream Laya many-option benchmark implementation."""
from __future__ import annotations
import argparse,json,os,platform,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
DATASETS=[("banking77","test"),("clinc150","test"),("massive","test")]
def main():
 parser=argparse.ArgumentParser(); parser.add_argument("--laya-root",default="external/laya")
 parser.add_argument("--limit",type=int,default=5); parser.add_argument("--seed",type=int,default=17)
 parser.add_argument("--model",default="english"); parser.add_argument("--output-dir",default="results/smoke")
 parser.add_argument("--strategies",default="single,tournament"); args=parser.parse_args()
 root=Path(args.laya_root).resolve(); benchmark=root/"research/benchmarks/tournament/benchmark.py"
 if not benchmark.exists(): raise SystemExit(f"Upstream benchmark not found: {benchmark}. Checkout Laya first.")
 out_dir=Path(args.output_dir).resolve(); out_dir.mkdir(parents=True,exist_ok=True)
 try:
  import torch
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","3"))); torch.set_num_interop_threads(1)
  if torch.cuda.is_available(): raise RuntimeError("CPU-only run refused: CUDA is available; isolate or explicitly disable CUDA.")
 except ImportError as exc: raise SystemExit("PyTorch is not installed") from exc
 report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),"python_version":sys.version,
  "torch_version":torch.__version__,"device":"cpu","cpu_threads":torch.get_num_threads(),"laya_root":str(root),
  "laya_commit":None,"limit_per_dataset":args.limit,"seed":args.seed,"strategies":args.strategies,"runs":[],"status":"passed"}
 try: report["laya_commit"]=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 except Exception as exc: report["notes"]=[f"Could not resolve Laya commit: {exc}"]
 for dataset,split in DATASETS:
  result_path=out_dir/f"{dataset}.json"
  command=[sys.executable,str(benchmark),"--dataset",dataset,"--split",split,"--limit",str(args.limit),"--seed",str(args.seed),
   "--strategies",args.strategies,"--model",args.model,"--device","cpu","--out",str(result_path)]
  print("RUN:"," ".join(command),flush=True); start=time.monotonic()
  env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":str(torch.get_num_threads()),
   "MKL_NUM_THREADS":str(torch.get_num_threads()),"OPENBLAS_NUM_THREADS":str(torch.get_num_threads())}
  proc=subprocess.run(command,cwd=root,text=True,capture_output=True,env=env)
  item={"dataset":dataset,"command":command,"return_code":proc.returncode,"elapsed_seconds":round(time.monotonic()-start,2),
   "stdout_tail":proc.stdout[-6000:],"stderr_tail":proc.stderr[-6000:],"result_path":str(result_path),"result_exists":result_path.exists()}
  report["runs"].append(item); print(json.dumps(item,indent=2),flush=True)
  if proc.returncode!=0: report["status"]="partial_failure"
 path=out_dir/"baseline_smoke_report.json"; path.write_text(json.dumps(report,indent=2)+"\n")
 print(f"Final report: {path}; status={report['status']}")
 return 0 if report["status"]=="passed" else 2
if __name__=="__main__": raise SystemExit(main())
