from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from adr_rcdgate.stage1 import run_stage1
from adr_rcdgate.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--nbfnet-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        summary = run_stage1(args.nbfnet_root, args.data_dir, args.output_dir, load_config(args.config), args.device)
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
