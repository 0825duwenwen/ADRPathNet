"""Evaluate a freshly trained ADRPathNet checkpoint with the unchanged ranking code."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir', required=True)
p.add_argument('--data-dir', default=str(ROOT / 'data/DPIKG'))
p.add_argument('--output', required=True)
p.add_argument('--device', default='cuda')
a = p.parse_args()
import torch
from adr_rcdgate.config import load_config
from adr_rcdgate.overlay import construct_model
from adr_rcdgate.stage2 import _load_test_once_data, _evaluate_test
run = Path(a.run_dir)
config = load_config(run / 'config_snapshot.yaml')
output = Path(a.output)
if output.exists():
    raise FileExistsError(output)
model = construct_model(ROOT / 'backbone', True, config.model).to(a.device)
checkpoint = torch.load(run / 'best_valid_adr_mrr.pth', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
data = _load_test_once_data(a.data_dir, torch.device(a.device))
metrics = _evaluate_test(model, data['graph'], data['test'], data['adr'], data['all_true'], data['relation_id'], config.validation.batch_size)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'seed': config.seed, 'metrics': metrics}, indent=2) + '\n', encoding='utf-8')
print(json.dumps(metrics, indent=2))
