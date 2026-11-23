"""Read-only syntax, entry and dependency check; never starts training."""
import ast
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
files = list(ROOT.rglob('*.py'))
for f in files:
    ast.parse(f.read_text(encoding='utf-8'), filename=str(f))
print(f'Python syntax: PASS ({len(files)} files)')
for mod in ('torch', 'torch_geometric', 'torch_scatter', 'torch_sparse', 'yaml', 'numpy', 'jinja2', 'easydict', 'ninja'):
    print(f'{mod}: {"available" if importlib.util.find_spec(mod) else "MISSING"}')
required = json.loads((ROOT / 'data/DPIKG/required_files.json').read_text())['required']
for name in required:
    print(f'data/{name}: {"available" if (ROOT / "data/DPIKG" / name).is_file() else "MISSING"}')
for name in ('train.py', 'evaluate.py', 'summarize_results.py'):
    assert (ROOT / 'scripts' / name).is_file(), name
print('Entry files: PASS')
