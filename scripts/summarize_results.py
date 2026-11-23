"""Summarize the five published test results (sample SD, n-1)."""
import json
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads((ROOT / f'results/seed{s}/test_once_results.json').read_text()) for s in (42, 2024, 2025, 2026, 2027)]
for key in ('MR', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10'):
    values = [row['metrics'][key] for row in rows]
    print(f'{key}: {statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}')
