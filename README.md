# ADRPathNet

Code and frozen results for **ADRPathNet: A Path-Reasoning Model for Traceable
Adverse Drug Reaction Prediction**. ADRPathNet uses an NBFNet backbone with the
receiver-state-aware RCDGate v0.2.1 as its internal message-gating module.
ADRPathNet is the public project and paper name. Existing class names, module
names, import paths and experiment version labels are retained for compatibility. The paper model has five 32-dimensional layers,
DistMult messages, PNA aggregation, gate scale 0.25, 485 gate parameters and 103,622
parameters in total. The actual gate is `g = 1 + 0.25 * tanh(z)`,
with a modulation range of 0.75 to 1.25. Historical internal defaults or comments
mentioning 0.5 do not override the formal configurations. Algorithms, formal configurations and recorded metrics are
preserved from the supplied project.

## Layout

- `backbone/nbfnet/`: NBFNet implementation and sparse extension sources.
- `adr_rcdgate/`: final gate, model overlay, training and test-ranking code.
- `configs/`: formal seeds 42, 2024, 2025, 2026 and 2027.
- `interpretation/`: frozen 12-query signed-gradient / beam-search implementation.
- `results/`: original formal results, path tables and Fig. 3 attribution exports.
- `data/DPIKG/`: required original-data checklist; verified original data is bundled.
- `scripts/`: train, evaluate, summarize, data preparation and release check.

## Environment

Linux with an NVIDIA GPU is recommended; the paper used an RTX 4090, PyTorch 2.1
and PyG 2.4.0. A compatible reference installation (Python 3.10) is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install -r requirements.txt
```

CUDA extension builds may require a matching CUDA toolkit and C++ compiler.
These install commands are guidance; no dependencies were installed during packaging.

## Data preparation

The verified original dataset is included under `data/DPIKG/`. DPIKG is the paper
name; `PVKG_811_CauseTailADR` is its historical engineering directory name.
The entity/relation mappings and all three split files match the SHA-256 records
of the paper attribution pipeline. ADR IDs and statistics also match the original
code snapshot byte-for-byte. No split, order or file content has been regenerated.
See `data/DATA_VERIFICATION.json` for checks and file hashes.

| File | Role |
|---|---|
| entity2id.json | Entity names to original IDs |
| relation2id.json | Contains=0, Affects=1, Causes=2 (Chinese relation names) |
| adr_ids.txt | 1,460 ADR candidate IDs |
| train_id.txt | 65,148 original training triples |
| valid_id.txt | 6,170 Causes validation triples |
| test_id.txt | 6,170 Causes test triples |
| all_true_triples.txt | Exact split union, used for filtered ranking only |
| dataset_stats.json | Original counts and type distributions |
| conversion_report.json | Original conversion provenance |

ID split rows are `head_id relation_id tail_id`; the original all-true file uses
entity/relation names. Loaders parse these formats, construct reasoning edges from
TRAIN only and add inverse relations. Training uses `--data-dir data/DPIKG`;
evaluation defaults to the same directory. Candidate ranking is ADR-only and
filtered with optimistic ties. Split sets are disjoint and contain no duplicates.
The original conversion report retains historical local paths as provenance;
those paths are not required at runtime. Runtime caches and redundant text/dict
copies from the ZIP are excluded. The dataset was provided by the supervising research team, which has authorized
public release of the code and data.

## Training

Run every command below from this repository root. Each output directory must be new:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/train.py --config configs/seed42.yaml --nbfnet-root backbone --data-dir data/DPIKG --output-dir outputs/seed42 --device cuda
```

Repeat with seeds 2024, 2025, 2026 and 2027 and corresponding new output directories.
The original policy calls torch.manual_seed and does not force deterministic
algorithms. Settings remain Adam 0.001, 10 epochs, batch size 16 and 16 strict
ADR-only negatives. Checkpoints are selected by validation ADR-only filtered MRR.

## Evaluation

```bash
python scripts/evaluate.py --run-dir outputs/seed42 --data-dir data/DPIKG --output outputs/seed42_test.json --device cuda
python scripts/summarize_results.py
```

The evaluation wrapper uses the original model constructor, strict checkpoint load,
TRAIN-only graph builder and unchanged final test metric functions. It supports
new training runs instead of requiring the historical seed42 checkpoint hash.
The summary command reads the five *published* result JSON files, using sample SD;
it does not summarize newly generated `outputs/` files.

## Paper path attribution and Fig. 3 data

The repository includes the paper Fig. 3 attribution results, but **does not
include the original seed42 checkpoint**. Retraining can produce different learned
weights and attribution paths; matching the seed alone does not guarantee the
exact paper paths, scores or subgraphs. The frozen attribution loader deliberately
checks the original checkpoint hash and rejects a different checkpoint.

The following commands reproduce the frozen paper-case workflow **only after
the exact checkpoint has been supplied**. To obtain it, use the original seed42
`best_valid_adr_mrr.pth` and put it in `results/seed42/`. Its expected SHA-256 is
`7e99b13d90f0c0fbff42304eaa92ae4e35316034bec7847c71dcc4dc0e568e08`.
The original config snapshot is already present. Defaults resolve relative to this
repository, without the old server directory:

```bash
python interpretation/scripts/stage3a0_runtime_preflight.py --output-dir outputs/path_preflight
python interpretation/scripts/stage3a1_extract_paths.py --preflight-summary outputs/path_preflight/stage3a0c_summary.json --query-manifest interpretation/queries_12_frozen.json --output-dir outputs/paths
python interpretation/scripts/stage3a1b_export_local_edge_attribution.py --path-dir results/paths --output-dir outputs/fig3
```

The local export reads the bundled frozen `results/paths/` tables and writes to a
new output directory. It is separate from the freshly extracted `outputs/paths/`;
use `--path-dir outputs/paths` explicitly if checking that new extraction.
Q04 (Levofloxacin Hydrochloride Capsules - Allergic reaction) and Q11
(Ciprofloxacin Hydrochloride Tablets - Rash) are the two paper cases. Q12 remains
because the original exporter also checks it. Beam size and path top-k remain 10.
The Fig. 3 CSVs in `results/fig3/` can be inspected without a checkpoint. The
original final figure drawing script is not supplied.

## Checks and provenance

```bash
python scripts/check_release.py
python scripts/train.py --help
python scripts/evaluate.py --help
```

`RELEASE_MANIFEST.json` records original archive paths and source hashes. Source
changes are limited to relocatable path defaults and the added evaluation wrapper.
Original result and provenance files retain historical absolute paths as evidence;
these are not runtime requirements. Historical experiments, logs, caches, backups
and model weights are excluded. See THIRD_PARTY_NOTICES.md for licensing status.
The complete workflow has not been rerun locally: the frozen paper checkpoint and
PyG/scatter/sparse dependencies are still needed for complete paper-case execution.
