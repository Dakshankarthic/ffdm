# Controlled Localization v1.1

Forensic localization of small AI-edited regions in photographs using matched controls and directed consistency.

## Scientific Goal
Investigate whether matched control data plus directed consistency improves localization of small AI edits on unseen source images and editors at controlled false-alarm rates.

Deployment operates on **ONE image only**. Originals, masks, and recipes are strictly training/evaluation assets.

## Directory Structure
- `src/controlled_localization/`: Core source modules (data schema, canonicalization, models, losses, training, inference).
- `configs/`: Experiment configs (`t0.yaml`, `t1.yaml`, `t2.yaml`, `t2o.yaml`, `t3.yaml`, `debug.yaml`).
- `tests/`: Contract, numerical, and architecture unit tests.
- `manifests/`: JSONL manifests for verified source/edit groups and evaluation sets.
- `runs/`: Training outputs, checkpoints, and evaluation predictions.
- `reports/`: Audit logs, inventory, and environment reports.

## Model Versions
- **T0**: A,A,C,C with segmentation only (no-processed-controls reference, matched forward count).
- **T1**: A,B,C,D with segmentation only (processed-controls exposure).
- **T2**: A,B,C,D with segmentation + directed view consistency (A->B, C->D).
- **T2O**: T2 + eligible outside-mask consistency (A->C, B->D).
- **T3**: T2O + auxiliary processing classifier.

