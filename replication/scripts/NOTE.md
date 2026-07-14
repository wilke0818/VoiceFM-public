# Reconstruction scripts

These fill gaps in the released repo so the Fig-3 held-out results can be reproduced:

- **`run_heldout138.py`** — per seed: load the trained VoiceFM checkpoint, extract the 138
  held-out participants' embeddings **using the training task-map (811)** so the 16
  held-out-only v2 task types map to id 0 (avoids the trained-task-embedding overflow that
  the naive path hits), merge with the saved 846-train embeddings, then run the authors'
  prospective probe (StandardScaler-fit-on-train, LR C=1.0). Reproduces the Run B / Run A
  held-out columns in `../FINDINGS.md`.
- **`frozen_heldout138.py`** — the frozen whisper-large-v2 OOD baseline (1280-d mean-pool,
  no task embedding) the paper omits from Fig 3.

Both take `--run-dir <path>` pointing at a VoiceFM-public checkout that holds `data/`,
`results_v3/`, and the `checkpoints_exp_whisper_ft4_gsd_v3_seed{42..46}/` produced by training;
no paths are hardcoded. They otherwise assume the layout this repo produces (e.g. the training
task-map is built from `data/processed_v3eval`), so treat them as a documented reference for the
missing procedure rather than turnkey tools. The upstream `scripts/train.py` and
`scripts/unified_gsd_probes_v3.py` are used unchanged for training and the in-distribution eval.
