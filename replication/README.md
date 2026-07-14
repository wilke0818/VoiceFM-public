# VoiceFM replication audit (fork)

This is a fork of `oelemento/VoiceFM-public` that (a) **makes the released code correctly
runnable** on the paper's v3.0.0 cohort and (b) adds an independent **replication and audit**
under `replication/`. It reproduces the paper's in-distribution headline (0.947 vs 0.952) and
documents where the code diverges from the paper and where the out-of-sample results fall short.

**Changes to upstream (needed to run Run B correctly)** — see `replication/upstream_changes.patch`:
- `src/utils/preprocessing.py` — wrap `torchaudio.load` in try/except returning silence for
  undecodable files, so training doesn't crash on the handful of unreadable recordings.
- `scripts/unified_gsd_probes_v3.py` — write per-seed output (`unified_gsd_probes_seed{N}.json`)
  so parallel per-seed eval jobs don't clobber a shared file.
- `requirements.txt` — add `torchcodec` (needed by `torchaudio.load`; missing upstream).

**`replication/`:**
- **`FINDINGS.md`** — the full writeup, every comparison table with `n` (paper↔code differences,
  paper-vs-Run-B in-distribution + held-out, the frozen OOD baseline, split methodology, the
  silent batching bug, small-N handling, versioning/silent-file issues). Renders on GitHub.
- **`results/`** — the result JSONs the tables are computed from + `participant_counts.csv`
  (per-label positive `n` for the 846 train and 138 held-out cohorts).
- **`scripts/`** — the reconstruction code the shipped repo is missing: `run_heldout138.py`
  (Fig-3 held-out embedding extraction + prospective probe) and `frozen_heldout138.py` (frozen
  OOD baseline). `run/` holds the SLURM harness (training + eval sbatch, smoke config) used.

## What we did (replication steps)

1. **Cohort & labels.** Ran the repo's `preprocess.py --use-gsd` on the **v3.0.0** REDCap export
   (the paper's snapshot; the code's default points at v2.3.0 — see FINDINGS §8) and confirmed
   the 846/138 split and per-diagnosis labels against Table S2 (Σ|Δ|=0 over 19 diagnoses). The
   846/138 membership comes from the authors' `voicefm_train_test_splits.csv` oracle.

2. **Trained the code as-is (Run B)** on the exact 846 cohort, 5 seeds (42–46), on an 80 GB H100
   — with the upstream changes above (audio-decode shim, `torchcodec`, HF offline cache). We take
   this to be the model the paper intends. Command per seed:
   `python scripts/train.py --experiment exp_whisper_ft4_gsd_v3_seed<seed> --data-dir data/processed_v300`
   (see `replication/run/`). See FINDINGS §6 for the batch-config caveat: it trains at the base
   32×2. We also ran a paper-faithful variant (Run A, patching the 4 architecture divergences in
   FINDINGS §2); it is slightly worse — its result JSONs are in `results/` (`*_RUNA*.json`), kept
   out of the featured tables.

3. **Evaluation** with the authors' scripts:
   - **In-distribution (Fig 2):** `unified_gsd_probes_v3.py` — participant-mean 256-d embeddings,
     `create_participant_splits(seed)` 70/15/15 stratified by disease category, StandardScaler +
     LogisticRegression(C=1.0), 5 seeds.
   - **Held-out (Fig 3):** train the probe on all 846, score on the fixed 138 (added to v3.0.0
     after the training snapshot, so encoder-unseen). We reconstructed the held-out embedding
     extraction (using the **training** task-map so unseen v2 tasks → id 0), which the shipped
     scripts don't provide (FINDINGS §8).
   - **Frozen baseline:** raw whisper-large-v2 encoder, 1280-d mean-pool (FINDINGS §4).

4. **Cross-checked the split and batching against the paper text** (FINDINGS §5, §6).

## Caveat

The underlying B2AI-Voice audio and REDCap data are DUA-gated (PhysioNet doi:10.13026/k81f-qr68)
and are **not** included here. Only aggregate result JSONs and per-label counts are committed.
