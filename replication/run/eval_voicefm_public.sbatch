#!/bin/bash
#SBATCH --job-name=vfm_eval_B
#SBATCH --output=$SCRATCH/voicefm/logs/eval_%j.out
#SBATCH --error=$SCRATCH/voicefm/logs/eval_%j.err
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00

# GSD probe eval for Run B (authors' as-is code). Step 1 = FROZEN Whisper baseline
# (no training) -> independent harness check vs paper's Frozen Whisper 0.926. Step 2 =
# unified probes: frozen_whisper (validation) + voicefm_whisper (= our Run B checkpoints);
# voicefm_hubert / frozen_hubert are skipped (not trained). Reads data/processed_v3
# (-> processed_v300, 846 cohort). NOTE: results_v3/unified_gsd_probes.json (paper ref)
# is backed up as unified_gsd_probes_PAPER_committed.json.
source "$HOME/.bashrc"
conda activate <env>
module load ffmpeg/5.1.4
set -eo pipefail
export HF_HUB_CACHE=$SCRATCH/voicefm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd $REPO
echo "host=$(hostname)"; nvidia-smi | head -20

echo "=== [1/2] extract FROZEN Whisper baseline embeddings (no training) ==="
python -u scripts/whisper_extract_embeddings_v3.py

echo "=== [2/2] unified GSD probes (frozen_whisper check + voicefm_whisper = Run B) ==="
python -u scripts/unified_gsd_probes_v3.py --seeds 42 43 44 45 46

echo "EVAL DONE (Run B) rc=$?"
