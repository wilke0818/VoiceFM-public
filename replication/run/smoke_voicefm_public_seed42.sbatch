#!/bin/bash
#SBATCH --job-name=vfm_smoke_s42
#SBATCH --output=$SCRATCH/voicefm/logs/smoke_s42_%j.out
#SBATCH --error=$SCRATCH/voicefm/logs/smoke_s42_%j.err
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00

# SMOKE TEST — seed 42, 2 epochs. Validates build/data/forward-backward/checkpoint
# on an 80GB H100 at the paper's micro-batch 32 before committing the full 5-seed array.
source "$HOME/.bashrc"
conda activate <env>      # transformers 5.13 verified compatible + all deps present
module load ffmpeg/5.1.4          # torchcodec (torchaudio.load) needs libav*
set -eo pipefail                  # AFTER env setup — bashrc/conda are not `set -u`-clean (BASHRCSOURCED)
export HF_HUB_CACHE=$SCRATCH/voicefm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd $REPO

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi | head -20

python scripts/train.py \
    --experiment exp_whisper_ft4_gsd_v3_seed42_smoke \
    --data-dir data/processed_v300

echo "SMOKE DONE rc=$?"
