#!/bin/bash
#SBATCH --job-name=vfm_s%a
#SBATCH --output=$SCRATCH/voicefm/logs/train_s%a_%A.out
#SBATCH --error=$SCRATCH/voicefm/logs/train_s%a_%A.err
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=42-46

# VoiceFM-public AS-IS, 5-seed replication on the v3.0.0 cohort (846 train, exact authors' splits).
# Run B of the audit (ORIGINAL authors code, NOT the spec-054 replication): original authors' code, paper's batch (32x2), micro-batch 32 on 80GB H100.
# DO NOT submit until the seed-42 smoke test passes.
source "$HOME/.bashrc"
conda activate <env>
module load ffmpeg/5.1.4          # torchcodec (torchaudio.load) needs libav*
set -eo pipefail                  # AFTER env setup — bashrc/conda are not `set -u`-clean (BASHRCSOURCED)
export HF_HUB_CACHE=$SCRATCH/voicefm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd $REPO

SEED=${SLURM_ARRAY_TASK_ID}
echo "seed=${SEED}  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi | head -20

python scripts/train.py \
    --experiment exp_whisper_ft4_gsd_v3_seed${SEED} \
    --data-dir data/processed_v300

echo "TRAIN DONE seed=${SEED} rc=$?"
