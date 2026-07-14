#!/bin/bash
#SBATCH --job-name=vfm_evBa
#SBATCH --output=$SCRATCH/voicefm/logs/evBa_%A_%a.out
#SBATCH --error=$SCRATCH/voicefm/logs/evBa_%A_%a.err
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=42-46
source "$HOME/.bashrc"; conda activate <env>; module load ffmpeg/5.1.4; set -eo pipefail
export HF_HUB_CACHE=$SCRATCH/voicefm/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $REPO
echo "host=$(hostname) seed=${SLURM_ARRAY_TASK_ID}"; nvidia-smi | head -20 || true
python -u scripts/unified_gsd_probes_v3.py --seeds ${SLURM_ARRAY_TASK_ID}   # writes results_v3/unified_gsd_probes_seed${SLURM_ARRAY_TASK_ID}.json
echo "DONE seed ${SLURM_ARRAY_TASK_ID}"
