#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --job-name=shp_dpo
#SBATCH --output=/scratch/%u/direct-preference-optimization/slurm_output/%x_%j.out
#SBATCH --error=/scratch/%u/direct-preference-optimization/slurm_output/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60GB
#SBATCH --mail-type=BEGIN,END,FAIL

# note that with AE and Pythia2.8, we can handle batch size 4 per gpu without OOM

# note that with A and Pythia2.8, we can handle batch size 8 per gpu without OOM
# one epoch takes 1.5 hours on 4 gpus. 3 should take 4.5 hours

singularity exec --nv --overlay /scratch/lhz209/conda_envs/pref/overlay-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif bash -c """
source /ext3/env.sh
source env/bin/activate
python -u train.py model=pythia28 \
datasets=[shp] \
loss=dpo \
exp_name=shp_dpo_one_epoch_sft_pythia28 \
gradient_accumulation_steps=8 \
batch_size=64 \
eval_batch_size=32 \
trainer=FSDPTrainer \
sample_during_eval=true \
model.fsdp_policy_mp=bfloat16 \
model.archive=.cache/lhz209/shp_pythia28_2024-03-08_17-20-09_762316/step-199680/policy.pt \
do_first_eval=true \
n_epochs=3
"""
