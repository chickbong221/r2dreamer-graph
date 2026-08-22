#!/bin/bash
#SBATCH --job-name=r2d-msk-b0
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodelist=worker-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Control arm. Every loss is on and both progress heads train exactly as in the
# other arm; beta=0 only keeps the progress advantage out of the actor, so any
# difference between the two arms is the shaping and nothing else.

source "$(dirname "$(readlink -f "$0")")/_common.sh"

echo "Arm: beta=0 (progress heads train, actor never sees them)"
run_all 0.0 beta0
echo "Job finished"
