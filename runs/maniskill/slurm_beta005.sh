#!/bin/bash
#SBATCH --job-name=r2d-msk-b005
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --nodelist=worker-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Treatment arm. beta ramps from 0 to 0.05 across steps 400k-700k, so the
# progress head and the progress critic are both fitted before either can move
# the actor.

source "$(dirname "$(readlink -f "$0")")/_common.sh"

echo "Arm: beta=0.05 (warm-up 400k-700k)"
run_all 0.05 beta005
echo "Job finished"
