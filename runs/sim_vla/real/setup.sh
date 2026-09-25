# Sourced from the repository root by the scripts in this folder: where each
# account on the H100 cluster keeps the SO-101 data and the SmolVLA cache.
case "$USER" in
  tuannl)
    SO101_DATA=/home/tuannl/mnt_data/data/so101
    export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint
    ;;
  duongnm2)
    SO101_DATA=/home/duongnm2/projects/r2dreamer-graph/data/so101
    export HF_HOME=/home/duongnm2/projects/r2dreamer-graph/checkpoints
    ;;
  *)
    echo "runs/sim_vla/real/setup.sh: no data paths for user $USER" >&2
    return 1
    ;;
esac
mkdir -p data "$SO101_DATA/sim_vla_real"
ln -sfn "$SO101_DATA/sim_vla_real" data/sim_vla_real
