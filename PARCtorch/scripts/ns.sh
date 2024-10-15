#!/usr/bin/env bash

#SBATCH -A sds_baek_energetic  # Account name (replace with yours)
#SBATCH -J PARCtorch_em_icml    # Job name
#SBATCH -o %x.out              # Standard output (%x expands to job name)
#SBATCH -e %x.err              # Standard error (%x expands to job name)
#SBATCH -p gpu                 # Partition name (adjust if needed)
#SBATCH --gres=gpu:a40:1        # Request one A40 GPU
#SBATCH -t 2:00:00              # Time limit (hh:mm:ss)
#SBATCH -c 1                   # Number of CPU cores
#SBATCH --mem=32G              # Memory pool for all cores

# ---------------------------
# Load Modules and Set Paths
# ---------------------------
module purge
module load apptainer pytorch/2.4.0  # Load specific PyTorch version

export PATH=~/.local/bin:$PATH  # Add user's local bin path

# Install PARCtorch within the container (adjust path if needed)
apptainer exec $CONTAINERDIR/pytorch-2.4.0.sif python -m pip install --user ./PARCtorch-0.2.2-py3-none-any.whl

# Adjust the dataset path as needed
TRAIN_DATA_PATH="/project/vil_baek/data/physics/PARCTorch/NavierStokes/train"
TEST_DATA_PATH="/project/vil_baek/data/physics/PARCTorch/NavierStokes/test"
MIN_MAX="../data/ns_min_max.json"
SAVE="../weights"
LOSS="../loss"

# Run your training script (replace ns_slurm.py with your actual script)
apptainer run --nv $CONTAINERDIR/pytorch-2.4.0.sif ns_slurm.py \
  --train_dirs $TRAIN_DATA_PATH \  # Adjust arguments as needed
  --min_max_output $MIN_MAX \
  --batch_size 8 \
  --num_epochs 50 \
  --save_dir  $SAVE \
  --loss_dir  $LOSS \
  --log_file training.log \
  --device cuda