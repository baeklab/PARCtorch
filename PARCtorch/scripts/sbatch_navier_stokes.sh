#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic            # Account name
#SBATCH -J PARCtorch_training             # Job name
#SBATCH -o PARCtorch_training_%j.out      # Standard output with job ID
#SBATCH -e PARCtorch_training_%j.err      # Standard error with job ID
#SBATCH -p gpu                            # Partition name
#SBATCH --gres=gpu:a40:1                  # Request 1 NVIDIA A40 GPU (instead of A100 if you meant A40)
#SBATCH -t 1:00:00                        # Time limit (1 hour, adjust as needed)
#SBATCH -c 4                              # Number of CPU cores
#SBATCH --mem=64G                         # Total memory

# ---------------------------
# Environment Setup
# ---------------------------
module purge
module load apptainer/1.2.2 pytorch/2.4.0   # Ensure correct version of pytorch is loaded

# Run the PyTorch container with GPU support
export PATH=~/.local/bin:$PATH
apptainer exec $CONTAINERDIR/pytorch-2.4.0.sif python -m pip install --user ./PARCtorch-0.2.2-py3-none-any.whl
apptainer run --nv $CONTAINERDIR/pytorch-2.4.0.sif python navierStokes.py

echo "Training job completed successfully."
