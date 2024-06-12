#!/bin/sh

cd /net/tscratch/people/$USER/mamba-music-generation

# conda activate mamba
echo "Activating poetry shell"
poetry shell
echo "Loading cuda"
module load CUDA/12.1.1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Set the cache directory to tscratch
source /net/tscratch/people/$USER/mamba-music-generation/plgrid_change_cache_dirs.sh

srun python tests/test_env.py