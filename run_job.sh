    #!/bin/bash
    #SBATCH --job-name=mamba_music_generation
    #SBATCH --output=logs/%x_%j.out  # Save output to logs directory with job name and ID
    #SBATCH --error=logs/%x_%j.err   # Save errors to logs directory with job name and ID
    #SBATCH --time=24:00:00          # Maximum runtime
    #SBATCH --partition=compute      # Partition to submit to
    #SBATCH --ntasks=1               # Number of tasks
    #SBATCH --cpus-per-task=4        # Number of CPU cores per task
    #SBATCH --mem=16G                # Memory per node
    #SBATCH --gres=gpu:1             # Number of GPUs (if needed)

    # Activate the virtual environment
    source $(poetry env info --path)/bin/activate

    # Run your Python script
    python src/train_model.py