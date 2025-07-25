#!/bin/bash
#SBATCH --job-name=landslides
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task 10
#SBATCH --output=/auto/users/kelsen/slurm/%j.out
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --nodelist=oat4

# Load modules if needed

# Define source and destination directories
SRC_DIR="clpc278:/scratch-ssd/kelsen/landslide-data/UNet_Samples_14Day_GPMv07/"
SRC_DIR2="clpc278:/scratch-ssd/kelsen/landslide-data/Binary_Landslide_Labels_14day/"
SRC_FILE="clpc278:/scratch-ssd/kelsen/landslide-data/District_Labels.tif"
DEST_DIR="/scratch-ssd/kelsen/landslide-data/UNet_Samples_14Day_GPMv07/"
DEST_DIR2="/scratch-ssd/kelsen/landslide-data/Binary_Landslide_Labels_14day/"
DEST_FILE="/scratch-ssd/kelsen/landslide-data"


# Print for logging
echo "Starting rsync at $(date)"

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory does not exist. Creating and syncing files..."
    mkdir -p "$DEST_DIR"

    # Use rsync to copy files from SRC to DEST
    rsync -avh "$SRC_DIR/" "$DEST_DIR/"
    echo "rsync completed."
else
    echo "Destination directory already exists. Skipping rsync."
fi

# Check if destination directory exists
if [ ! -d "$DEST_DIR2" ]; then
    echo "Destination directory does not exist. Creating and syncing files..."
    mkdir -p "$DEST_DIR2"

    # Use rsync to copy files from SRC to DEST
    rsync -avh "$SRC_DIR2/" "$DEST_DIR2/"
    echo "rsync completed."
else
    echo "Destination directory already exists. Skipping rsync."
fi

echo "rsync completed at $(date)"

# Check if the destination file exists
if [ ! -f "$DEST_FILE" ]; then
    echo "Destination file does not exist. Copying via scp..."

    # Create destination directory if it doesn't exist
    mkdir -p "$(dirname "$DEST_FILE")"

    # Copy the file via scp
    scp "$SRC_FILE" "$DEST_FILE"

    if [ $? -eq 0 ]; then
        echo "File copied successfully."
    else
        echo "Failed to copy file."
    fi
else
    echo "Destination file already exists. Skipping scp."
fi

echo "File check completed at $(date)"

source  /users/kelsen/miniconda3/etc/profile.d/conda.sh
conda activate landslides-unet

cd /auto/users/kelsen/code/landslide-forecast-nepal/unet

python run_pipeline.py --epochs 1000 --batch_size 32 --learning-rate 0.001 --loss dice_bce --root_dir /scratch-ssd/kelsen/landslide-data --save_dir /users/kelsen/landslide-results/GPMv07 --val_percent 0.2 --ensemble KMA --ensemble_member 0 --tags testing-fix --exp_type norm --test_year 2023 --norm_type zscore