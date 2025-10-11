# #!/bin/bash
# #SBATCH --job-name=galaxy_moco_msp
# #SBATCH --partition=gpu
# #SBATCH --output=logs/800/output_pre_%j.log
# #SBATCH --error=logs/800/output_pre_%j.log
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:2
# #SBATCH --time=45:00:00
# #SBATCH --mem=40G

# module avail hpcx-mpi

# source galaxy/bin/activate

# module load cudnn cuda
# # python -c "import torch; print(co)

# MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 torchrun --standalone --npoc_per_node=2 \
#     main_moco.py \
#     --arch vit_base
#     --batch_size 512 \
#     --workers 8 \
#     --epochs 100 \
    