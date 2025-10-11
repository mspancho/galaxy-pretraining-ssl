source galaxy/bin/activate

srun --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=40G   python main_moco.py /tmp/placeholder --hf-dataset matthieulel/galaxy10_decals --hf-split train --batch-size 64 --workers 8 --epochs 200 --multiprocessing-distributed --world-size 1 --rank 0 

srun --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=40G   python main_lincls.py /tmp/placeholder --hf-dataset matthieulel/galaxy10_decals  --hf-train-split train --hf-val-split test --batch-size 64 --workers 8  --epochs 100 --multiprocessing-distributed --world-size 1 --rank 0 --pretrained checkpoint_0199.pth.tar

mv checkpoint_0199.pth.tar init_checkpoint_0199.pth.tar
mv checkpoint.pth.tar 0199_post_checkpoint.pth.tar
mv model_best.pth.tar 0199_post_model_best.pth.tar

srun --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=40G   python main_moco.py /tmp/placeholder --hf-dataset matthieulel/galaxy10_decals --hf-split train --batch-size 64 --workers 8 --epochs 200 --multiprocessing-distributed --world-size 1 --rank 0 --momentum 0.95
mv checkpoint_0199.pth.tar m_95_checkpoint_0199.pth.tar

# srun --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=40G   torchrun main_lincls.py /tmp/placeholder --hf-dataset matthieulel/galaxy10_decals  --hf-train-split train --hf-val-split test --batch-size 64 --workers 8  --epochs 100 --multiprocessing-distributed --world-size 1 --rank 0 --pretrained m_95_checkpoint_0199.pth.tar
# mv checkpoint.pth.tar m_95_post_checkpoint.pth.tar
# mv model_best.pth.tar m_95_0199_post_model_best.pth.tar