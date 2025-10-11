# Setup

## Python venv dependencies
I originally created a Python environment that has all the necessary packages (main odd one out is an older version of timm for running the original publicly archived MoCov3 script).

## GPU Specs
To run my Python scripts for pre- and post-training, I used DDP with 2 8-core NVIDIA GeForce RTX 3090 GPUs with 40 GB total memory. \
To create an interactive job with this GPU partition that lasts 12 hours, **run the following terminal command** and wait for your resource allocation
to be granted:

```interact -q gpu -g 2 -f ampere -m 40g -n 8 -t 12:00:00```

**SSH into your GPU node** using ```ssh gpu<<node #>>```
To see your GPU node id, run* ```myq``` *and find the interactive job with your GPU # (e.g. gpu2107).

## Enabling CUDA for PyTorch
To enable CUDA to run PyTorch scripts within an appropriate environment, **please run the following**:

```bash
module purge
unset LD_LIBRARY_PATH
module load cudnn cuda
nvcc --version
python -m venv galaxy
source galaxy/bin/activate
pip install -r requirements.txt
python
```
```python
import torch
torch.cuda.is_available()
```

If the last line returns ```True```, CUDA is properly running. \
If the line returns ```False```, please refer to CCV's GPU documentation for Oscar or contact Brown CCV directly.

# Running Pre- and Post-Training Scripts
Once setup is complete, run the following ```srun``` commands in your terminal.

### Initial pretraining command:
```
srun --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=40G torchrun main_moco.py /tmp/placeholder --hf-dataset matthieulel/galaxy10_decals     --hf-split train --batch-size 64 --workers 8 --multiprocessing-distributed --world-size 1 --rank 0 --seed 42
```
### Initial posttraining command:
```
srun --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=40G torchrun main_lincls.py /tmp/placeholder --hf-dataset matthieulel/galaxy10_decals     --hf-train-split train --hf-val-split test --batch-size 64 --workers 8 --multiprocessing-distributed --world-size 1 --rank 0 --seed 42
```

To see all parameter arguments you can modify to run different experiments, run ```torchrun <file_name> --help```.

# Experiments:
Below is a short table describing my experiments.
```
--arch: [default]; Result: Baseline for comparison
--arch: [default] -> vit_base; Result: Worse
Added rotate arg; --rotate: [default] -> y (Rotation of 30º); Result: Slightly worse
--epochs: 100 -> 200; Result: Slightly better
--momentum @ 200 epochs: [default] -> 0.95; Result: Better
--momentum @ 200 epochs: 0.95 -> 0.99; Result: Best
```
***NOTE:*** Baseline MoCov3 hyperparameters with a batch size of 64 was used as a baseline for comparison.

# Evaluation:
To see the **peak top-1 test accuracy** for a given model, run ```python print_best_acc1.py <<checkpoint_name>>.pth.tar``` in your terminal.

***NOTE:*** Peak top-1 accuracy is defined by the authors of MoCov3 in their original Python script as the highest top-1 accuracy 
srun --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=40G   torchrun main_moco.py /tmp/placeholder --hf-dataset matthieulel/galaxy10_decals --hf-split train --batch-size 64 --workers 8 --epochs 200 --multiprocessing-distributed --world-size 1 --rank 0 --seed 42


### Miscellaneous Notes:
- To see TensorBoard logs of pretraining loss, run tensorboard --logdir runs --port 6006 --bind_all and open the resulting URL in an Oscar OOD Desktop Web Browser.
- I did not use seed 42 for the initial 200 epoch + momentum 95 runs. By and large, however, this should not become too much of an issue for reproducibility.
