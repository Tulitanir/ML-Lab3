from pathlib import Path

import torch

MNIST_ROOT = Path("mnist_data")
CKPT_DIR = Path("hopfield_checkpoints")

IMG_SIZE = 28
LATENT_DIM = 64          # ранг факторизации весовой матрицы W = A @ A^T
NUM_STEPS = 4            # число итераций обновления Хопфилда
BETA = 1.0               # коэффициент крутизны для tanh (1.0 = standard)
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
NOISE_FACTOR = 0.4       # sigma гауссовского шума при обучении

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
