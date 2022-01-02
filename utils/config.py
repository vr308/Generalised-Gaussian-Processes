
""" Overall configuration """

from pathlib import Path
import torch 
import gpytorch 

TORCH_VERSION = torch.__version__
GPYTORCH_VERSION = gpytorch.__version__

AVAILABLE_GPU = torch.cuda.device_count()
GPU_ACTIVE = bool(AVAILABLE_GPU)
EPSILON = 1e-6
BASE_SEED = 173 

BASE_PATH = Path(__file__).parent.parent
RESULTS_DIR = BASE_PATH / "results"
DATASET_DIR = BASE_PATH / "utils" / "data"
LOG_DIR = BASE_PATH / "logs"
