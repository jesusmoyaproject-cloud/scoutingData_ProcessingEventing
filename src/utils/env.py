"""
Módulo de Detección Automática de Entorno y Hardware
Soporta detección para Local (laptop / workstation), Google Colab y Kaggle.
"""

import os
import torch
from pathlib import Path


def detect_environment() -> str:
    """
    Detecta automáticamente si la ejecución es en LOCAL, KAGGLE o COLAB.
    Permite sobreescritura manual mediante variable de entorno ENVIRONMENT o APP_ENV.
    """
    forced = os.getenv("ENVIRONMENT") or os.getenv("APP_ENV")
    if forced:
        return forced.upper()

    # Detección Kaggle
    if (
        os.path.exists("/kaggle/working")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_CONTAINER_NAME" in os.environ
    ):
        return "KAGGLE"

    # Detección Google Colab
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "COLAB"

    return "LOCAL"


def is_cuda_available() -> bool:
    """Verifica disponibilidad de aceleración GPU CUDA con PyTorch."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def detect_device(env: str = None) -> str:
    """Selecciona el dispositivo de inferencia por defecto según el entorno y hardware."""
    current_env = env or detect_environment()
    if current_env in ("KAGGLE", "COLAB") and is_cuda_available():
        return "cuda"
    return "cpu"


def detect_headless_mode(env: str = None) -> bool:
    """Determina si debe omitirse la renderización de video y minimapas."""
    env_val = os.getenv("HEADLESS_MODE")
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")

    current_env = env or detect_environment()
    return current_env in ("KAGGLE", "COLAB")
