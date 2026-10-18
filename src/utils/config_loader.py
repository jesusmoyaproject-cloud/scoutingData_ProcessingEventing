"""
Módulo de Carga y Fusión de Configuraciones YAML
Combina base.yaml con el archivo de inferencia específico (local o cloud).
"""

import os
from pathlib import Path
from typing import Dict, Any, Union
import yaml

from src.utils.env import detect_environment


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Combina recursivamente dos diccionarios, sobrescribiendo base con override."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config(dict):
    """Diccionario que permite acceso por atributo (config.section.param)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, Config):
                self[k] = Config(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Configuracion no tiene el atributo '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value


def load_config(
    inference_config_path: Union[str, Path] = None,
    base_config_path: Union[str, Path] = None,
    project_root: Union[str, Path] = None,
) -> Config:
    """
    Carga y fusiona base.yaml con la configuración de inferencia adecuada.
    Si no se especifica inference_config_path, se auto-detecta por entorno (LOCAL vs KAGGLE/COLAB).
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent
    base_path = Path(base_config_path) if base_config_path else root / "configs" / "base.yaml"

    if not base_path.exists():
        raise FileNotFoundError(f"Archivo base de configuración no encontrado: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        base_data = yaml.safe_load(f) or {}

    # Auto-selección de inferencia si no se especifica
    if not inference_config_path:
        env = detect_environment()
        inf_name = "inference_cloud.yaml" if env in ("KAGGLE", "COLAB") else "inference_local.yaml"
        target_path = root / "configs" / inf_name
    else:
        target_path = Path(inference_config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Archivo de configuración de inferencia no encontrado: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        target_data = yaml.safe_load(f) or {}

    merged = deep_merge(base_data, target_data)
    merged["_project_root"] = str(root)
    merged["_config_path"] = str(target_path)

    return Config(merged)
