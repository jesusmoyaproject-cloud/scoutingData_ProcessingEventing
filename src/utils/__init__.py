"""Módulo de utilidades transversales para analitica_futbol."""

from src.utils.env import detect_environment, detect_device, detect_headless_mode, is_cuda_available
from src.utils.geometry import (
    pixel_to_field,
    field_to_pixel,
    euclidean_distance,
    bbox_to_center,
    bbox_to_center_bottom,
    compute_pitch_keypoint_coordinates,
    get_pitch_template_specification,
)
from src.utils.video_stream import ThreadedVideoReader, ThreadedOpenCVReader, AsyncVideoWriter
from src.utils.config_loader import load_config, Config

__all__ = [
    "detect_environment",
    "detect_device",
    "detect_headless_mode",
    "is_cuda_available",
    "pixel_to_field",
    "field_to_pixel",
    "euclidean_distance",
    "bbox_to_center",
    "bbox_to_center_bottom",
    "compute_pitch_keypoint_coordinates",
    "get_pitch_template_specification",
    "ThreadedVideoReader",
    "ThreadedOpenCVReader",
    "AsyncVideoWriter",
    "load_config",
    "Config",
]

