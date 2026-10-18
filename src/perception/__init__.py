"""Módulo de Percepción y Visión Artificial."""

from src.perception.detector import YOLODetector, DirectInferenceEngine
from src.perception.homography import (
    compute_homography,
    TemporalHomographyTracker,
    KeypointAuditLogger,
    HomographyDiagnostics,
    HomographyEstimateResult,
    PointResidual,
    KP_MAP,
    FIELD_COORDS,
)

__all__ = [
    "YOLODetector",
    "DirectInferenceEngine",
    "compute_homography",
    "TemporalHomographyTracker",
    "KeypointAuditLogger",
    "HomographyDiagnostics",
    "HomographyEstimateResult",
    "PointResidual",
    "KP_MAP",
    "FIELD_COORDS",
]
