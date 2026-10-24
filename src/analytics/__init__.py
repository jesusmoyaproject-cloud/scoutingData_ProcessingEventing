"""Módulo de Analítica Deportiva y Detección de Eventos."""

from src.analytics.ball_possession import BallPossessionEngine
from src.analytics.events import MatchEventDetector
from src.analytics.offline_processor import OfflineTrajectoryProcessor

__all__ = [
    "BallPossessionEngine",
    "MatchEventDetector",
    "OfflineTrajectoryProcessor",
]
