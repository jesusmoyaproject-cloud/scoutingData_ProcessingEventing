"""Módulo de Analítica Deportiva y Detección de Eventos."""

from src.analytics.ball_possession import BallPossessionEngine
from src.analytics.events import MatchEventDetector

__all__ = [
    "BallPossessionEngine",
    "MatchEventDetector",
]
