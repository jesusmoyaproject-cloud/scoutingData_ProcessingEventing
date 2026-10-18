"""Módulo de Apariencia y Agrupamiento de Equipos."""

from src.appearance.color_extractor import (
    JerseyColorExtractor,
    JerseyDescriptorExtractor,
    bgr_to_hex,
    lab_to_bgr,
)
from src.appearance.clustering import TeamClassifier, TeamDetector
from src.appearance.palette import TEAM_PALETTE_BGR, TEAM_PALETTE_HEX, team_color_bgr, team_color_hex

__all__ = [
    "JerseyColorExtractor",
    "JerseyDescriptorExtractor",
    "bgr_to_hex",
    "lab_to_bgr",
    "TeamClassifier",
    "TeamDetector",
    "TEAM_PALETTE_BGR",
    "TEAM_PALETTE_HEX",
    "team_color_bgr",
    "team_color_hex",
]
