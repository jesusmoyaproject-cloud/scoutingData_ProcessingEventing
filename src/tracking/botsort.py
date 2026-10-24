"""
Módulo de Seguimiento de Trayectorias (Tracking): BoT-SORT y BallTracker
Integra:
- BallTracker: Supresión de saltos espaciales espurios e interpolación cinemática balística.
- PlayerKinematicTracker: Multi-rate stride (skip adaptativo) con extrapolación por velocidad.
- SportsTracker: Orquestador integral de trayectorias para jugadores y balón.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.utils.geometry import pixel_to_field


class BallTracker:
    """
    Seguimiento y estabilización de la trayectoria del balón.
    - Filtra saltos bruscos irrazonables (falsos positivos lejanos).
    - Interpola la posición mediante extrapolación lineal de velocidad cuando se pierde hasta N frames.
    """

    def __init__(
        self,
        max_history: int = 10,
        max_missing_frames: int = 5,
        max_distance_jump: float = 180.0,
    ):
        self.max_history = max_history
        self.max_missing_frames = max_missing_frames
        self.max_distance_jump = max_distance_jump
        self.history: List[Optional[Tuple[float, float]]] = []
        self.missing_count = 0
        self.last_known_detection: Optional[Dict[str, Any]] = None

    def update(
        self,
        current_detection: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], List[Tuple[float, float]]]:
        """
        Actualiza el estado del balón.
        Retorna (ball_detection_dict, lista_de_puntos_historicos).
        """
        if current_detection is not None:
            curr_x = current_detection["pixel_x"]
            curr_y = current_detection["pixel_y"]

            # Descartar saltos bruscos espurios
            if self.last_known_detection is not None and self.missing_count < 3:
                prev_x = self.last_known_detection["pixel_x"]
                prev_y = self.last_known_detection["pixel_y"]
                dist = np.hypot(curr_x - prev_x, curr_y - prev_y)
                if dist > self.max_distance_jump:
                    current_detection = None

        if current_detection is not None:
            curr_x = current_detection["pixel_x"]
            curr_y = current_detection["pixel_y"]
            self.missing_count = 0
            self.last_known_detection = current_detection.copy()
            self.history.append((curr_x, curr_y))
            if len(self.history) > self.max_history:
                self.history.pop(0)

            return current_detection, [p for p in self.history if p is not None]

        # Manejo de frame sin detección directa -> Intento de interpolación cinemática
        self.missing_count += 1
        if (
            self.missing_count <= self.max_missing_frames
            and self.last_known_detection is not None
            and len(self.history) >= 2
        ):
            valid_pts = [p for p in self.history if p is not None]
            if len(valid_pts) >= 2:
                p2 = valid_pts[-1]
                p1 = valid_pts[-2]
                vx = p2[0] - p1[0]
                vy = p2[1] - p1[1]

                est_x = p2[0] + vx * 0.5
                est_y = p2[1] + vy * 0.5

                interpolated_det = self.last_known_detection.copy()
                interpolated_det["pixel_x"] = round(est_x, 2)
                interpolated_det["pixel_y"] = round(est_y, 2)
                interpolated_det["confidence"] = round(max(0.1, interpolated_det.get("confidence", 0.5) * 0.85), 2)
                interpolated_det["interpolated"] = True

                self.history.append((est_x, est_y))
                if len(self.history) > self.max_history:
                    self.history.pop(0)

                return interpolated_det, [p for p in self.history if p is not None]

        return None, [p for p in self.history if p is not None]


class PlayerKinematicTracker:
    r"""
    Gestiona la cinemática de jugadores para el modo multi-rate stride:
    En frames donde se omite la inferencia pesada de jugadores, interpola
    posiciones con base en la velocidad $(\Delta x, \Delta y)$ del track_id.
    """

    def __init__(self, max_consecutive_extrapolations: int = 3):
        self.max_consecutive_extrapolations = max_consecutive_extrapolations
        self.consecutive_extrapolations = 0
        self.positions: Dict[int, Tuple[float, float]] = {}
        self.velocities: Dict[int, Tuple[float, float]] = {}
        self.last_detections: List[Dict[str, Any]] = []

    def update_from_detections(self, detections: List[Dict[str, Any]]):
        """Actualiza velocidades observadas a partir de una inferencia directa."""
        self.consecutive_extrapolations = 0
        for p in detections:
            p_id = p["player_id"]
            xc = p["pixel_x"]
            yb = p["pixel_y"]

            if p_id in self.positions:
                prev_x, prev_y = self.positions[p_id]
                self.velocities[p_id] = (xc - prev_x, yb - prev_y)
            else:
                self.velocities[p_id] = (0.0, 0.0)

            self.positions[p_id] = (xc, yb)

        self.last_detections = [dict(p) for p in detections]

    def extrapolate_next_frame(self, H: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Extrapola la posición de los jugadores solo en frames omitidos por stride corto (máx 3 frames)."""
        self.consecutive_extrapolations += 1
        if self.consecutive_extrapolations > self.max_consecutive_extrapolations:
            # Corte de toma o ausencia de jugadores: expirar tracklets
            self.last_detections = []
            self.positions = {}
            self.velocities = {}
            return []

        extrapolated: List[Dict[str, Any]] = []
        for p_prev in self.last_detections:
            p = dict(p_prev)
            p_id = p["player_id"]
            vx, vy = self.velocities.get(p_id, (0.0, 0.0))

            p["pixel_x"] = round(p["pixel_x"] + vx, 2)
            p["pixel_y"] = round(p["pixel_y"] + vy, 2)

            if p.get("bbox") is not None:
                bx1, by1, bx2, by2 = p["bbox"]
                p["bbox"] = [
                    round(bx1 + vx, 2),
                    round(by1 + vy, 2),
                    round(bx2 + vx, 2),
                    round(by2 + vy, 2),
                ]

            if H is not None:
                fx, fy = pixel_to_field(p["pixel_x"], p["pixel_y"], H)
                p["pitch_x_m"] = round(fx, 2)
                p["pitch_y_m"] = round(fy, 2)
            else:
                p["pitch_x_m"] = None
                p["pitch_y_m"] = None

            p["is_interpolated"] = True
            extrapolated.append(p)

        self.last_detections = [dict(p) for p in extrapolated]
        return extrapolated
