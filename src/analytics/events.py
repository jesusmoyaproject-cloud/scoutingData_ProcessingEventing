"""
Módulo de Lógica de Negocio: Detección Cinemática de Eventos Tácticos
Detecta pases, pérdidas/recuperaciones y remates a portería a partir de trayectorias métricas.
"""

from typing import List, Dict, Optional, Any
import numpy as np


class MatchEventDetector:
    """
    Detector de eventos tácticos de fútbol basado en estados cinemáticos.
    - PASS: Cambio de posesión entre jugadores del mismo equipo.
    - TURNOVER: Transición de posesión entre equipos contrarios (recuperación/intercepción).
    - SHOT: Trayectoria acelerada del balón hacia el arco rival dentro del último tercio.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        shot_proximity_m: float = 25.0,
        min_pass_velocity_mps: float = 2.0,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.shot_proximity = shot_proximity_m
        self.min_pass_velocity = min_pass_velocity_mps

        self.last_possessor_id: Optional[int] = None
        self.last_team_id: Optional[int] = None
        self.last_possession_frame: int = 0
        self.last_possession_ts: float = 0.0

        self.events: List[Dict[str, Any]] = []

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        possession_info: Dict[str, Any],
        ball_det: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Evalúa si en el frame actual se disparó un evento táctico.
        Retorna la lista de nuevos eventos detectados en este frame.
        """
        new_events: List[Dict[str, Any]] = []

        curr_possessor = possession_info.get("possessor_id")
        curr_team = possession_info.get("team_id")
        is_confirmed = possession_info.get("is_confirmed", False)

        if not is_confirmed or curr_possessor is None or curr_team is None:
            return new_events

        # Si cambió el jugador con posesión confirmada
        if self.last_possessor_id is not None and curr_possessor != self.last_possessor_id:
            frames_elapsed = frame_idx - self.last_possession_frame

            # Evitar falsos positivos si la alternancia es en frames consecutivos instantáneos (forcejeo)
            if frames_elapsed >= 2:
                if curr_team == self.last_team_id and curr_team in (0, 1):
                    # Pase completado
                    evt = {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "event_type": "PASS",
                        "team_id": curr_team,
                        "from_player_id": self.last_possessor_id,
                        "to_player_id": curr_possessor,
                        "confidence": 0.88,
                    }
                    new_events.append(evt)
                    self.events.append(evt)

                elif curr_team != self.last_team_id and self.last_team_id in (0, 1) and curr_team in (0, 1):
                    # Intercepción o recuperación
                    evt = {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "event_type": "TURNOVER",
                        "lost_by_team": self.last_team_id,
                        "won_by_team": curr_team,
                        "player_id": curr_possessor,
                        "confidence": 0.85,
                    }
                    new_events.append(evt)
                    self.events.append(evt)

        # Detección de Remate (SHOT) si el balón está en zona de definición
        if ball_det and ball_det.get("pitch_x_m") is not None:
            bx = ball_det["pitch_x_m"]
            by = ball_det["pitch_y_m"]

            # Portería izquierda (x ~ 0) o derecha (x ~ 105)
            is_near_left_goal = bx < self.shot_proximity
            is_near_right_goal = bx > (self.pitch_length - self.shot_proximity)

            if is_near_left_goal or is_near_right_goal:
                # Comprobar si el balón está centrado hacia el arco (y ~ 34m ± 10m)
                if abs(by - (self.pitch_width / 2.0)) < 12.0:
                    # Si no hay posesor inmediato pero venía de un jugador atacante
                    if curr_possessor is None and self.last_possessor_id is not None:
                        # Verificar que no hayamos registrado un remate en los últimos 45 frames
                        recent_shots = [
                            e for e in self.events
                            if e["event_type"] == "SHOT" and (frame_idx - e["frame_idx"]) < 45
                        ]
                        if not recent_shots:
                            evt = {
                                "frame_idx": frame_idx,
                                "timestamp_sec": round(timestamp_sec, 2),
                                "event_type": "SHOT",
                                "team_id": self.last_team_id,
                                "shooter_player_id": self.last_possessor_id,
                                "target_goal": "left" if is_near_left_goal else "right",
                                "confidence": 0.78,
                            }
                            new_events.append(evt)
                            self.events.append(evt)

        self.last_possessor_id = curr_possessor
        self.last_team_id = curr_team
        self.last_possession_frame = frame_idx
        self.last_possession_ts = timestamp_sec

        return new_events

    def get_all_events(self) -> List[Dict[str, Any]]:
        return self.events
