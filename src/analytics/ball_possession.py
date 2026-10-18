"""
Módulo de Lógica de Negocio: Control y Estadísticas de Posesión del Balón
Calcula la posesión individual y colectiva utilizando coordenadas métricas 2D en metros FIFA.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np


class BallPossessionEngine:
    """
    Motor cinemático-geométrico de posesión del balón.
    - Asigna el balón al jugador más cercano dentro de un radio de control (1.8m).
    - Valida el vector de velocidad relativa (evita asignar posesión a jugadores cuando el balón pasa volando a alta velocidad).
    - Aplica cono direccional anatómico según el vector de desplazamiento del jugador.
    - Aplica filtro de histéresis temporal (3 frames consecutivos) para evitar parpadeos.
    - Mantiene estadísticas continuas de porcentaje de posesión por equipo.
    """

    def __init__(
        self,
        possession_radius_m: float = 1.8,
        min_possession_frames: int = 3,
        max_relative_speed_mps: float = 6.0,
        enable_directional_cone: bool = True,
    ):
        self.radius = possession_radius_m
        self.min_frames = min_possession_frames
        self.max_relative_speed = max_relative_speed_mps
        self.enable_directional_cone = enable_directional_cone

        self.candidate_track_id: Optional[int] = None
        self.candidate_team_id: Optional[int] = None
        self.candidate_count: int = 0

        self.current_possessor_id: Optional[int] = None
        self.current_team_id: Optional[int] = None

        self.team_frames: Dict[int, int] = {0: 0, 1: 0}
        self.total_possession_frames: int = 0

        # Historial cinemático para estimación de velocidades
        self._last_ball_pos: Optional[Tuple[float, float]] = None
        self._last_ball_ts: Optional[float] = None
        self._ball_velocity: Tuple[float, float] = (0.0, 0.0)

        self._player_positions: Dict[int, Tuple[float, float]] = {}
        self._player_timestamps: Dict[int, float] = {}
        self._player_velocities: Dict[int, Tuple[float, float]] = {}

    def update(
        self,
        ball_det: Optional[Dict[str, Any]],
        player_dets: List[Dict[str, Any]],
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evalúa el estado de posesión en el frame actual considerando geometría y cinemática.
        Retorna:
            {
                "possessor_id": Optional[int],
                "candidate_id": Optional[int],
                "team_id": Optional[int],
                "distance_m": Optional[float],
                "relative_speed_mps": Optional[float],
                "is_confirmed": bool,
                "team_0_pct": float,
                "team_1_pct": float,
            }
        """
        if (
            ball_det is None
            or ball_det.get("pitch_x_m") is None
            or ball_det.get("pitch_y_m") is None
        ):
            stats = self.get_possession_stats()
            return {
                "possessor_id": self.current_possessor_id,
                "candidate_id": self.candidate_track_id,
                "team_id": self.current_team_id,
                "distance_m": None,
                "relative_speed_mps": None,
                "is_confirmed": self.current_possessor_id is not None,
                **stats,
            }

        bx = float(ball_det["pitch_x_m"])
        by = float(ball_det["pitch_y_m"])

        # ── 1. Estimación de velocidad del balón en m/s ──────────────────────
        ball_has_velocity = False
        if self._last_ball_pos is not None:
            if timestamp_sec is not None and self._last_ball_ts is not None:
                dt_b = timestamp_sec - self._last_ball_ts
            else:
                dt_b = 0.04  # Asumir ~25 FPS por defecto si no se pasa ts

            if dt_b > 0.002:
                v_bx = (bx - self._last_ball_pos[0]) / dt_b
                v_by = (by - self._last_ball_pos[1]) / dt_b
                self._ball_velocity = (v_bx, v_by)
                ball_has_velocity = True

        self._last_ball_pos = (bx, by)
        if timestamp_sec is not None:
            self._last_ball_ts = timestamp_sec

        # ── 2. Estimación de velocidades de jugadores y búsqueda de candidato ─
        closest_player = None
        min_distance = float("inf")
        best_rel_speed: Optional[float] = None

        for p in player_dets:
            # Descartar árbitros
            if p.get("role") == "referee" or p.get("team_id") == -1:
                continue

            px = p.get("pitch_x_m")
            py = p.get("pitch_y_m")
            if px is None or py is None:
                continue

            px = float(px)
            py = float(py)
            p_id = p.get("player_id", -1)

            # Estimación de velocidad del jugador
            v_px, v_py = (0.0, 0.0)
            player_has_velocity = False
            if p_id in self._player_positions:
                if timestamp_sec is not None and p_id in self._player_timestamps:
                    dt_p = timestamp_sec - self._player_timestamps[p_id]
                else:
                    dt_p = 0.04

                if dt_p > 0.002:
                    v_px = (px - self._player_positions[p_id][0]) / dt_p
                    v_py = (py - self._player_positions[p_id][1]) / dt_p
                    self._player_velocities[p_id] = (v_px, v_py)
                    player_has_velocity = True

            self._player_positions[p_id] = (px, py)
            if timestamp_sec is not None:
                self._player_timestamps[p_id] = timestamp_sec

            dist = float(np.hypot(bx - px, by - py))

            # ── 3. Cono Direccional Anatómico ────────────────────────────────
            effective_radius = self.radius
            player_speed = float(np.hypot(v_px, v_py))
            if self.enable_directional_cone and player_has_velocity and player_speed > 0.8:
                dx = bx - px
                dy = by - py
                d_len = float(np.hypot(dx, dy))
                if d_len > 1e-3:
                    # Coseno entre el vector de avance del jugador y el vector hacia el balón
                    cos_theta = (v_px * dx + v_py * dy) / (player_speed * d_len)
                    # Si el balón queda a la espalda del jugador en carrera (cos < -0.2), cono reducido
                    if cos_theta < -0.2:
                        effective_radius = self.radius * 0.55

            if dist > effective_radius:
                continue

            # ── 4. Validación de Velocidad Relativa ───────────────────────────
            # Si el balón vuela a gran velocidad relativa respecto al jugador, no es posesión
            rel_speed = 0.0
            if ball_has_velocity and player_has_velocity:
                rel_v_x = self._ball_velocity[0] - v_px
                rel_v_y = self._ball_velocity[1] - v_py
                rel_speed = float(np.hypot(rel_v_x, rel_v_y))

                if rel_speed > self.max_relative_speed:
                    continue  # El balón está pasando de largo a alta velocidad

            if dist < min_distance:
                min_distance = dist
                closest_player = p
                best_rel_speed = rel_speed

        # ── 5. Actualización de Histéresis Temporal ───────────────────────────
        if closest_player is not None:
            p_id = closest_player.get("player_id")
            t_id = closest_player.get("team_id")

            if p_id == self.candidate_track_id:
                self.candidate_count += 1
            else:
                self.candidate_track_id = p_id
                self.candidate_team_id = t_id
                self.candidate_count = 1

            if self.candidate_count >= self.min_frames:
                self.current_possessor_id = p_id
                self.current_team_id = t_id
                if self.current_team_id in self.team_frames:
                    self.team_frames[self.current_team_id] += 1
                self.total_possession_frames += 1
        else:
            self.candidate_track_id = None
            self.candidate_team_id = None
            self.candidate_count = 0

        stats = self.get_possession_stats()
        return {
            "possessor_id": self.current_possessor_id,
            "candidate_id": self.candidate_track_id,
            "team_id": self.current_team_id,
            "distance_m": round(min_distance, 2) if min_distance != float("inf") else None,
            "relative_speed_mps": round(best_rel_speed, 2) if best_rel_speed is not None else None,
            "is_confirmed": (self.current_possessor_id is not None),
            **stats,
        }

    def get_possession_stats(self) -> Dict[str, float]:
        """Calcula el porcentaje de posesión acumulada por equipo."""
        if self.total_possession_frames == 0:
            return {"team_0_pct": 50.0, "team_1_pct": 50.0}

        t0 = (self.team_frames.get(0, 0) / self.total_possession_frames) * 100.0
        t1 = (self.team_frames.get(1, 0) / self.total_possession_frames) * 100.0
        return {
            "team_0_pct": round(t0, 1),
            "team_1_pct": round(t1, 1),
        }
