"""
Módulo de Procesamiento Batch / Offline:
- Filtro Savitzky-Golay (w=7, n=2) para atenuar jitter sin distorsionar picos de aceleración y Vmax.
- Corrección Geométrica de Cruces basada en colinealidad vectorial de ángulos θ entre trayectorias en conflicto.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy.signal import savgol_filter


class OfflineTrajectoryProcessor:
    """
    Procesador Offline / Batch para trayectorias de jugadores recolectadas durante el clip/partido.
    """

    def __init__(
        self,
        sg_window_length: int = 7,
        sg_polyorder: int = 2,
        crossing_distance_thresh_m: float = 1.5,
        min_track_length: int = 10,
    ):
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder
        self.crossing_distance_thresh_m = crossing_distance_thresh_m
        self.min_track_length = min_track_length

    def smooth_trajectory_savgol(
        self,
        coords: np.ndarray,
    ) -> np.ndarray:
        r"""
        Aplica el filtro Savitzky-Golay a un array de coordenadas N x 2 [[X_0, Y_0], ...].
        Si N < window_length, reduce la ventana adaptativamente a un número impar válido.
        """
        if len(coords) < 3:
            return coords.copy()

        n_samples = len(coords)
        w = self.sg_window_length
        if w >= n_samples:
            w = n_samples if n_samples % 2 != 0 else n_samples - 1
            w = max(3, w)

        p = min(self.sg_polyorder, w - 1)

        try:
            smoothed_x = savgol_filter(coords[:, 0], window_length=w, polyorder=p)
            smoothed_y = savgol_filter(coords[:, 1], window_length=w, polyorder=p)
            return np.column_stack([smoothed_x, smoothed_y])
        except Exception:
            return coords.copy()

    def resolve_crossings(
        self,
        trajectories: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        r"""
        Detecta potenciales intercambios de identidad (ID Switches) cuando dos jugadores
        se acercan a una distancia crítica (<= 1.5m) y evalúa el coseno de ángulo entre
        los vectores de entrada y salida:
        \cos \theta = \frac{\vec{v}_{in} \cdot \vec{v}_{out}}{\|\vec{v}_{in}\| \|\vec{v}_{out}\|}
        """
        track_ids = list(trajectories.keys())
        if len(track_ids) < 2:
            return trajectories

        # Mapeo de frames para consultas temporales rápidas
        # frame_idx -> {track_id: dict_punto}
        frame_map: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for t_id, pts in trajectories.items():
            for p in pts:
                f_idx = p["frame_idx"]
                if f_idx not in frame_map:
                    frame_map[f_idx] = {}
                frame_map[f_idx][t_id] = p

        frames_sorted = sorted(frame_map.keys())
        if not frames_sorted:
            return trajectories

        # Buscar ventanas de cruce entre pares de jugadores
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id_a = track_ids[i]
                id_b = track_ids[j]

                crossing_frames = []
                for f_idx in frames_sorted:
                    if id_a in frame_map[f_idx] and id_b in frame_map[f_idx]:
                        pa = frame_map[f_idx][id_a]
                        pb = frame_map[f_idx][id_b]
                        if pa.get("pitch_x_m") is not None and pb.get("pitch_x_m") is not None:
                            dist = np.hypot(
                                pa["pitch_x_m"] - pb["pitch_x_m"],
                                pa["pitch_y_m"] - pb["pitch_y_m"],
                            )
                            if dist <= self.crossing_distance_thresh_m:
                                crossing_frames.append(f_idx)

                # Si hubo un encuentro cercano durante al menos 1 frame
                if len(crossing_frames) >= 1:
                    f_start = crossing_frames[0]
                    f_end = crossing_frames[-1]

                    # Calcular vectores antes del cruce (in)
                    v_in_a = self._get_velocity_vector(trajectories[id_a], before_frame=f_start)
                    v_in_b = self._get_velocity_vector(trajectories[id_b], before_frame=f_start)

                    # Calcular vectores después del cruce (out)
                    v_out_a = self._get_velocity_vector(trajectories[id_a], after_frame=f_end)
                    v_out_b = self._get_velocity_vector(trajectories[id_b], after_frame=f_end)

                    if (
                        v_in_a is not None
                        and v_in_b is not None
                        and v_out_a is not None
                        and v_out_b is not None
                    ):
                        # Similitud sin cambio: (A->A) + (B->B)
                        sim_straight = self._cosine_sim(v_in_a, v_out_a) + self._cosine_sim(v_in_b, v_out_b)
                        # Similitud con cruce intercambiado: (A->B) + (B->A)
                        sim_swapped = self._cosine_sim(v_in_a, v_out_b) + self._cosine_sim(v_in_b, v_out_a)

                        # Si la colinealidad física es mayor intercambiando
                        if sim_swapped > sim_straight:
                            # Realizar swap de las trayectorias desde f_end en adelante
                            self._swap_tails(trajectories, id_a, id_b, from_frame=f_end + 1 if f_end + 1 in frame_map else f_end)

        return trajectories

    def process_all(
        self,
        trajectories: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Ejecuta el pipeline completo Offline:
        1. Suavizado Savitzky-Golay de las coordenadas de campo métricas (pitch_x_m, pitch_y_m).
        2. Corrección de Cruces mediante análisis angular de vectores cinemáticos.
        """
        # Paso 1: Suavizado Savitzky-Golay
        smoothed_trajectories: Dict[int, List[Dict[str, Any]]] = {}
        for t_id, pts in trajectories.items():
            if len(pts) < 3:
                smoothed_trajectories[t_id] = [dict(p) for p in pts]
                continue

            valid_indices = [
                idx for idx, p in enumerate(pts)
                if p.get("pitch_x_m") is not None and p.get("pitch_y_m") is not None
            ]

            if len(valid_indices) >= 3:
                coords = np.array(
                    [[pts[i]["pitch_x_m"], pts[i]["pitch_y_m"]] for i in valid_indices],
                    dtype=np.float32,
                )
                filtered_coords = self.smooth_trajectory_savgol(coords)

                pts_copy = [dict(p) for p in pts]
                for k, orig_idx in enumerate(valid_indices):
                    pts_copy[orig_idx]["pitch_x_m"] = round(float(filtered_coords[k, 0]), 2)
                    pts_copy[orig_idx]["pitch_y_m"] = round(float(filtered_coords[k, 1]), 2)
                    pts_copy[orig_idx]["is_smoothed_sg"] = True

                smoothed_trajectories[t_id] = pts_copy
            else:
                smoothed_trajectories[t_id] = [dict(p) for p in pts]

        # Paso 2: Corrección geométrica de cruces
        resolved_trajectories = self.resolve_crossings(smoothed_trajectories)
        return resolved_trajectories

    def _get_velocity_vector(
        self,
        pts: List[Dict[str, Any]],
        before_frame: Optional[int] = None,
        after_frame: Optional[int] = None,
        window: int = 5,
    ) -> Optional[np.ndarray]:
        """Extrae el vector director promedio de velocidad en una ventana temporal."""
        if before_frame is not None:
            sub = [p for p in pts if p["frame_idx"] < before_frame and p.get("pitch_x_m") is not None]
            if len(sub) < 2:
                return None
            p_end = sub[-1]
            p_start = sub[-min(len(sub), window)]
            vx = p_end["pitch_x_m"] - p_start["pitch_x_m"]
            vy = p_end["pitch_y_m"] - p_start["pitch_y_m"]
            return np.array([vx, vy], dtype=np.float32)

        if after_frame is not None:
            sub = [p for p in pts if p["frame_idx"] > after_frame and p.get("pitch_x_m") is not None]
            if len(sub) < 2:
                return None
            p_start = sub[0]
            p_end = sub[min(len(sub) - 1, window)]
            vx = p_end["pitch_x_m"] - p_start["pitch_x_m"]
            vy = p_end["pitch_y_m"] - p_start["pitch_y_m"]
            return np.array([vx, vy], dtype=np.float32)

        return None

    def _cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-4 or norm2 < 1e-4:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _swap_tails(
        self,
        trajectories: Dict[int, List[Dict[str, Any]]],
        id_a: int,
        id_b: int,
        from_frame: int,
    ):
        """Intercambia los puntos futuros de dos trayectorias tras una oclusión errónea."""
        tail_a = [p for p in trajectories[id_a] if p["frame_idx"] >= from_frame]
        tail_b = [p for p in trajectories[id_b] if p["frame_idx"] >= from_frame]

        # Reasignar player_id en los metadatos de los puntos
        for p in tail_a:
            p["player_id"] = id_b
            p["crossing_corrected"] = True
        for p in tail_b:
            p["player_id"] = id_a
            p["crossing_corrected"] = True

        trajectories[id_a] = [p for p in trajectories[id_a] if p["frame_idx"] < from_frame] + tail_b
        trajectories[id_b] = [p for p in trajectories[id_b] if p["frame_idx"] < from_frame] + tail_a
