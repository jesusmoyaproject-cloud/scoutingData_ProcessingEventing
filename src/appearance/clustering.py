"""
Módulo de Agrupamiento y Clasificación de Equipos (K-Means en CIELAB)
Clasifica jugadores en Team 0 y Team 1 con consistencia temporal y detección de roles (Porteros, Árbitros).
"""

from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import cv2
import numpy as np

from src.appearance.color_extractor import JerseyColorExtractor, bgr_to_hex, lab_to_bgr


class TeamClassifier:
    """
    Clasificador no supervisado de equipos basado en K-Means y consistencia temporal.
    - Calibración global con K=2 en espacio CIELAB.
    - Atenuación de la luminancia (L*) para evitar confusión por sombras.
    - Ventana móvil (deque) y bloqueo permanente (locking) por track_id.
    - Detección contextual de roles: Portero en áreas de meta y Árbitro en zona neutral.
    """

    def __init__(
        self,
        min_samples_to_cluster: int = 15,
        history_window: int = 30,
        lock_threshold: int = 10,
        outlier_distance_lab: float = 28.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        luminance_weight: float = 0.30,
        min_cluster_separation: float = 20.0,
    ):
        self.min_samples_to_cluster = min_samples_to_cluster
        self.history_window = history_window
        self.lock_threshold = lock_threshold
        self.outlier_distance_lab = outlier_distance_lab
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.luminance_weight = luminance_weight
        self.min_cluster_separation = min_cluster_separation

        self.color_extractor = JerseyColorExtractor()

        self.centroids_lab: Optional[np.ndarray] = None
        self.team_colors_bgr: Dict[int, Tuple[int, int, int]] = {
            0: (0, 0, 255),       # Rojo BGR por defecto
            1: (255, 255, 255),   # Blanco BGR por defecto
            -1: (40, 40, 40),     # Árbitro Negro BGR
        }
        self.team_colors_hex: Dict[int, str] = {
            0: "#FF0000",
            1: "#FFFFFF",
            -1: "#282828",
        }

        self._calibration_samples: List[np.ndarray] = []
        self._is_calibrated: bool = False

        self._track_history: Dict[int, deque] = {}
        self._locked_teams: Dict[int, int] = {}
        self._locked_roles: Dict[int, str] = {}

    def extract_jersey_color(
        self,
        frame_bgr: np.ndarray,
        bbox: List[float],
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
        """Extrae el color de camiseta delegando en JerseyColorExtractor."""
        return self.color_extractor.extract_jersey_color(frame_bgr, bbox)

    def _calibrate_centroids(self):
        """Ejecuta K-Means K=2 sobre las muestras acumuladas en espacio CIELAB."""
        if len(self._calibration_samples) < self.min_samples_to_cluster:
            return

        samples = np.array(self._calibration_samples, dtype=np.float32)

        features = samples.copy()
        features[:, 0] *= float(np.sqrt(self.luminance_weight))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _, labels, centers_feat = cv2.kmeans(
            features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        c0 = np.mean(samples[labels.flatten() == 0], axis=0) if np.any(labels == 0) else centers_feat[0]
        c1 = np.mean(samples[labels.flatten() == 1], axis=0) if np.any(labels == 1) else centers_feat[1]

        self.centroids_lab = np.array([c0, c1], dtype=np.float32)

        bgr0 = lab_to_bgr(c0)
        bgr1 = lab_to_bgr(c1)

        self.team_colors_bgr[0] = bgr0
        self.team_colors_bgr[1] = bgr1
        self.team_colors_hex[0] = bgr_to_hex(bgr0)
        self.team_colors_hex[1] = bgr_to_hex(bgr1)

        # Solo bloquear calibración si ambos centroides tienen suficiente separación inter-equipo
        dist_sep = float(np.linalg.norm(c0 - c1))
        if dist_sep >= self.min_cluster_separation or len(self._calibration_samples) >= 100:
            self._is_calibrated = True

    def classify_player(
        self,
        lab_color: np.ndarray,
        bgr_color: Tuple[int, int, int],
        track_id: Optional[int] = None,
        pitch_x_m: Optional[float] = None,
        pitch_y_m: Optional[float] = None,
        initial_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clasifica al jugador en Team 0 o Team 1 respetando el rol detectado por el modelo."""
        if not self._is_calibrated:
            self._calibration_samples.append(lab_color)
            if len(self._calibration_samples) >= self.min_samples_to_cluster:
                self._calibrate_centroids()

        if track_id is not None and track_id in self._locked_teams:
            locked_team = self._locked_teams[track_id]
            locked_role = self._locked_roles.get(track_id, initial_role or "player")
            color_bgr = self.team_colors_bgr.get(locked_team, bgr_color)
            color_hex = self.team_colors_hex.get(locked_team, bgr_to_hex(bgr_color))
            return {
                "team_id": locked_team,
                "role": locked_role,
                "team_color": {
                    "hex": color_hex,
                    "rgb": [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])],
                    "bgr": list(color_bgr),
                },
                "confidence": 0.98,
                "is_locked": True,
            }

        if self.centroids_lab is None:
            return {
                "team_id": 0,
                "role": initial_role or "player",
                "team_color": {
                    "hex": bgr_to_hex(bgr_color),
                    "rgb": [int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])],
                    "bgr": list(bgr_color),
                },
                "confidence": 0.50,
                "is_locked": False,
            }

        c0 = self.centroids_lab[0]
        c1 = self.centroids_lab[1]

        w_l = self.luminance_weight
        d0 = float(np.sqrt(w_l * (lab_color[0] - c0[0]) ** 2 + (lab_color[1] - c0[1]) ** 2 + (lab_color[2] - c0[2]) ** 2))
        d1 = float(np.sqrt(w_l * (lab_color[0] - c1[0]) ** 2 + (lab_color[1] - c1[1]) ** 2 + (lab_color[2] - c1[2]) ** 2))

        min_d = min(d0, d1)
        raw_team = 0 if d0 < d1 else 1

        tau = 18.0
        e0 = np.exp(-d0 / tau)
        e1 = np.exp(-d1 / tau)
        conf = float(max(e0, e1) / (e0 + e1 + 1e-9))

        # Detección y preservación de roles
        if initial_role == "player":
            role = "player"
            assigned_team = raw_team
        elif initial_role == "goalkeeper":
            role = "goalkeeper"
            if pitch_x_m is not None:
                assigned_team = 0 if pitch_x_m < (self.pitch_length / 2.0) else 1
            else:
                assigned_team = raw_team
        elif initial_role == "referee":
            role = "referee"
            assigned_team = -1
        else:
            # Comportamiento heurístico contextual si initial_role no fue especificado
            role = "player"
            assigned_team = raw_team
            if min_d > self.outlier_distance_lab:
                if pitch_x_m is not None:
                    is_near_goal = (pitch_x_m < 18.0) or (pitch_x_m > (self.pitch_length - 18.0))
                    if is_near_goal:
                        role = "goalkeeper"
                        assigned_team = 0 if pitch_x_m < (self.pitch_length / 2.0) else 1
                    else:
                        role = "referee"
                        assigned_team = -1
                else:
                    role = "referee"
                    assigned_team = -1

        # Consistencia temporal por track_id
        is_locked = False
        if track_id is not None:
            if track_id not in self._track_history:
                self._track_history[track_id] = deque(maxlen=self.history_window)
            self._track_history[track_id].append((assigned_team, conf, role))

            hist = list(self._track_history[track_id])
            if len(hist) >= self.lock_threshold:
                team_counts = {}
                for t_id, _, _ in hist:
                    team_counts[t_id] = team_counts.get(t_id, 0) + 1
                most_common_team, count = max(team_counts.items(), key=lambda x: x[1])

                if count >= self.lock_threshold:
                    self._locked_teams[track_id] = most_common_team
                    role_counts = {}
                    for _, _, r in hist:
                        role_counts[r] = role_counts.get(r, 0) + 1
                    self._locked_roles[track_id] = max(role_counts.items(), key=lambda x: x[1])[0]
                    assigned_team = self._locked_teams[track_id]
                    role = self._locked_roles[track_id]
                    is_locked = True

        color_bgr = self.team_colors_bgr.get(assigned_team, bgr_color)
        color_hex = self.team_colors_hex.get(assigned_team, bgr_to_hex(bgr_color))

        return {
            "team_id": assigned_team,
            "role": role,
            "team_color": {
                "hex": color_hex,
                "rgb": [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])],
                "bgr": list(color_bgr),
            },
            "confidence": round(conf, 3),
            "is_locked": is_locked,
        }

    def annotate_player_detections(
        self,
        frame_bgr: np.ndarray,
        player_detections: List[Dict[str, Any]],
    ):
        """Anota la lista de detecciones in-place con color y equipo asignado respetando el rol detectado."""
        for p in player_detections:
            bbox = p.get("bbox")
            if bbox is None:
                continue

            init_role = p.get("role", "player")
            extracted = self.extract_jersey_color(frame_bgr, bbox)
            if extracted is not None:
                lab_vec, bgr_col = extracted
                clf = self.classify_player(
                    lab_vec,
                    bgr_col,
                    track_id=p.get("player_id"),
                    pitch_x_m=p.get("pitch_x_m"),
                    pitch_y_m=p.get("pitch_y_m"),
                    initial_role=init_role,
                )
                p["team_id"] = clf["team_id"]
                p["role"] = clf["role"]
                p["team_color"] = clf["team_color"]
                p["team_confidence"] = clf["confidence"]
                p["is_team_locked"] = clf["is_locked"]
            else:
                p["team_id"] = 0
                p["role"] = init_role
                p["team_color"] = {
                    "hex": self.team_colors_hex[0],
                    "rgb": [int(self.team_colors_bgr[0][2]), int(self.team_colors_bgr[0][1]), int(self.team_colors_bgr[0][0])],
                    "bgr": list(self.team_colors_bgr[0]),
                }
                p["team_confidence"] = 0.5
                p["is_team_locked"] = False

    @staticmethod
    def get_team_key(team_id: int) -> str:
        """Mapea el identificador numérico de equipo a su clave canónica de metadatos."""
        if team_id == 0:
            return "team_a"
        elif team_id == 1:
            return "team_b"
        elif team_id == -1:
            return "referee"
        return f"team_{team_id}"

    def get_teams_metadata(self) -> Dict[str, Any]:
        """Retorna resumen serializable para el JSON Contract y el archivo de metadatos."""
        bgr0 = self.team_colors_bgr[0]
        bgr1 = self.team_colors_bgr[1]
        bgr_ref = self.team_colors_bgr[-1]

        team_a_meta = {
            "id": 0,
            "name": "Equipo A (Local)",
            "color_hex": self.team_colors_hex[0],
            "color_rgb": [int(bgr0[2]), int(bgr0[1]), int(bgr0[0])],
            "color_bgr": list(bgr0),
        }
        team_b_meta = {
            "id": 1,
            "name": "Equipo B (Visitante)",
            "color_hex": self.team_colors_hex[1],
            "color_rgb": [int(bgr1[2]), int(bgr1[1]), int(bgr1[0])],
            "color_bgr": list(bgr1),
        }
        referee_meta = {
            "id": -1,
            "name": "Árbitro / Referee",
            "color_hex": self.team_colors_hex[-1],
            "color_rgb": [int(bgr_ref[2]), int(bgr_ref[1]), int(bgr_ref[0])],
            "color_bgr": list(bgr_ref),
        }

        return {
            "team_a": team_a_meta,
            "team_b": team_b_meta,
            "referee": referee_meta,
            # Compatibilidad hacia atrás
            "team_0": team_a_meta,
            "team_1": team_b_meta,
            "is_calibrated": self._is_calibrated,
        }


# Alias para compatibilidad con código existente
TeamDetector = TeamClassifier
