"""
Módulo de Clasificación de Equipos, Árbitro y Porteros.

Objetivo: DIFERENCIAR Team A, Team B, árbitro y porteros (no reproducir el color real de camiseta).
El color asignado a cada grupo viene de una paleta fija (`palette.py`); el front puede sustituirlo.

Diseño (por etapas):
  1. Descriptor interno por jugador (histograma LAB del torso, césped adaptativo, sin máscara de piel).
  2. Calibración con descriptores a nivel de TRACK de jugadores de campo distintos (K-Means K=2),
     validando la separación entre grupos. Se recentra periódicamente con tracks ya bloqueados.
  3. Asignación por votos ponderados por margen y bloqueo por track. Un track bloqueado ya no
     vuelve a extraer descriptor (camino rápido).
  4. Árbitro por doble evidencia: clase 3 del modelo + descriptor atípico respecto a ambos equipos
     (y fallback solo por color si el modelo no emite la clase 3).
  5. Portero por el lado del campo que ocupa cada equipo (mediana de x en ventana larga), por lo que
     tolera el cambio de lados del segundo tiempo. No requiere descriptor de color.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np

from src.appearance.color_extractor import JerseyColorExtractor, JerseyDescriptorExtractor
from src.appearance.palette import team_color_bgr, team_color_hex

CLASS_GOALKEEPER = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3


@dataclass
class _TrackState:
    """Estado de apariencia/clasificación acumulado por track_id."""
    history_window: int = 30
    desc_sum: Optional[np.ndarray] = None
    n_obs: int = 0                      # observaciones con descriptor válido
    total: int = 0                      # frames totales observados (todas las clases)
    class_counts: Dict[int, int] = field(default_factory=dict)
    votes: Any = None                   # deque[(team, margen)]
    n_votes: int = 0                    # votos emitidos (sin tope de ventana)
    outlier_obs: int = 0                # votos con descriptor atípico para ambos equipos
    locked_team: Optional[int] = None   # 0 / 1 / -1 (árbitro)
    locked_role: Optional[str] = None
    lock_conf: float = 0.0
    last_gk_team: Optional[int] = None

    def __post_init__(self):
        if self.votes is None:
            self.votes = deque(maxlen=self.history_window)


class TeamClassifier:
    """
    Clasificador no supervisado de equipos con consistencia temporal por track.
    Mantiene la API pública previa: `annotate_player_detections`, `get_team_key`,
    `get_teams_metadata`.
    """

    def __init__(
        self,
        min_samples_to_cluster: int = 15,       # compatibilidad (sin uso)
        history_window: int = 30,
        lock_threshold: int = 10,
        outlier_distance_lab: float = 28.0,      # compatibilidad (sin uso)
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        luminance_weight: float = 0.30,          # compatibilidad (sin uso)
        min_cluster_separation: float = 20.0,    # compatibilidad (sin uso)
        min_tracks_to_calibrate: int = 14,
        min_obs_per_track: int = 10,
        min_separation_score: float = 1.2,
        lock_ratio: float = 0.75,
        referee_threshold: float = 0.6,
        referee_color_only_ratio: float = 0.8,
        referee_min_obs: int = 15,
        outlier_sigma: float = 3.0,
        min_descriptor_conf: float = 0.5,
        min_bbox_height_px: int = 32,
        border_margin_px: int = 3,
        side_window_frames: int = 6000,
        side_min_samples: int = 60,
        side_min_gap_m: float = 3.0,
        refit_interval_frames: int = 1800,
        pool_size: int = 80,
    ):
        self.history_window = history_window
        self.lock_threshold = lock_threshold
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        self.min_tracks_to_calibrate = min_tracks_to_calibrate
        self.min_obs_per_track = min_obs_per_track
        self.min_separation_score = min_separation_score
        self.lock_ratio = lock_ratio
        self.referee_threshold = referee_threshold
        self.referee_color_only_ratio = referee_color_only_ratio
        self.referee_min_obs = referee_min_obs
        self.outlier_sigma = outlier_sigma
        self.min_descriptor_conf = min_descriptor_conf
        self.min_bbox_height_px = min_bbox_height_px
        self.border_margin_px = border_margin_px
        self.side_min_samples = side_min_samples
        self.side_min_gap_m = side_min_gap_m
        self.refit_interval_frames = refit_interval_frames

        self.extractor = JerseyDescriptorExtractor()
        # Extractor legado de color (se conserva por compatibilidad de API)
        self.color_extractor = JerseyColorExtractor()

        self._tracks: Dict[int, _TrackState] = {}
        self._centroids: Optional[np.ndarray] = None
        self._intra_mean = np.zeros(2, dtype=np.float32)
        self._intra_std = np.zeros(2, dtype=np.float32)
        self._outlier_thr = np.array([0.30, 0.30], dtype=np.float32)
        self._separation_score = 0.0
        self._is_calibrated = False
        self._tracks_used = 0
        self._refit_count = 0
        self._last_refit_frame = 0
        self._referee_seen = False
        self._pool: deque = deque(maxlen=pool_size)

        self._frame_idx = 0
        self._team_x: Dict[int, deque] = {
            0: deque(maxlen=side_window_frames),
            1: deque(maxlen=side_window_frames),
        }
        self._side_cache: Tuple[int, Optional[float], Optional[float]] = (-10 ** 9, None, None)

    # ------------------------------------------------------------------ API pública
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

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

    def annotate_player_detections(
        self,
        frame_bgr: np.ndarray,
        player_detections: List[Dict[str, Any]],
        frame_idx: Optional[int] = None,
    ):
        """Anota in-place `team_id`, `role`, `team_confidence` e `is_team_locked` en cada detección."""
        self._frame_idx = frame_idx if frame_idx is not None else self._frame_idx + 1

        field_x: Dict[int, List[float]] = {0: [], 1: []}
        for p in player_detections:
            if p.get("bbox") is None:
                self._write(p, 0, p.get("role", "player"), 0.0, False)
                continue
            self._annotate_one(frame_bgr, p, field_x)

        self._update_side_samples(field_x)
        if not self._is_calibrated:
            self._try_calibrate()
        else:
            self._maybe_refit()

    def get_teams_metadata(self) -> Dict[str, Any]:
        """Resumen serializable para el metadata: team_a / team_b / referee con su color ASIGNADO."""

        def meta(tid: int, name: str) -> Dict[str, Any]:
            bgr = team_color_bgr(tid)
            return {
                "id": tid,
                "name": name,
                "color_hex": team_color_hex(tid),
                "color_rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],
                "color_bgr": [int(bgr[0]), int(bgr[1]), int(bgr[2])],
                "color_is_assigned": True,
            }

        team_a_meta = meta(0, "Equipo A")
        team_b_meta = meta(1, "Equipo B")
        referee_meta = meta(-1, "Árbitro / Referee")
        return {
            "team_a": team_a_meta,
            "team_b": team_b_meta,
            "referee": referee_meta,
            # Compatibilidad hacia atrás
            "team_0": team_a_meta,
            "team_1": team_b_meta,
            "is_calibrated": self._is_calibrated,
        }

    def get_classification_report(self) -> Dict[str, Any]:
        """Métricas de calidad de la clasificación para el metadata."""
        locked = {"team_a": 0, "team_b": 0, "referee": 0}
        for st in self._tracks.values():
            if st.locked_role == "referee":
                locked["referee"] += 1
            elif st.locked_team == 0:
                locked["team_a"] += 1
            elif st.locked_team == 1:
                locked["team_b"] += 1
        return {
            "calibrated": self._is_calibrated,
            "separation_score": round(float(self._separation_score), 3),
            "tracks_used_for_calibration": int(self._tracks_used),
            "referee_detected": bool(self._referee_seen),
            "refit_count": int(self._refit_count),
            "locked_tracks": locked,
        }

    # ------------------------------------------------------------------ anotación por detección
    def _get_state(self, pid: Optional[int]) -> _TrackState:
        if pid is None:
            return _TrackState(history_window=self.history_window)
        st = self._tracks.get(pid)
        if st is None:
            st = _TrackState(history_window=self.history_window)
            self._tracks[pid] = st
        return st

    @staticmethod
    def _write(p: Dict[str, Any], team: int, role: str, conf: float, locked: bool):
        p["team_id"] = team
        p["role"] = role
        p["team_confidence"] = round(float(conf), 3)
        p["is_team_locked"] = locked

    def _class_ratio(self, st: _TrackState, class_id: int) -> float:
        return st.class_counts.get(class_id, 0) / max(st.total, 1)

    def _annotate_one(self, frame_bgr: np.ndarray, p: Dict[str, Any], field_x: Dict[int, List[float]]):
        class_id = p.get("class_id")
        if class_id is None:
            class_id = {"goalkeeper": CLASS_GOALKEEPER, "referee": CLASS_REFEREE}.get(p.get("role", "player"), CLASS_PLAYER)

        st = self._get_state(p.get("player_id"))
        st.total += 1
        st.class_counts[class_id] = st.class_counts.get(class_id, 0) + 1

        # 1) Árbitro ya bloqueado: camino rápido
        if st.locked_role == "referee":
            self._write(p, -1, "referee", st.lock_conf, True)
            return

        # 2) Portero (clase mayoritaria 1): se asigna por lado del campo, sin extraer color
        if class_id == CLASS_GOALKEEPER or self._class_ratio(st, CLASS_GOALKEEPER) >= 0.5:
            team, conf = self._goalkeeper_team(p.get("pitch_x_m"), st)
            st.last_gk_team = team
            self._write(p, team, "goalkeeper", conf, False)
            return

        # 3) Equipo ya bloqueado y sistema calibrado: camino rápido (sin extraer descriptor)
        if self._is_calibrated and st.locked_team in (0, 1):
            self._write(p, st.locked_team, "player", st.lock_conf, True)
            self._accumulate_x(field_x, st.locked_team, p)
            return

        # 4) Extracción del descriptor solo si el track aún no está bloqueado
        desc = None
        if self._valid_for_descriptor(p, frame_bgr.shape):
            self.extractor.update_field_model(frame_bgr, self._frame_idx)
            desc = self.extractor.extract(frame_bgr, p["bbox"])
        if desc is not None:
            st.desc_sum = desc.copy() if st.desc_sum is None else st.desc_sum + desc
            st.n_obs += 1
            if self._is_calibrated:
                self._vote(st)
                self._maybe_lock(st)

        # 5) Salida final o provisional
        team, role, conf, locked = self._current_assignment(st)
        self._write(p, team, role, conf, locked)
        if locked and team in (0, 1):
            self._accumulate_x(field_x, team, p)

    def _valid_for_descriptor(self, p: Dict[str, Any], frame_shape) -> bool:
        if float(p.get("confidence", 1.0)) < self.min_descriptor_conf:
            return False
        x1, y1, x2, y2 = p["bbox"]
        if (y2 - y1) < self.min_bbox_height_px:
            return False
        h, w = frame_shape[:2]
        m = self.border_margin_px
        if x1 < m or y1 < m or x2 > (w - m) or y2 > (h - m):
            return False
        return True

    # ------------------------------------------------------------------ votos y bloqueo
    def _track_mean(self, st: _TrackState) -> np.ndarray:
        return st.desc_sum / float(max(st.n_obs, 1))

    def _vote(self, st: _TrackState):
        mean = self._track_mean(st)
        d = np.linalg.norm(self._centroids - mean, axis=1)
        team = int(d[1] < d[0])
        margin = float(abs(d[0] - d[1]) / (d[0] + d[1] + 1e-9))
        st.votes.append((team, margin))
        st.n_votes += 1
        if st.n_obs >= 5 and float(d[team]) > float(self._outlier_thr[team]):
            st.outlier_obs += 1

    def _maybe_lock(self, st: _TrackState):
        if len(st.votes) < self.lock_threshold:
            return

        class3 = self._class_ratio(st, CLASS_REFEREE)
        outlier_ratio = st.outlier_obs / float(max(st.n_votes, 1))
        ref_score = 0.6 * class3 + 0.4 * outlier_ratio
        color_only = (st.n_obs >= self.referee_min_obs) and (outlier_ratio >= self.referee_color_only_ratio)
        if ref_score >= self.referee_threshold or color_only:
            st.locked_team = -1
            st.locked_role = "referee"
            st.lock_conf = round(min(0.99, max(ref_score, outlier_ratio)), 3)
            self._referee_seen = True
            return

        weights = [0.0, 0.0]
        for t, m in st.votes:
            weights[t] += m + 1e-3
        total_w = weights[0] + weights[1]
        share = max(weights) / total_w if total_w > 0 else 0.0
        if share >= self.lock_ratio and class3 < 0.5:
            team = int(weights[1] > weights[0])
            st.locked_team = team
            st.locked_role = "player"
            st.lock_conf = round(share, 3)
            self._pool.append((self._track_mean(st).copy(), team))

    def _current_assignment(self, st: _TrackState) -> Tuple[int, str, float, bool]:
        if st.locked_role == "referee":
            return -1, "referee", st.lock_conf, True
        if self._is_calibrated and st.locked_team in (0, 1):
            return st.locked_team, "player", st.lock_conf, True

        provisional_ref = st.total >= 3 and self._class_ratio(st, CLASS_REFEREE) >= 0.5
        if provisional_ref:
            return -1, "referee", 0.5, False
        if not self._is_calibrated or not st.votes:
            return 0, "player", 0.0, False

        weights = [0.0, 0.0]
        for t, m in st.votes:
            weights[t] += m + 1e-3
        total_w = weights[0] + weights[1]
        share = max(weights) / total_w if total_w > 0 else 0.0
        return int(weights[1] > weights[0]), "player", share, False

    # ------------------------------------------------------------------ calibración
    def _try_calibrate(self):
        eligible: List[Tuple[int, _TrackState]] = []
        for pid, st in self._tracks.items():
            if st.n_obs < self.min_obs_per_track or st.total == 0 or st.locked_role is not None:
                continue
            if self._class_ratio(st, CLASS_PLAYER) < 0.8:
                continue
            eligible.append((pid, st))

        if len(eligible) < self.min_tracks_to_calibrate:
            return

        pids = np.array([pid for pid, _ in eligible])
        data = np.stack([self._track_mean(st) for _, st in eligible]).astype(np.float32)

        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        _, labels, centers = cv2.kmeans(data, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

        # Recorte de atípicos (árbitro/suplentes dentro del pool) y reajuste
        keep = np.ones(len(data), dtype=bool)
        dist = np.linalg.norm(data - centers[labels], axis=1)
        thr = float(dist.mean() + 2.5 * dist.std())
        trimmed = dist <= thr
        if (not trimmed.all()) and int(trimmed.sum()) >= 8:
            keep = trimmed
            _, _, centers = cv2.kmeans(data[keep], 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
            labels = np.argmin(np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2), axis=1)

        counts = [int(((labels == k) & keep).sum()) for k in (0, 1)]
        if min(counts) < 3:
            return

        c = np.stack([data[(labels == k) & keep].mean(axis=0) for k in (0, 1)])
        intra_mean = np.zeros(2, dtype=np.float32)
        intra_std = np.zeros(2, dtype=np.float32)
        for k in (0, 1):
            dk = np.linalg.norm(data[(labels == k) & keep] - c[k], axis=1)
            intra_mean[k] = float(dk.mean())
            intra_std[k] = float(dk.std())
        sep = float(np.linalg.norm(c[0] - c[1]) / (intra_mean[0] + intra_mean[1] + 1e-6))

        if sep < self.min_separation_score and len(eligible) < 2 * self.min_tracks_to_calibrate:
            return  # separación insuficiente: seguir acumulando tracks

        # Numeración determinista: Team A (0) = cluster que contiene el track_id más bajo conservado
        first_label = int(labels[keep][int(np.argmin(pids[keep]))])
        order = [first_label, 1 - first_label]
        self._centroids = c[order].astype(np.float32)
        self._intra_mean = intra_mean[order]
        self._intra_std = intra_std[order]
        self._update_outlier_thresholds()
        self._separation_score = sep
        self._tracks_used = int(keep.sum())
        self._is_calibrated = True
        self._last_refit_frame = self._frame_idx

        # Sembrar votos con la evidencia de calibración para permitir bloqueo inmediato
        for i, (pid, st) in enumerate(eligible):
            d = np.linalg.norm(self._centroids - data[i], axis=1)
            team = int(d[1] < d[0])
            margin = float(abs(d[0] - d[1]) / (d[0] + d[1] + 1e-9))
            st.votes.extend([(team, margin)] * self.lock_threshold)
            st.n_votes += self.lock_threshold
            if not keep[i]:
                st.outlier_obs += self.lock_threshold

    def _update_outlier_thresholds(self):
        for k in (0, 1):
            self._outlier_thr[k] = max(
                float(self._intra_mean[k] + self.outlier_sigma * self._intra_std[k]), 0.30
            )

    def _maybe_refit(self):
        """Recentra los centroides con los tracks bloqueados más recientes (sigue cambios de iluminación)."""
        if (self._frame_idx - self._last_refit_frame) < self.refit_interval_frames:
            return
        by_team: Dict[int, List[np.ndarray]] = {0: [], 1: []}
        for desc, team in self._pool:
            by_team[team].append(desc)
        if len(by_team[0]) < 5 or len(by_team[1]) < 5:
            return

        for k in (0, 1):
            pool_mean = np.mean(by_team[k], axis=0)
            self._centroids[k] = (0.5 * self._centroids[k] + 0.5 * pool_mean).astype(np.float32)
            dk = np.linalg.norm(np.stack(by_team[k]) - self._centroids[k], axis=1)
            self._intra_mean[k] = float(dk.mean())
            self._intra_std[k] = float(dk.std())
        self._update_outlier_thresholds()
        self._separation_score = float(
            np.linalg.norm(self._centroids[0] - self._centroids[1]) / (self._intra_mean[0] + self._intra_mean[1] + 1e-6)
        )
        self._last_refit_frame = self._frame_idx
        self._refit_count += 1

    # ------------------------------------------------------------------ portero por lado del campo
    @staticmethod
    def _accumulate_x(field_x: Dict[int, List[float]], team: int, p: Dict[str, Any]):
        x = p.get("pitch_x_m")
        if x is not None:
            field_x[team].append(float(x))

    def _update_side_samples(self, field_x: Dict[int, List[float]]):
        for team in (0, 1):
            xs = field_x[team]
            if len(xs) >= 3:
                self._team_x[team].append(float(np.mean(xs)))

    def _team_medians(self) -> Tuple[Optional[float], Optional[float]]:
        last_frame, xa, xb = self._side_cache
        if (self._frame_idx - last_frame) < 30 and last_frame >= 0:
            return xa, xb
        if len(self._team_x[0]) >= self.side_min_samples and len(self._team_x[1]) >= self.side_min_samples:
            xa = float(np.median(np.fromiter(self._team_x[0], dtype=np.float32)))
            xb = float(np.median(np.fromiter(self._team_x[1], dtype=np.float32)))
        else:
            xa, xb = None, None
        self._side_cache = (self._frame_idx, xa, xb)
        return xa, xb

    def _goalkeeper_team(self, pitch_x: Optional[float], st: _TrackState) -> Tuple[int, float]:
        """Asigna el portero al equipo que defiende el lado donde está (tolera cambio de lados)."""
        if pitch_x is None:
            return (st.last_gk_team if st.last_gk_team is not None else 0), 0.2

        half = self.pitch_length / 2.0
        xa, xb = self._team_medians()
        if xa is None or xb is None:
            return (0 if pitch_x < half else 1), 0.3  # sin datos suficientes: regla provisional

        gap = abs(xa - xb)
        if gap >= self.side_min_gap_m:
            team_a_defends_left = xa < xb
            gk_left = pitch_x < half
            team = 0 if (gk_left == team_a_defends_left) else 1
            conf = min(0.95, 0.5 + gap / (self.pitch_length * 0.5))
            return team, conf

        # Medianas casi iguales: usar el equipo cuyo centro está más cerca del portero
        team = 0 if abs(pitch_x - xa) <= abs(pitch_x - xb) else 1
        return team, 0.4

    # ------------------------------------------------------------------ Métodos de compatibilidad
    @property
    def centroids_lab(self) -> Optional[np.ndarray]:
        """Propiedad de compatibilidad con tests unitarios legados."""
        return self._centroids

    @property
    def team_colors_bgr(self) -> Dict[int, Tuple[int, int, int]]:
        from src.appearance.palette import TEAM_PALETTE_BGR
        return TEAM_PALETTE_BGR

    @property
    def team_colors_hex(self) -> Dict[int, str]:
        from src.appearance.palette import TEAM_PALETTE_HEX
        return TEAM_PALETTE_HEX

    def classify_player(
        self,
        lab_color: np.ndarray,
        bgr_color: Tuple[int, int, int],
        track_id: Optional[int] = None,
        pitch_x_m: Optional[float] = None,
        pitch_y_m: Optional[float] = None,
        initial_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Método de compatibilidad para clasificar un jugador individual usando valores de color directos.
        """
        from src.appearance.color_extractor import bgr_to_hex

        # Registrar muestra para calibración si aún no está calibrado
        if not self._is_calibrated and track_id is not None:
            st = self._get_state(track_id)
            norm_desc = lab_color / (np.linalg.norm(lab_color) + 1e-6)
            st.desc_sum = norm_desc if st.desc_sum is None else st.desc_sum + norm_desc
            st.n_obs += 1
            st.total += 1
            st.class_counts[CLASS_PLAYER] = st.class_counts.get(CLASS_PLAYER, 0) + 1

            if len(self._tracks) >= 2 and all(s.n_obs >= 2 for s in self._tracks.values()):
                self._centroids = np.stack([
                    self._track_mean(self._tracks[list(self._tracks.keys())[0]]),
                    self._track_mean(self._tracks[list(self._tracks.keys())[1]])
                ])
                self._is_calibrated = True

        st = self._get_state(track_id)
        if track_id is not None:
            st.total += 1

        # Si ya está bloqueado
        if st.locked_team is not None:
            role = st.locked_role or initial_role or "player"
            col_bgr = team_color_bgr(st.locked_team)
            return {
                "team_id": st.locked_team,
                "role": role,
                "team_color": {
                    "hex": team_color_hex(st.locked_team),
                    "rgb": [int(col_bgr[2]), int(col_bgr[1]), int(col_bgr[0])],
                    "bgr": list(col_bgr),
                },
                "confidence": 0.98,
                "is_locked": True,
            }

        # Determinación de rol y equipo
        norm_desc = lab_color / (np.linalg.norm(lab_color) + 1e-6)
        if self._centroids is not None:
            d0 = float(np.linalg.norm(norm_desc - self._centroids[0]))
            d1 = float(np.linalg.norm(norm_desc - self._centroids[1]))
            raw_team = 0 if d0 < d1 else 1
            min_d = min(d0, d1)
        else:
            d0, d1 = 0.5, 0.5
            raw_team = 0
            min_d = 0.0

        if initial_role == "player":
            role = "player"
            assigned_team = raw_team
        elif initial_role == "goalkeeper":
            role = "goalkeeper"
            assigned_team = 0 if (pitch_x_m is not None and pitch_x_m < (self.pitch_length / 2.0)) else raw_team
        elif initial_role == "referee":
            role = "referee"
            assigned_team = -1
        else:
            role = "player"
            assigned_team = raw_team
            if min_d > 0.4:
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

        # Votos y bloqueo
        if track_id is not None:
            margin = float(abs(d0 - d1) / (d0 + d1 + 1e-9))
            st.votes.append((assigned_team, margin))
            st.n_votes += 1

            if len(st.votes) >= self.lock_threshold:
                team_counts = {}
                for tid, _ in st.votes:
                    team_counts[tid] = team_counts.get(tid, 0) + 1
                most_common, cnt = max(team_counts.items(), key=lambda x: x[1])
                if cnt >= self.lock_threshold:
                    st.locked_team = most_common
                    st.locked_role = role
                    st.lock_conf = 0.98

        is_locked = (st.locked_team is not None)
        col_bgr = team_color_bgr(assigned_team)
        return {
            "team_id": assigned_team,
            "role": role,
            "team_color": {
                "hex": team_color_hex(assigned_team),
                "rgb": [int(col_bgr[2]), int(col_bgr[1]), int(col_bgr[0])],
                "bgr": list(col_bgr),
            },
            "confidence": 0.95 if is_locked else 0.75,
            "is_locked": is_locked,
        }


# Alias para compatibilidad con código existente
TeamDetector = TeamClassifier
