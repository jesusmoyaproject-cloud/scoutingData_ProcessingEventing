"""
BoT-SORT Evolucionado (UKF + DIoU Tracker).
Conserva la arquitectura de ciclo de vida de BoT-SORT (asociación en dos etapas de alta/baja
confianza y buffer temporal de 45 frames) e inyecta:
- UKF con modelo CTRV para maniobras no lineales de futbolistas.
- DIoU para mitigar pérdida de tracks en sprints o aceleraciones repentinas.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.tracking.ukf import UnscentedKalmanFilter2D
from src.tracking.matching import bbox_diou_matrix, linear_assignment


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Representa una trayectoria individual gestionada por UKF."""

    _count = 0

    def __init__(self, bbox: List[float], score: float, class_id: int):
        STrack._count += 1
        self.track_id = STrack._count
        self.bbox = list(bbox)  # [x1, y1, x2, y2]
        self.score = float(score)
        self.class_id = int(class_id)
        self.state = TrackState.New
        self.is_activated = False

        self.ukf = UnscentedKalmanFilter2D()
        self.ukf.init_from_bbox(self.bbox)

        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0
        self.history: List[Tuple[float, float]] = []

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    def predict(self):
        """Predice el nuevo estado usando UKF."""
        self.ukf.predict()
        self.bbox = list(self.ukf.to_bbox())

    def update(self, new_track: "STrack", frame_id: int):
        """Corrige el estado con una nueva detección observada."""
        self.frame_id = frame_id
        self.time_since_update = 0
        self.score = new_track.score
        self.class_id = new_track.class_id

        self.ukf.update(new_track.bbox)
        self.bbox = list(self.ukf.to_bbox())
        self.state = TrackState.Tracked
        self.is_activated = True

        xc = (self.bbox[0] + self.bbox[2]) / 2.0
        yc = self.bbox[3]  # base del pie
        self.history.append((xc, yc))

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class UKFBoTSORTTracker:
    """
    Tracker híbrido BoT-SORT con UKF (Unscented Kalman Filter) y DIoU matching.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.20,
        new_track_thresh: float = 0.55,
        track_buffer: int = 45,
        match_thresh: float = 0.80,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh  # umbral de costo DIoU (1 - DIoU)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recibe detecciones [{bbox: [x1, y1, x2, y2], confidence: float, class_id: int}, ...]
        y retorna la lista de jugadores trackeados con track_id asignado.
        """
        self.frame_id += 1

        # 1. Separar detecciones en Alta Confianza y Baja Confianza
        high_dets: List[STrack] = []
        low_dets: List[STrack] = []

        for d in detections:
            bbox = d["bbox"]
            conf = float(d.get("confidence", 0.0))
            cls_id = int(d.get("class_id", 2))
            strack = STrack(bbox, conf, cls_id)

            if conf >= self.track_high_thresh:
                high_dets.append(strack)
            elif conf >= self.track_low_thresh:
                low_dets.append(strack)

        # 2. Paso de Predicción UKF para todos los tracks activos y perdidos
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()

        # 3. Primera Ronda de Asociación: Detecciones de Alta Confianza vs (Tracked + Lost)
        pool_tracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked] + self.lost_stracks
        cost_matrix_1 = self._compute_cost_matrix(pool_tracks, high_dets)

        matches_1, u_tracks_idx, u_high_idx = linear_assignment(cost_matrix_1, thresh=self.match_thresh)

        for t_idx, d_idx in matches_1:
            pool_tracks[t_idx].update(high_dets[d_idx], self.frame_id)

        # 4. Segunda Ronda de Asociación: Detecciones de Baja Confianza vs Tracks no emparejados
        unmatched_tracks = [pool_tracks[i] for i in u_tracks_idx if pool_tracks[i].state == TrackState.Tracked]
        cost_matrix_2 = self._compute_cost_matrix(unmatched_tracks, low_dets)

        matches_2, u_tracks_2_idx, _ = linear_assignment(cost_matrix_2, thresh=self.match_thresh)

        for t_idx, d_idx in matches_2:
            unmatched_tracks[t_idx].update(low_dets[d_idx], self.frame_id)

        # Tracks que siguen sin emparejar se marcan como perdidos
        for idx in u_tracks_2_idx:
            track = unmatched_tracks[idx]
            if track.state != TrackState.Lost:
                track.mark_lost()

        # 5. Inicializar Nuevos Tracks desde Detecciones de Alta Confianza no asignadas
        for d_idx in u_high_idx:
            track = high_dets[d_idx]
            if track.score >= self.new_track_thresh:
                track.frame_id = self.frame_id
                track.start_frame = self.frame_id
                track.state = TrackState.Tracked
                track.is_activated = True
                self.tracked_stracks.append(track)

        # 6. Mantenimiento del Buffer Temporal (remover tras track_buffer frames perdidos)
        new_tracked = []
        new_lost = []

        for track in self.tracked_stracks:
            if track.state == TrackState.Tracked:
                new_tracked.append(track)
            elif track.state == TrackState.Lost:
                new_lost.append(track)

        for track in self.lost_stracks:
            if track.state == TrackState.Tracked:
                new_tracked.append(track)
            else:
                track.time_since_update += 1
                if track.time_since_update > self.track_buffer:
                    track.mark_removed()
                    self.removed_stracks.append(track)
                else:
                    new_lost.append(track)

        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost

        # 7. Formatear salida con player_id asignado
        results: List[Dict[str, Any]] = []
        for t in self.tracked_stracks:
            x1, y1, x2, y2 = t.bbox
            xc = (x1 + x2) / 2.0
            yb = y2
            results.append({
                "player_id": t.track_id,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "pixel_x": round(xc, 2),
                "pixel_y": round(yb, 2),
                "confidence": round(t.score, 2),
                "class_id": t.class_id,
                "role": "goalkeeper" if t.class_id == 1 else "player",
                "is_interpolated": False,
            })

        return results

    def _compute_cost_matrix(self, tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        track_boxes = np.array([t.bbox for t in tracks], dtype=np.float32)
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)

        return bbox_diou_matrix(track_boxes, det_boxes)
