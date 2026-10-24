"""
analitica_futbol: Orquestador Principal (End-to-End) - Modo Prueba (Restringido a 3000 frames)
Pipeline unificado de alto rendimiento para visión artificial, tracking cinemático,
clasificación de equipos y analítica de eventos tácticos, con límite máximo de 3000 fotogramas.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import cv2
import numpy as np

# Asegurar importación de src desde la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config_loader import load_config, Config
from src.utils.env import detect_environment, detect_headless_mode, is_cuda_available
from src.utils.geometry import pixel_to_field
from src.utils.video_stream import ThreadedVideoReader, AsyncVideoWriter
from src.perception.detector import YOLODetector
from src.perception.homography import (
    compute_homography,
    TemporalHomographyTracker,
    KeypointAuditLogger,
    HomographyEstimateResult,
    AuxFlowTracker,
)
from src.tracking.botsort import BallTracker, PlayerKinematicTracker
from src.tracking.ukf_botsort import UKFBoTSORTTracker
from src.analytics.offline_processor import OfflineTrajectoryProcessor
from src.appearance.clustering import TeamClassifier
from src.analytics.ball_possession import BallPossessionEngine
from src.analytics.events import MatchEventDetector
from src.storage.exporter import DataExporter
from src.visualization.renderer import MatchRenderer

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("analitica_futbol.tests.main")
logger.setLevel(logging.WARNING)

logging.getLogger("ultralytics").setLevel(logging.ERROR)
try:
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
    ULTRALYTICS_LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

MAX_TEST_FRAMES = 4000


def extract_ball_detection(ball_result, conf_threshold: float = 0.40) -> Optional[Dict[str, Any]]:
    """Extrae la detección más confiable de balón."""
    if ball_result.boxes is None or len(ball_result.boxes) == 0:
        return None
    b_boxes = ball_result.boxes.xyxy.cpu().numpy()
    b_confs = ball_result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(b_confs))
    if float(b_confs[best_idx]) < conf_threshold:
        return None
    bx1, by1, bx2, by2 = b_boxes[best_idx]
    return {
        "pixel_x": round(float((bx1 + bx2) / 2.0), 2),
        "pixel_y": round(float((by1 + by2) / 2.0), 2),
        "confidence": float(b_confs[best_idx]),
        "interpolated": False,
        "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
    }


class SoccerAnalyticsPipeline:
    """Orquestador integral del pipeline de analítica de fútbol (restringido para pruebas)."""

    def __init__(self, config: Config):
        self.config = config
        self.env = getattr(config, "environment", detect_environment())
        self.headless = getattr(config, "headless", detect_headless_mode(self.env))

        # 1. Percepción
        logger.info(f"[Pipeline] Inicializando YOLODetector en dispositivo: {config.device.upper()} ({config.format.upper()})")
        self.detector = YOLODetector(config)

        # 2. Homografía y telemetría
        homog_cfg = getattr(config, "homography", {})
        self.homography_interval = homog_cfg.get("interval", 1)
        self.enable_subpixel = homog_cfg.get("enable_subpixel", True)
        self.subpixel_win_size = tuple(homog_cfg.get("subpixel_win_size", [5, 5]))
        self.max_subpixel_shift_px = float(homog_cfg.get("max_subpixel_shift_px", 3.5))
        self.homography_tracker = TemporalHomographyTracker(
            enable_corner_shift=homog_cfg.get("enable_corner_shift", True),
        )
        self.auxflow_tracker = AuxFlowTracker(
            fb_threshold_px=float(homog_cfg.get("auxflow_fb_threshold_px", 1.0))
        )
        self.keypoint_audit_logger = KeypointAuditLogger()
        self._cached_H: Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None
        self.consecutive_no_kps_frames: int = 0

        # 3. Tracking (BoT-SORT Evolucionado: UKF + DIoU)
        track_cfg = getattr(config, "tracking", {})
        self.player_stride = max(1, track_cfg.get("player_stride", 1))
        det_cfg = getattr(config, "detection", {})
        conf_pl = float(det_cfg.get("conf_player", 0.50))
        self.ukf_botsort = UKFBoTSORTTracker(
            track_high_thresh=conf_pl,
            track_low_thresh=0.20,
            new_track_thresh=0.55,
            track_buffer=45,
            match_thresh=0.80,
        )
        ball_cfg = track_cfg.get("ball", {})
        self.ball_tracker = BallTracker(
            max_history=ball_cfg.get("max_history", 10),
            max_missing_frames=ball_cfg.get("max_missing_frames", 5),
            max_distance_jump=ball_cfg.get("max_distance_jump", 180.0),
        )
        self.player_kinematic = PlayerKinematicTracker()
        self.offline_processor = OfflineTrajectoryProcessor(
            sg_window_length=7,
            sg_polyorder=2,
            crossing_distance_thresh_m=1.5,
        )

        # 4. Apariencia
        app_cfg = getattr(config, "appearance", {})
        pitch_cfg = getattr(config, "pitch", {})
        self.team_classifier = TeamClassifier(
            min_samples_to_cluster=app_cfg.get("min_samples_to_cluster", 15),
            history_window=app_cfg.get("history_window", 30),
            lock_threshold=app_cfg.get("lock_threshold", 10),
            outlier_distance_lab=app_cfg.get("outlier_distance_lab", 28.0),
            pitch_length=pitch_cfg.get("length", 105.0),
            pitch_width=pitch_cfg.get("width", 68.0),
            luminance_weight=app_cfg.get("luminance_weight", 0.30),
        )

        # 5. Analítica de negocio
        ana_cfg = getattr(config, "analytics", {})
        self.possession_engine = BallPossessionEngine(
            possession_radius_m=ana_cfg.get("possession_radius_m", 1.8),
            min_possession_frames=ana_cfg.get("min_possession_frames", 3),
            max_relative_speed_mps=ana_cfg.get("max_control_relative_speed_mps", 6.0),
            enable_directional_cone=ana_cfg.get("enable_directional_cone", True),
        )
        self.event_detector = MatchEventDetector(
            pitch_length=pitch_cfg.get("length", 105.0),
            pitch_width=pitch_cfg.get("width", 68.0),
            shot_proximity_m=ana_cfg.get("shot_goal_proximity_m", 25.0),
        )

        # 6. Almacenamiento y persistencia
        self.exporter = DataExporter()

        # 7. Renderizado (solo si no es headless)
        vis_cfg = getattr(config, "visualization", {})
        if not self.headless:
            self.renderer = MatchRenderer(
                pitch_length=pitch_cfg.get("length", 105.0),
                pitch_width=pitch_cfg.get("width", 68.0),
                minimap_scale=vis_cfg.get("minimap_scale", 8.0),
                minimap_padding_m=vis_cfg.get("minimap_padding_m", 5.0),
                show_frame_number=vis_cfg.get("show_frame_number", True),
                show_player_ids=vis_cfg.get("show_player_ids", True),
                show_minimap_ids=vis_cfg.get("show_minimap_ids", True),
            )
        else:
            self.renderer = None

    def process_video(
        self,
        video_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        output_metadata_path: Optional[str] = None,
        max_frames: Optional[int] = MAX_TEST_FRAMES,
        export_sqlite: bool = False,
    ) -> Dict[str, Any]:
        """Procesa un video completo ejecutando el pipeline de principio a fin restringido a 3000 frames."""
        cfg_data = getattr(self.config, "data", {})
        v_in = video_path or cfg_data.get("input_video", "data/raw/input.mp4")
        v_out = output_video_path if output_video_path is not None else cfg_data.get("output_video", None)
        j_out = output_json_path or cfg_data.get("output_json", "data/processed/tracking_output.json")
        c_out = output_csv_path or cfg_data.get("output_csv", "data/processed/events_output.csv")
        m_out_raw = output_metadata_path or cfg_data.get("output_metadata", str(Path(j_out).parent / "match_metadata.json"))

        # Determinar si se generará video: solo si no es headless y se proporcionó una ruta de salida de video
        generate_video = (not self.headless) and (v_out is not None) and (str(v_out).strip() != "")

        # Resolver rutas relativas contra ROOT_DIR si es necesario
        video_file = Path(v_in)
        if not video_file.is_absolute() and not video_file.exists():
            cand = ROOT_DIR / video_file
            if cand.exists():
                video_file = cand

        if not video_file.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_file}")

        def resolve_output_path(path_str: Optional[str]) -> Optional[str]:
            if path_str is None:
                return None
            p = Path(path_str)
            if not p.is_absolute() and not p.parent.exists():
                cand_parent = ROOT_DIR / p.parent
                if cand_parent.exists():
                    return str(ROOT_DIR / p)
            return path_str

        v_out = resolve_output_path(v_out)
        j_out = resolve_output_path(j_out)
        c_out = resolve_output_path(c_out)
        m_out = resolve_output_path(m_out_raw)

        # Inicializar renderer bajo demanda si se va a generar video
        if generate_video and self.renderer is None:
            vis_cfg = getattr(self.config, "visualization", {})
            pitch_cfg = getattr(self.config, "pitch", {})
            self.renderer = MatchRenderer(
                pitch_length=pitch_cfg.get("length", 105.0),
                pitch_width=pitch_cfg.get("width", 68.0),
                minimap_scale=vis_cfg.get("minimap_scale", 8.0),
                minimap_padding_m=vis_cfg.get("minimap_padding_m", 5.0),
                show_frame_number=vis_cfg.get("show_frame_number", True),
                show_player_ids=vis_cfg.get("show_player_ids", True),
                show_minimap_ids=vis_cfg.get("show_minimap_ids", True),
            )

        # Restricción estricta a un máximo de 3000 frames
        if max_frames is None or max_frames > MAX_TEST_FRAMES:
            effective_limit = MAX_TEST_FRAMES
        else:
            effective_limit = max(1, max_frames)

        logger.info(f"🔒 [Modo Test] Restricción de fotogramas activa: máximo {effective_limit} frames.")

        stream_cfg = getattr(self.config, "stream", {})
        use_reader = stream_cfg.get("use_threaded_reader", True)
        buf_size = stream_cfg.get("reader_buffer_size", 64)

        if use_reader:
            reader = ThreadedVideoReader(str(video_file), buffer_size=buf_size)
        else:
            reader = cv2.VideoCapture(str(video_file))

        if not reader.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_file}")

        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(reader.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_total = min(total_frames, effective_limit) if total_frames > 0 else effective_limit

        # Escalar resolución si supera max_width para acelerar inferencia y reducir consumo de memoria/bus
        max_w = stream_cfg.get("max_width", 1920)
        scale_input = 1.0
        proc_width = width
        proc_height = height
        if max_w and width > max_w:
            scale_input = float(max_w) / float(width)
            proc_width = int(round(width * scale_input))
            proc_height = int(round(height * scale_input))

        res_str = f"{proc_width}x{proc_height}" if scale_input == 1.0 else f"{proc_width}x{proc_height} (original: {width}x{height})"
        device_str = f"GPU: CUDA | Formato: {self.config.format.upper()}" if is_cuda_available() and self.config.device == "cuda" else f"CPU | Formato: {self.config.format.upper()}"

        print("=" * 65)
        print("🚀 INICIANDO ANÁLISIS DE FÚTBOL (MODO PRUEBA)")
        print(f"   Entorno:      {self.env.upper()} ({device_str})")
        print(f"   Video:        {video_file.name}")
        print(f"   Resolución:   {res_str}")
        print(f"   Total Frames: {total_frames:,} (Límite Test: {effective_total:,}) @ {fps:.1f} fps")
        print(f"   Modo:         {'VIDEO + OVERLAYS (Salida: ' + str(v_out) + ')' if generate_video else 'HEADLESS (Extracción directa JSON/CSV sin video)'}")
        print("=" * 65, flush=True)

        writer = None
        if generate_video and v_out:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            minimap_aspect = (self.config.pitch.length + 10.0) / (self.config.pitch.width + 10.0)
            combined_w = proc_width + int(proc_height * minimap_aspect)
            if stream_cfg.get("use_async_writer", True):
                writer = AsyncVideoWriter(v_out, fourcc, fps, (combined_w, proc_height), buffer_size=stream_cfg.get("writer_buffer_size", 60))
            else:
                Path(v_out).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(v_out, fourcc, fps, (combined_w, proc_height))

        frame_count = 0
        start_time = time.time()
        tracking_frames: List[Dict[str, Any]] = []
        raw_trajectories_map: Dict[int, List[Dict[str, Any]]] = {}

        total_read_decode_time = 0.0
        total_inference_time = 0.0
        total_tracking_time = 0.0
        total_homography_time = 0.0
        total_render_time = 0.0

        det_cfg = getattr(self.config, "detection", {})
        conf_kp = det_cfg.get("conf_keypoint", 0.35)
        conf_pl = det_cfg.get("conf_player", 0.40)
        conf_ba = det_cfg.get("conf_ball", 0.40)

        try:
            while reader.isOpened():
                t_read0 = time.perf_counter()
                ret, frame_bgr = reader.read()
                total_read_decode_time += time.perf_counter() - t_read0

                if not ret or frame_bgr is None:
                    break
                if frame_count >= effective_limit:
                    break

                if scale_input < 1.0:
                    frame_bgr = cv2.resize(frame_bgr, (proc_width, proc_height), interpolation=cv2.INTER_LINEAR)

                frame_count += 1
                ts = frame_count / fps

                # ── Inferencia y Tracking ────────────────────────────────────
                t_inf0 = time.perf_counter()
                should_infer_players = (frame_count % self.player_stride == 0) or (len(self.player_kinematic.last_detections) == 0)

                if should_infer_players:
                    kp_res, player_res, ball_res = self.detector.infer(
                        frame_bgr, conf_kp=conf_kp, conf_player=conf_pl, conf_ball=conf_ba, is_video=True
                    )
                else:
                    kp_res, ball_res = self.detector.infer_keypoints_and_ball(
                        frame_bgr, conf_kp=conf_kp, conf_ball=conf_ba
                    )
                    player_res = None
                total_inference_time += time.perf_counter() - t_inf0

                # ── Ball Tracker ─────────────────────────────────────────────
                t_tr0 = time.perf_counter()
                raw_ball = extract_ball_detection(ball_res, conf_ba)
                ball_det, ball_history = self.ball_tracker.update(raw_ball)
                total_tracking_time += time.perf_counter() - t_tr0

                # ── Homografía ───────────────────────────────────────────────
                t_hg0 = time.perf_counter()
                recalc_h = (frame_count % self.homography_interval == 0) or (self.homography_tracker.cached_H is None)
                H_cand = None
                mask_cand = None
                diag = None
                cand_residuals = None

                # Extraer cajas de personas para descarte de keypoints y detector de plano cerrado
                player_boxes = None
                is_close_up = False
                h_img_f, w_img_f = frame_bgr.shape[:2]

                # Criterio 1: Tamaño relativo y cobertura de jugadores en la escena
                if player_res is not None and player_res.boxes is not None and len(player_res.boxes) > 0:
                    player_boxes = player_res.boxes.xyxy.cpu().numpy()
                    box_heights = player_boxes[:, 3] - player_boxes[:, 1]
                    box_widths = player_boxes[:, 2] - player_boxes[:, 0]
                    max_box_h = float(np.max(box_heights)) if len(box_heights) > 0 else 0.0
                    total_people_area = float(np.sum(box_heights * box_widths))
                    people_area_ratio = total_people_area / float(h_img_f * w_img_f)

                    # En tomas tácticas normales los jugadores miden ~5-12% de la altura y ocupan < 3% del área
                    if max_box_h > (0.16 * h_img_f) or people_area_ratio > 0.07:
                        is_close_up = True

                # Criterio 2: Cobertura global de césped verde en el encuadre
                # En un plano táctico de fútbol el césped ocupa > 55% de la toma. En planos medios o vallas ocupa mucho menos.
                hsv_global = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                green_mask_global = (
                    (hsv_global[:, :, 0] >= 28)
                    & (hsv_global[:, :, 0] <= 88)
                    & (hsv_global[:, :, 1] >= 25)
                    & (hsv_global[:, :, 2] >= 25)
                )
                field_green_ratio = float(np.mean(green_mask_global))
                # Contar keypoints válidos en el frame actual
                n_valid_kps = 0
                if kp_res is not None and kp_res.keypoints is not None and len(kp_res.keypoints.data) > 0:
                    kps_confs = kp_res.keypoints.conf[0].cpu().numpy()
                    n_valid_kps = int(np.sum(kps_confs >= conf_kp))

                if n_valid_kps == 0:
                    self.consecutive_no_kps_frames += 1
                else:
                    self.consecutive_no_kps_frames = 0

                # Si no hay keypoints por más de 3 frames (plano cerrado, corte, repetición) -> Forzar expiración de H
                if self.consecutive_no_kps_frames > 3:
                    is_close_up = True

                if is_close_up:
                    # En plano cerrado o toma sin keypoints de campo: invalidar H completamente
                    H_cand = None
                    self._cached_H = None
                    self._cached_mask = None
                    if hasattr(self.homography_tracker, "cached_H"):
                        self.homography_tracker.cached_H = None
                    if hasattr(self.auxflow_tracker, "prev_H"):
                        self.auxflow_tracker.prev_H = None
                elif recalc_h:
                    try:
                        magsac_cfg_dict = getattr(self.config.homography, "magsac", None)
                        if hasattr(magsac_cfg_dict, "__dict__"):
                            magsac_cfg_dict = vars(magsac_cfg_dict)

                        res = compute_homography(
                            kp_res,
                            conf_threshold=conf_kp,
                            ransac_threshold=self.config.homography.ransac_threshold,
                            img_width=proc_width,
                            img_height=proc_height,
                            frame_bgr=frame_bgr,
                            enable_subpixel=self.enable_subpixel,
                            subpixel_win_size=self.subpixel_win_size,
                            max_subpixel_shift_px=self.max_subpixel_shift_px,
                            exclude_boxes=player_boxes,
                            method=getattr(self.config.homography, "method", "MAGSAC++"),
                            magsac_cfg=magsac_cfg_dict,
                            prev_H=self._cached_H,
                        )
                        H_cand = res.H
                        mask_cand = res.mask
                        diag = res.diagnostics
                        cand_residuals = res.residuals
                    except Exception:
                        H_cand = None

                    H_eff, mask_eff, _, _ = self.homography_tracker.update(
                        H_cand, mask_cand, img_width=proc_width, img_height=proc_height, frame_id=frame_count, diagnostics=diag
                    )
                    # AuxFlow con Verificación Forward-Backward (<= 1px) para estabilización
                    H_eff = self.auxflow_tracker.update(frame_bgr, current_H=H_eff, exclude_boxes=player_boxes)
                    self._cached_H = H_eff
                    self._cached_mask = mask_eff

                H = self._cached_H if not is_close_up else None
                total_homography_time += time.perf_counter() - t_hg0

                # ── Proyección Balón a Campo ─────────────────────────────────
                if ball_det is not None and H is not None:
                    bf_x, bf_y = pixel_to_field(ball_det["pixel_x"], ball_det["pixel_y"], H)
                    ball_det["pitch_x_m"] = round(bf_x, 2)
                    ball_det["pitch_y_m"] = round(bf_y, 2)
                elif ball_det is not None:
                    ball_det["pitch_x_m"] = None
                    ball_det["pitch_y_m"] = None

                # ── Extracción y Tracking de Jugadores (BoT-SORT Evolucionado: UKF + DIoU) ──
                player_detections: List[Dict[str, Any]] = []
                if should_infer_players and player_res is not None and player_res.boxes is not None and len(player_res.boxes) > 0:
                    raw_dets = []
                    boxes = player_res.boxes.xyxy.cpu().numpy()
                    confs = player_res.boxes.conf.cpu().numpy()
                    classes = player_res.boxes.cls.cpu().numpy()

                    for box, conf, cls_id in zip(boxes, confs, classes):
                        cls_int = int(cls_id)
                        if cls_int in (1, 2) and conf >= 0.20:
                            raw_dets.append({
                                "bbox": [float(b) for b in box],
                                "confidence": float(conf),
                                "class_id": cls_int,
                            })

                    # Asociación en 2 etapas + predicción CTRV con UKF + costo DIoU
                    tracked_players = self.ukf_botsort.update(raw_dets)

                    for p in tracked_players:
                        xc = p["pixel_x"]
                        yb = p["pixel_y"]
                        fx, fy = pixel_to_field(xc, yb, H) if H is not None else (None, None)
                        if fx is not None and fy is not None:
                            if not (-5.0 <= fx <= self.config.pitch.length + 5.0 and -5.0 <= fy <= self.config.pitch.width + 5.0):
                                fx, fy = None, None

                        p["pitch_x_m"] = round(fx, 2) if fx is not None else None
                        p["pitch_y_m"] = round(fy, 2) if fy is not None else None
                        player_detections.append(p)

                    self.team_classifier.annotate_player_detections(frame_bgr, player_detections)
                    self.player_kinematic.update_from_detections(player_detections)
                else:
                    player_detections = self.player_kinematic.extrapolate_next_frame(H=H)

                # Acumular en trayectorias crudas para post-procesado batch
                for p in player_detections:
                    pid = p["player_id"]
                    if pid not in raw_trajectories_map:
                        raw_trajectories_map[pid] = []
                    raw_trajectories_map[pid].append({
                        "frame_idx": frame_count,
                        "player_id": pid,
                        "pitch_x_m": p.get("pitch_x_m"),
                        "pitch_y_m": p.get("pitch_y_m"),
                        "pixel_x": p.get("pixel_x"),
                        "pixel_y": p.get("pixel_y"),
                        "bbox": p.get("bbox"),
                    })

                # ── Analítica de Posesión y Eventos ──────────────────────────
                possession_info = self.possession_engine.update(ball_det, player_detections, timestamp_sec=ts)
                self.event_detector.update(frame_count, ts, possession_info, ball_det)

                # Keypoints detectados para calibración interactiva y renderizado en front
                frame_keypoints = []
                if kp_res is not None and kp_res.keypoints is not None and len(kp_res.keypoints.data) > 0:
                    from src.perception.homography import KP_MAP, FIELD_COORDS
                    kps_arr = kp_res.keypoints.data[0].cpu().numpy()
                    for idx_kp, pt_kp in enumerate(kps_arr):
                        k_x, k_y, k_conf = pt_kp
                        if k_conf >= conf_kp and idx_kp in KP_MAP:
                            kp_id = KP_MAP[idx_kp]
                            canon_coords = FIELD_COORDS.get(kp_id, (None, None))
                            frame_keypoints.append({
                                "id": kp_id,
                                "pixel_x": round(float(k_x), 2),
                                "pixel_y": round(float(k_y), 2),
                                "pitch_x": round(float(canon_coords[0]), 2) if canon_coords[0] is not None else None,
                                "pitch_y": round(float(canon_coords[1]), 2) if canon_coords[1] is not None else None,
                                "conf": round(float(k_conf), 3),
                            })

                # ── JSON Frame Data ──────────────────────────────────────────
                tracking_frames.append({
                    "frame": frame_count,
                    "timestamp": round(ts, 3),
                    "keypoints": frame_keypoints,
                    "players": [
                        {
                            "track_id": p["player_id"],
                            "team": self.team_classifier.get_team_key(p.get("team_id", 0)),
                            "team_id": p.get("team_id", 0),
                            "role": p.get("role", "player"),
                            "confidence": p.get("confidence", 1.0),
                            "is_interpolated": p.get("is_interpolated", False),
                            "bbox": p.get("bbox"),
                            "pixel_x": p["pixel_x"],
                            "pixel_y": p["pixel_y"],
                            "pitch_x": p["pitch_x_m"],
                            "pitch_y": p["pitch_y_m"],
                        }
                        for p in player_detections
                    ],
                    "ball": {
                        "detected": True,
                        "confidence": ball_det.get("confidence", 1.0),
                        "bbox": ball_det.get("bbox"),
                        "pixel_x": ball_det["pixel_x"],
                        "pixel_y": ball_det["pixel_y"],
                        "pitch_x": ball_det["pitch_x_m"],
                        "pitch_y": ball_det["pitch_y_m"],
                        "is_interpolated": ball_det.get("interpolated", False),
                    } if ball_det is not None else None,
                    "possession": possession_info,
                    "homography": {
                        "valid": bool(H is not None),
                        "matrix": [[round(float(val), 6) for val in row] for row in H] if H is not None else None,
                        "inv_matrix": (
                            [[round(float(val), 6) for val in row] for row in np.linalg.inv(H)]
                            if H is not None else None
                        ),
                    },
                    "homography_valid": bool(H is not None),
                })

                # ── Renderizado de Video ─────────────────────────────────────
                if not self.headless and self.renderer is not None and writer is not None:
                    t_rend0 = time.perf_counter()
                    annotated = frame_bgr.copy()
                    if H is not None and not is_close_up:
                        self.renderer.draw_pitch_lines_on_frame(annotated, H)
                        self.renderer.draw_keypoints_on_frame(annotated, frame_keypoints)

                    self.renderer.draw_players_on_frame(annotated, player_detections)
                    self.renderer.draw_ball_on_frame(annotated, ball_det, ball_history)

                    # Superponer número de fotograma en la esquina superior izquierda (Zona izquierda superior)
                    frame_text = f"FRAME: {frame_count:06d}"
                    font_f = cv2.FONT_HERSHEY_SIMPLEX
                    (tw_f, th_f), _ = cv2.getTextSize(frame_text, font_f, 0.65, 2)
                    # Fondo semi-transparente oscuro
                    cv2.rectangle(annotated, (20, 20), (35 + tw_f, 35 + th_f), (15, 23, 42), -1)
                    cv2.rectangle(annotated, (20, 20), (35 + tw_f, 35 + th_f), (0, 215, 255), 1)
                    cv2.putText(annotated, frame_text, (28, 26 + th_f), font_f, 0.65, (255, 255, 255), 2, lineType=cv2.LINE_AA)

                    # Si es plano cerrado o H es None, renderizar el minimap vacío sin proyecciones fantasma
                    valid_for_minimap = player_detections if (H is not None and not is_close_up) else []
                    valid_ball_minimap = ball_det if (H is not None and not is_close_up) else None
                    minimap = self.renderer.render_minimap(valid_for_minimap, valid_ball_minimap)
                    combined = self.renderer.compose_output_frame(annotated, minimap)
                    writer.write(combined)
                    total_render_time += time.perf_counter() - t_rend0

                # Barra de progreso
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_avg = frame_count / elapsed
                    pct = (frame_count / effective_total) * 100.0 if effective_total > 0 else 0
                    print(f"\r  [{'█' * int(25 * pct / 100):25s}] {pct:5.1f}%  {frame_count}/{effective_total}  {fps_avg:.1f} FPS", end="", flush=True)

        finally:
            reader.release()
            if writer:
                writer.release()
            print()

        total_time = time.time() - start_time
        fps_final = frame_count / total_time if total_time > 0 else 0.0

        # ── ETAPA 2: BATCH / OFFLINE (Savitzky-Golay + Corrección Geométrica de Cruces) ──
        if raw_trajectories_map:
            logger.info("🔧 [Etapa 2 Batch] Ejecutando filtro Savitzky-Golay (w=7, n=2) y resolución de cruces...")
            refined_trajectories = self.offline_processor.process_all(raw_trajectories_map)

            # Re-inyectar las coordenadas suavizadas y corregidas en tracking_frames
            # Crear diccionario rápido (frame_idx, orig_pid) -> punto_refinado
            refined_lookup = {}
            for t_id, pts in refined_trajectories.items():
                for p in pts:
                    key = (p["frame_idx"], p.get("pixel_x"), p.get("pixel_y"))
                    refined_lookup[key] = p

            for frame_dict in tracking_frames:
                f_idx = frame_dict["frame"]
                for p in frame_dict["players"]:
                    key = (f_idx, p.get("pixel_x"), p.get("pixel_y"))
                    if key in refined_lookup:
                        ref_pt = refined_lookup[key]
                        p["pitch_x"] = ref_pt.get("pitch_x_m")
                        p["pitch_y"] = ref_pt.get("pitch_y_m")
                        if "player_id" in ref_pt:
                            p["track_id"] = ref_pt["player_id"]
                        if ref_pt.get("crossing_corrected"):
                            p["crossing_corrected"] = True
                        if ref_pt.get("is_smoothed_sg"):
                            p["is_smoothed_sg"] = True

        # Construir Dataset Contract Oficial y Archivo de Metadatos Independiente
        all_events = self.event_detector.get_all_events()
        teams_meta = self.team_classifier.get_teams_metadata()
        possession_stats = self.possession_engine.get_possession_stats()

        match_metadata = {
            "match_info": {
                "video_name": video_file.name,
                "fps": round(fps, 2),
                "total_frames": frame_count,
                "duration_seconds": round(frame_count / fps, 2),
                "resolution": {
                    "width": proc_width,
                    "height": proc_height,
                    "original_width": width,
                    "original_height": height,
                },
                "pitch_dimensions_m": {
                    "length": float(self.config.pitch.length),
                    "width": float(self.config.pitch.width),
                },
            },
            "teams": {
                "team_a": teams_meta["team_a"],
                "team_b": teams_meta["team_b"],
                "referee": teams_meta["referee"],
            },
            "possession_stats": possession_stats,
            "inference": {
                "device": self.config.device,
                "format": self.config.format,
                "cuda_available": is_cuda_available(),
                "environment": self.env,
            },
            "performance": {
                "total_time_s": round(total_time, 2),
                "fps_average": round(fps_final, 2),
                "read_and_decode_s": round(total_read_decode_time, 2),
                "inference_s": round(total_inference_time, 2),
                "tracking_s": round(total_tracking_time, 2),
                "homography_s": round(total_homography_time, 2),
            },
        }

        final_dataset = {
            "metadata_file": Path(m_out).name,
            "metadata": match_metadata,
            "frames": tracking_frames,
            "events": all_events,
        }

        # Exportación
        self.exporter.export_metadata_json(match_metadata, m_out)
        self.exporter.export_json(final_dataset, j_out)
        if all_events:
            self.exporter.export_events_csv(all_events, c_out)
        if export_sqlite:
            db_out = str(Path(j_out).with_suffix(".sqlite"))
            self.exporter.export_sqlite(final_dataset, db_out)

        # Resumen final limpio
        print()
        print("=" * 65)
        print("✅ PROCESAMIENTO COMPLETADO (MODO PRUEBA)")
        print(f"   Tiempo total:    {total_time:.1f}s (~{int(total_time // 60)} min {int(total_time % 60)}s)")
        print(f"   FPS promedio:    {fps_final:.1f} FPS")
        print(f"   Posesión:        Equipo A: {possession_stats.get('team_0_pct', 0)}% | Equipo B: {possession_stats.get('team_1_pct', 0)}%")
        print(f"   Eventos:         {len(all_events)} detectados")
        print("   Archivos generados:")
        print(f"     - Tracking JSON:  {j_out}")
        print(f"     - Metadata JSON:  {m_out}")
        if all_events:
            print(f"     - Eventos CSV:    {c_out}")
        if not self.headless:
            print(f"     - Video Output:   {v_out}")
        print("=" * 65, flush=True)

        return final_dataset


def main():
    parser = argparse.ArgumentParser(description="analitica_futbol: Orquestador de Prueba (restringido a máx 3000 frames)")
    parser.add_argument("--config", "-cfg", default=None, help="Ruta al archivo YAML de inferencia (ej: configs/inference_local.yaml)")
    parser.add_argument("--video", "-v", default=None, help="Ruta al video de entrada")
    parser.add_argument("--output", "-o", default=None, help="Ruta al video MP4 de salida")
    parser.add_argument("--json", "-j", default=None, help="Ruta al JSON de salida")
    parser.add_argument("--csv", "-c", default=None, help="Ruta al CSV de eventos de salida")
    parser.add_argument("--metadata", "-md", default=None, help="Ruta al archivo JSON de metadatos del partido")
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=MAX_TEST_FRAMES,
        help=f"Límite máximo de frames a procesar (por defecto {MAX_TEST_FRAMES}, restringido a máx {MAX_TEST_FRAMES})"
    )
    parser.add_argument("--max-width", type=int, default=None, help="Ancho máximo de resolución para procesar (ej: 1920)")
    parser.add_argument("--headless", action="store_true", default=None, help="Forzar modo headless (sin MP4 ni overlays)")
    parser.add_argument("--export-sqlite", action="store_true", default=False, help="Exportar base de datos SQLite")
    args = parser.parse_args()

    # Garantizar tope estricto de MAX_TEST_FRAMES
    capped_max_frames = min(args.max_frames, MAX_TEST_FRAMES) if args.max_frames is not None else MAX_TEST_FRAMES

    config = load_config(args.config, project_root=ROOT_DIR)
    if args.headless:
        config.headless = True
    if args.max_width is not None:
        if not hasattr(config, "stream") or config.stream is None:
            config.stream = {}
        if isinstance(config.stream, dict):
            config.stream["max_width"] = args.max_width
        else:
            setattr(config.stream, "max_width", args.max_width)

    pipeline = SoccerAnalyticsPipeline(config)
    pipeline.process_video(
        video_path=args.video,
        output_video_path=args.output,
        output_json_path=args.json,
        output_csv_path=args.csv,
        output_metadata_path=args.metadata,
        max_frames=capped_max_frames,
        export_sqlite=args.export_sqlite,
    )


if __name__ == "__main__":
    main()
