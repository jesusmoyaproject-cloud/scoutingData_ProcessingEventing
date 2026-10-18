"""
analitica_futbol: Orquestador Principal (End-to-End)
Pipeline unificado de alto rendimiento para visión artificial, tracking cinemático,
clasificación de equipos y analítica de eventos tácticos.
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

# Asegurar importación de src
ROOT_DIR = Path(__file__).resolve().parent
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
)
from src.tracking.botsort import BallTracker, PlayerKinematicTracker
from src.appearance.clustering import TeamClassifier
from src.analytics.ball_possession import BallPossessionEngine
from src.analytics.events import MatchEventDetector
from src.storage.exporter import DataExporter
from src.visualization.renderer import MatchRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("analitica_futbol.main")


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
    """Orquestador integral del pipeline de analítica de fútbol."""

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
        self.keypoint_audit_logger = KeypointAuditLogger()
        self._cached_H: Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None

        # 3. Tracking
        track_cfg = getattr(config, "tracking", {})
        self.player_stride = max(1, track_cfg.get("player_stride", 1))
        ball_cfg = track_cfg.get("ball", {})
        self.ball_tracker = BallTracker(
            max_history=ball_cfg.get("max_history", 10),
            max_missing_frames=ball_cfg.get("max_missing_frames", 5),
            max_distance_jump=ball_cfg.get("max_distance_jump", 180.0),
        )
        self.player_kinematic = PlayerKinematicTracker()

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
        max_frames: Optional[int] = None,
        export_sqlite: bool = False,
    ) -> Dict[str, Any]:
        """Procesa un video completo ejecutando el pipeline de principio a fin."""
        cfg_data = getattr(self.config, "data", {})
        v_in = video_path or cfg_data.get("input_video", "data/raw/input.mp4")
        v_out = output_video_path or cfg_data.get("output_video", "data/processed/scouting_output.mp4")
        j_out = output_json_path or cfg_data.get("output_json", "data/processed/tracking_output.json")
        c_out = output_csv_path or cfg_data.get("output_csv", "data/processed/events_output.csv")
        m_out = output_metadata_path or cfg_data.get("output_metadata", str(Path(j_out).parent / "match_metadata.json"))

        video_file = Path(v_in)
        if not video_file.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_file}")

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
        effective_total = min(total_frames, max_frames) if max_frames else total_frames

        logger.info(
            f"📹 Video: {video_file.name} | {width}x{height} @ {fps:.1f}fps | "
            f"Total: {total_frames} frames (Límite: {effective_total}) | "
            f"Modo: {'HEADLESS' if self.headless else 'VIDEO + OVERLAYS'}"
        )

        writer: Optional[AsyncVideoWriter] = None
        if not self.headless and stream_cfg.get("use_async_writer", True):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # El frame combinado añade el minimapa a la derecha
            # Minimap width aprox = (105 + 10)*8 = 920px escalado a la altura de imagen
            minimap_aspect = (self.config.pitch.length + 10.0) / (self.config.pitch.width + 10.0)
            combined_w = width + int(height * minimap_aspect)
            writer = AsyncVideoWriter(v_out, fourcc, fps, (combined_w, height), buffer_size=stream_cfg.get("writer_buffer_size", 60))

        frame_count = 0
        start_time = time.time()
        tracking_frames: List[Dict[str, Any]] = []

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
                if max_frames and frame_count >= max_frames:
                    break

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
                    kp_res = self.detector.infer_keypoints(frame_bgr, conf=conf_kp)
                    ball_res = self.detector.infer_ball(frame_bgr, conf=conf_ba)
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

                if recalc_h:
                    try:
                        res = compute_homography(
                            kp_res,
                            conf_threshold=conf_kp,
                            ransac_threshold=self.config.homography.ransac_threshold,
                            img_width=width,
                            img_height=height,
                            frame_bgr=frame_bgr,
                            enable_subpixel=self.enable_subpixel,
                            subpixel_win_size=self.subpixel_win_size,
                            max_subpixel_shift_px=self.max_subpixel_shift_px,
                        )
                        H_cand = res.H
                        mask_cand = res.mask
                        diag = res.diagnostics
                        cand_residuals = res.residuals
                    except Exception:
                        H_cand = None

                    H_eff, mask_eff, _, _ = self.homography_tracker.update(
                        H_cand, mask_cand, img_width=width, img_height=height, frame_id=frame_count, diagnostics=diag
                    )
                    self._cached_H = H_eff
                    self._cached_mask = mask_eff

                H = self._cached_H
                total_homography_time += time.perf_counter() - t_hg0

                # ── Proyección Balón a Campo ─────────────────────────────────
                if ball_det is not None and H is not None:
                    bf_x, bf_y = pixel_to_field(ball_det["pixel_x"], ball_det["pixel_y"], H)
                    ball_det["pitch_x_m"] = round(bf_x, 2)
                    ball_det["pitch_y_m"] = round(bf_y, 2)
                elif ball_det is not None:
                    ball_det["pitch_x_m"] = None
                    ball_det["pitch_y_m"] = None

                # ── Extracción y Tracking de Jugadores ───────────────────────
                player_detections: List[Dict[str, Any]] = []
                if should_infer_players and player_res is not None and player_res.boxes is not None and len(player_res.boxes) > 0:
                    boxes = player_res.boxes.xyxy.cpu().numpy()
                    confs = player_res.boxes.conf.cpu().numpy()
                    classes = player_res.boxes.cls.cpu().numpy()
                    track_ids = player_res.boxes.id.cpu().numpy().astype(int) if player_res.boxes.id is not None else None

                    for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
                        if conf < conf_pl:
                            continue
                        cls_int = int(cls_id)
                        # Filtrar estrictamente solo clases 1 (goalkeeper) y 2 (player) para evitar detecciones incorrectas
                        if cls_int not in (1, 2):
                            continue

                        x1, y1, x2, y2 = box
                        xc = float((x1 + x2) / 2.0)
                        yb = float(y2)
                        p_id = int(track_ids[idx]) if track_ids is not None else idx
                        fx, fy = pixel_to_field(xc, yb, H) if H is not None else (None, None)
                        role = "goalkeeper" if cls_int == 1 else "player"

                        player_detections.append({
                            "player_id": p_id,
                            "class_id": cls_int,
                            "role": role,
                            "confidence": float(conf),
                            "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                            "pixel_x": round(xc, 2),
                            "pixel_y": round(yb, 2),
                            "pitch_x_m": round(fx, 2) if fx is not None else None,
                            "pitch_y_m": round(fy, 2) if fy is not None else None,
                            "is_interpolated": False,
                        })

                    self.team_classifier.annotate_player_detections(frame_bgr, player_detections)
                    self.player_kinematic.update_from_detections(player_detections)
                else:
                    player_detections = self.player_kinematic.extrapolate_next_frame(H=H)

                # ── Analítica de Posesión y Eventos ──────────────────────────
                possession_info = self.possession_engine.update(ball_det, player_detections, timestamp_sec=ts)
                self.event_detector.update(frame_count, ts, possession_info, ball_det)

                # ── JSON Frame Data ──────────────────────────────────────────
                tracking_frames.append({
                    "frame": frame_count,
                    "timestamp": round(ts, 3),
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
                    "homography_valid": bool(H is not None),
                })

                # ── Renderizado de Video ─────────────────────────────────────
                if not self.headless and self.renderer is not None and writer is not None:
                    t_rend0 = time.perf_counter()
                    annotated = frame_bgr.copy()
                    if H is not None:
                        self.renderer.draw_pitch_lines_on_frame(annotated, H)
                    self.renderer.draw_players_on_frame(annotated, player_detections)
                    self.renderer.draw_ball_on_frame(annotated, ball_det, ball_history)

                    minimap = self.renderer.render_minimap(player_detections, ball_det)
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
                    "width": width,
                    "height": height,
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

        # Resumen
        logger.info("=" * 65)
        logger.info("📊 RESUMEN FINAL DE EJECUCIÓN (analitica_futbol)")
        logger.info(f"   Frames procesados: {frame_count} en {total_time:.2f}s ({fps_final:.2f} FPS)")
        logger.info(f"   Posesión:          Equipo A: {possession_stats.get('team_0_pct', 0)}% | "
                    f"Equipo B: {possession_stats.get('team_1_pct', 0)}%")
        logger.info(f"   Eventos tácticos:  {len(all_events)} detectados")
        logger.info(f"   Metadata JSON:     {m_out}")
        logger.info(f"   Tracking JSON:     {j_out}")
        if not self.headless:
            logger.info(f"   Video Output:      {v_out}")
        logger.info("=" * 65)

        return final_dataset


def main():
    parser = argparse.ArgumentParser(description="analitica_futbol: Orquestador Principal")
    parser.add_argument("--config", "-cfg", default=None, help="Ruta al archivo YAML de inferencia (ej: configs/inference_local.yaml)")
    parser.add_argument("--video", "-v", default=None, help="Ruta al video de entrada")
    parser.add_argument("--output", "-o", default=None, help="Ruta al video MP4 de salida")
    parser.add_argument("--json", "-j", default=None, help="Ruta al JSON de salida")
    parser.add_argument("--csv", "-c", default=None, help="Ruta al CSV de eventos de salida")
    parser.add_argument("--metadata", "-md", default=None, help="Ruta al archivo JSON de metadatos del partido")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Límite máximo de frames a procesar")
    parser.add_argument("--headless", action="store_true", default=None, help="Forzar modo headless (sin MP4 ni overlays)")
    parser.add_argument("--export-sqlite", action="store_true", default=False, help="Exportar base de datos SQLite")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.headless:
        config.headless = True

    pipeline = SoccerAnalyticsPipeline(config)
    pipeline.process_video(
        video_path=args.video,
        output_video_path=args.output,
        output_json_path=args.json,
        output_csv_path=args.csv,
        output_metadata_path=args.metadata,
        max_frames=args.max_frames,
        export_sqlite=args.export_sqlite,
    )


if __name__ == "__main__":
    main()
