"""
Módulo de Visión Artificial: Inferencia YOLO Unificada (PyTorch / OpenVINO)
Ejecuta inferencia directa sin sockets HTTP ni capas cliente-servidor para:
1. Keypoints del terreno de juego (YOLO Pose)
2. Jugadores y árbitros (YOLO Detect + BoT-SORT)
3. Balón de fútbol (YOLO Detect)
"""

import logging
import warnings
import time
from pathlib import Path
from typing import Tuple, Optional, Any, Union, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Silenciar advertencias de dependencias externas
warnings.filterwarnings("ignore")
try:
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
    ULTRALYTICS_LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

import torch
from ultralytics import YOLO

from src.utils.config_loader import Config, load_config
from src.utils.env import is_cuda_available

logger = logging.getLogger("src.perception.detector")


class YOLODetector:
    """
    Detector unificado multimodelo basado en Ultralytics YOLO.
    Soporta inferencia tanto sobre modelos PyTorch (.pt en CUDA GPU) como
    OpenVINO IR (.xml/.bin en Intel CPU).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        model_format: Optional[str] = None,
        player_model_path: Optional[Union[str, Path]] = None,
        ball_model_path: Optional[Union[str, Path]] = None,
        keypoint_model_path: Optional[Union[str, Path]] = None,
        tracker_config_path: Optional[Union[str, Path]] = None,
    ):
        if config is None:
            config = load_config()

        self.config = config
        self.device = device or getattr(config, "device", "cpu")
        self.format = (model_format or getattr(config, "format", "openvino")).lower()

        # Validación de CUDA
        if self.device == "cuda" and not is_cuda_available():
            logger.warning("[YOLODetector] ⚠️ CUDA no disponible en este entorno. Cambiando a CPU.")
            self.device = "cpu"

        # Resolver rutas de modelos
        root = Path(getattr(config, "_project_root", Path(__file__).resolve().parent.parent.parent))

        def _resolve_path(p: Optional[Union[str, Path]], default_rel: str) -> Path:
            if p:
                path_obj = Path(p)
                return path_obj if path_obj.is_absolute() else root / path_obj
            return root / default_rel

        # Rutas por defecto según formato
        if self.format == "engine":
            def_player = "models/modelPlayer.engine"
            def_ball = "models/modelBall.engine"
            def_kp = "models/modelKeypoint.engine"
        elif self.format == "pt":
            def_player = "models/modelPlayer.pt"
            def_ball = "models/modelBall.pt"
            def_kp = "models/modelKeypoint.pt"
        else:
            def_player = "models/modelPlayer_openvino_model"
            def_ball = "models/modelBall_openvino_model"
            def_kp = "models/modelKeypoint_openvino_model"

        cfg_models = getattr(config, "models", {})
        self.player_model_path = _resolve_path(
            player_model_path or cfg_models.get("player"), def_player
        )
        self.ball_model_path = _resolve_path(
            ball_model_path or cfg_models.get("ball"), def_ball
        )
        self.keypoint_model_path = _resolve_path(
            keypoint_model_path or cfg_models.get("keypoints"), def_kp
        )

        # Configuración de BoT-SORT
        cfg_track = getattr(config, "tracking", {})
        track_cfg_name = cfg_track.get("tracker_config", "configs/botsort_sports.yaml")
        self.tracker_config_path = _resolve_path(tracker_config_path, track_cfg_name)

        # Configuración de precisión FP16 (Half) para GPU CUDA
        cfg_half = getattr(config, "half", None)
        if cfg_half is not None:
            self.half = bool(cfg_half) and ("cuda" in str(self.device).lower())
        else:
            self.half = (self.format in ("pt", "engine") and "cuda" in str(self.device).lower())

        # Configuración de concurrencia y CUDA Streams
        self.is_cuda = ("cuda" in str(self.device).lower()) and is_cuda_available()
        # Se omite ThreadPoolExecutor y CUDA Streams en GPU para evitar colisión con el profiler interno y CUDA Graphs de Ultralytics/TensorRT
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="YOLOInfer") if not self.is_cuda else None

        logger.info(
            f"[YOLODetector] Inicializando motor — Formato: {self.format.upper()} | Dispositivo: {self.device.upper()} | FP16: {self.half} | CUDA Nativo: {self.is_cuda}"
        )
        self._load_models()
        logger.info("[YOLODetector] ✅ Los 3 modelos (Keypoints, Jugadores, Balón) cargados en memoria.")

    def _load_models(self):
        """Carga los tres modelos según el formato configurado."""
        for name, path in [
            ("Keypoints", self.keypoint_model_path),
            ("Jugadores", self.player_model_path),
            ("Balón", self.ball_model_path),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"[YOLODetector] Modelo '{name}' no encontrado en: {path}. "
                    f"Asegúrese de ubicar los pesos en la carpeta models/."
                )

        if self.format == "engine":
            # TensorRT Engine (.engine) nativo para GPU
            self.keypoint_model = YOLO(str(self.keypoint_model_path), task="pose")
            self.player_model = YOLO(str(self.player_model_path), task="detect")
            self.ball_model = YOLO(str(self.ball_model_path), task="detect")
        elif self.format == "pt":
            self.keypoint_model = YOLO(str(self.keypoint_model_path))
            self.player_model = YOLO(str(self.player_model_path))
            self.ball_model = YOLO(str(self.ball_model_path))
            if self.half and "cuda" in str(self.device).lower():
                try:
                    self.keypoint_model.to(self.device)
                    self.player_model.to(self.device)
                    self.ball_model.to(self.device)
                    if hasattr(self.keypoint_model, "model") and self.keypoint_model.model is not None:
                        self.keypoint_model.model.half()
                    if hasattr(self.player_model, "model") and self.player_model.model is not None:
                        self.player_model.model.half()
                    if hasattr(self.ball_model, "model") and self.ball_model.model is not None:
                        self.ball_model.model.half()
                except Exception as e:
                    logger.debug(f"[YOLODetector] Inicialización FP16 omitida: {e}")
        else:
            self.keypoint_model = YOLO(str(self.keypoint_model_path), task="pose")
            self.player_model = YOLO(str(self.player_model_path), task="detect")
            self.ball_model = YOLO(str(self.ball_model_path), task="detect")

    def infer_keypoints(self, frame_bgr: np.ndarray, conf: float = 0.35) -> Any:
        """Inferencia de keypoints del campo con YOLO Pose."""
        return self.keypoint_model.predict(
            source=frame_bgr,
            conf=conf,
            device=self.device,
            half=self.half,
            verbose=False,
            save=False,
        )[0]

    def infer_players(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.40,
        is_video: bool = True,
        persist: bool = True,
        classes: Optional[List[int]] = None,
    ) -> Any:
        """Inferencia y seguimiento de jugadores con BoT-SORT restringido a clases válidas (1: goalkeeper, 2: player)."""
        if classes is None:
            cfg_det = getattr(self.config, "detection", {})
            classes = cfg_det.get("player_filter_classes", [1, 2])

        if is_video:
            tracker_file = str(self.tracker_config_path) if self.tracker_config_path.exists() else "botsort.yaml"
            try:
                return self.player_model.track(
                    source=frame_bgr,
                    conf=conf,
                    persist=persist,
                    classes=classes,
                    tracker=tracker_file,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
            except Exception as e:
                logger.warning(f"[YOLODetector] Error en tracker BoT-SORT: {e}. Usando predict clásico.")
                return self.player_model.predict(
                    source=frame_bgr,
                    conf=conf,
                    classes=classes,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
        else:
            return self.player_model.predict(
                source=frame_bgr,
                conf=conf,
                classes=classes,
                device=self.device,
                half=self.half,
                verbose=False,
            )[0]

    def infer_ball(self, frame_bgr: np.ndarray, conf: float = 0.40) -> Any:
        """Inferencia de balón de fútbol."""
        return self.ball_model.predict(
            source=frame_bgr,
            conf=conf,
            device=self.device,
            half=self.half,
            verbose=False,
            save=False,
        )[0]

    def infer(
        self,
        frame_bgr: np.ndarray,
        conf_kp: float = 0.35,
        conf_player: float = 0.40,
        conf_ball: float = 0.40,
        is_video: bool = True,
        player_classes: Optional[List[int]] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Ejecuta inferencia de los 3 modelos sobre el frame.
        En GPU (CUDA) se ejecuta secuencialmente para máxima estabilidad y compatibilidad con TensorRT y CUDA Graphs.
        """
        if self.is_cuda or self._executor is None:
            kp_res = self.infer_keypoints(frame_bgr, conf_kp)
            pl_res = self.infer_players(frame_bgr, conf_player, is_video, True, player_classes)
            ba_res = self.infer_ball(frame_bgr, conf_ball)
            return kp_res, pl_res, ba_res

        fut_kp = self._executor.submit(self.infer_keypoints, frame_bgr, conf_kp)
        fut_pl = self._executor.submit(self.infer_players, frame_bgr, conf_player, is_video, True, player_classes)
        fut_ba = self._executor.submit(self.infer_ball, frame_bgr, conf_ball)

        return fut_kp.result(), fut_pl.result(), fut_ba.result()

    def infer_keypoints_and_ball(
        self,
        frame_bgr: np.ndarray,
        conf_kp: float = 0.35,
        conf_ball: float = 0.40,
    ) -> Tuple[Any, Any]:
        """Inferencia de keypoints y balón cuando se salta el stride de jugadores."""
        if self.is_cuda or self._executor is None:
            kp_res = self.infer_keypoints(frame_bgr, conf_kp)
            ba_res = self.infer_ball(frame_bgr, conf_ball)
            return kp_res, ba_res

        fut_kp = self._executor.submit(self.infer_keypoints, frame_bgr, conf_kp)
        fut_ba = self._executor.submit(self.infer_ball, frame_bgr, conf_ball)

        return fut_kp.result(), fut_ba.result()

    def close(self):
        """Libera el pool de hilos."""
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=False)


# Alias para compatibilidad con código existente
DirectInferenceEngine = YOLODetector
