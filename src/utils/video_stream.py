"""
Módulo de Streaming de Video Asíncrono y de Alto Rendimiento
Contiene:
- ThreadedVideoReader: Prefetching y decodificación multihilo en buffer desacoplado.
- AsyncVideoWriter: Escritura en disco desacoplada en segundo plano para evitar bloqueos I/O.
"""

import queue
import threading
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np

logger = logging.getLogger("src.utils.video_stream")


class ThreadedVideoReader:
    """
    Lector de video multihilo con buffer desacoplado y profiling de decodificación.
    Evita que el bucle principal de inferencia (CPU/GPU) se bloquee esperando lectura de disco
    y decodificación FFmpeg/OpenCV frame a frame.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        buffer_size: int = 64,
        timeout: float = 10.0,
    ):
        self.video_path = str(video_path)
        self.buffer_size = buffer_size
        self.timeout = timeout

        self._cap = cv2.VideoCapture(self.video_path)
        self._is_opened = self._cap.isOpened()

        if not self._is_opened:
            logger.error(f"[ThreadedVideoReader] ❌ No se pudo abrir el archivo de video: {self.video_path}")
            self.width = 0
            self.height = 0
            self.fps = 25.0
            self.total_frames = 0
            self._released = True
            return

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.fps = raw_fps if raw_fps > 0 else 25.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._stopped = False
        self._released = False
        self._reached_eof = False
        self._lock = threading.Lock()

        # Profiling de tiempos
        self._total_decode_time: float = 0.0
        self._frames_decoded: int = 0
        self._total_consumer_wait_time: float = 0.0
        self._frames_consumed: int = 0

        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ThreadedVideoReader",
        )
        self._thread.start()

        logger.info(
            f"[ThreadedVideoReader] 🚀 Iniciado → {self.video_path} "
            f"({self.width}x{self.height} @ {self.fps:.1f}fps, total={self.total_frames}, buffer={buffer_size})"
        )

    def isOpened(self) -> bool:
        """Compatibilidad con la API de cv2.VideoCapture."""
        if self._released:
            return False
        return self._is_opened and (not self._stopped or not self._queue.empty())

    def is_opened(self) -> bool:
        return self.isOpened()

    def get(self, prop_id: int) -> float:
        """Delega consulta de propiedades a VideoCapture subyacente."""
        if self._cap is not None:
            return self._cap.get(prop_id)
        return 0.0

    def _worker_loop(self):
        """Loop continuo del hilo lector: lee, decodifica y encola frames."""
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            decode_time = time.perf_counter() - t0

            if not ret or frame is None:
                self._reached_eof = True
                self._stopped = True
                while not self._stop_event.is_set():
                    try:
                        self._queue.put((False, None), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                break

            with self._lock:
                self._total_decode_time += decode_time
                self._frames_decoded += 1

            while not self._stop_event.is_set():
                try:
                    self._queue.put((True, frame), timeout=0.1)
                    break
                except queue.Full:
                    continue

    def read(self, timeout: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retorna (ret, frame_bgr) consumiendo del buffer desacoplado.
        """
        if self._released:
            return False, None

        to = timeout if timeout is not None else self.timeout
        t0 = time.perf_counter()

        try:
            ret, frame = self._queue.get(timeout=to)
            wait_time = time.perf_counter() - t0

            with self._lock:
                self._total_consumer_wait_time += wait_time
                if ret and frame is not None:
                    self._frames_consumed += 1

            if not ret or frame is None:
                self._stopped = True
                return False, None

            return True, frame

        except queue.Empty:
            if self._reached_eof or self._stopped:
                return False, None
            logger.warning(
                f"[ThreadedVideoReader] ⚠️ Timeout ({to}s) esperando frame del buffer de decodificación."
            )
            return False, None

    def release(self):
        """Detiene el hilo lector, drena la cola y libera el recurso."""
        if self._released:
            return

        self._stop_event.set()
        self._stopped = True

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._released = True

        logger.info(
            f"[ThreadedVideoReader] ✅ Cerrado. Decodificados: {self.frames_decoded} frames | "
            f"Consumidos: {self.frames_consumed} frames | "
            f"Tiempo decode worker: {self.total_decode_time:.2f}s ({self.avg_decode_time_ms:.2f} ms/frame, {self.decode_fps:.1f} FPS) | "
            f"Espera consumidor: {self.total_consumer_wait_time:.3f}s"
        )

    @property
    def total_decode_time(self) -> float:
        with self._lock:
            return self._total_decode_time

    @property
    def frames_decoded(self) -> int:
        with self._lock:
            return self._frames_decoded

    @property
    def frames_consumed(self) -> int:
        with self._lock:
            return self._frames_consumed

    @property
    def total_consumer_wait_time(self) -> float:
        with self._lock:
            return self._total_consumer_wait_time

    @property
    def avg_decode_time_ms(self) -> float:
        with self._lock:
            if self._frames_decoded == 0:
                return 0.0
            return (self._total_decode_time / self._frames_decoded) * 1000.0

    @property
    def avg_consumer_wait_ms(self) -> float:
        with self._lock:
            if self._frames_consumed == 0:
                return 0.0
            return (self._total_consumer_wait_time / self._frames_consumed) * 1000.0

    @property
    def decode_fps(self) -> float:
        with self._lock:
            if self._total_decode_time == 0.0:
                return 0.0
            return self._frames_decoded / self._total_decode_time

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_decode_time_s": round(self._total_decode_time, 4),
                "frames_decoded": self._frames_decoded,
                "frames_consumed": self._frames_consumed,
                "total_consumer_wait_s": round(self._total_consumer_wait_time, 4),
                "avg_decode_time_ms": round(self.avg_decode_time_ms, 2),
                "avg_consumer_wait_ms": round(self.avg_consumer_wait_ms, 2),
                "decode_fps": round(self.decode_fps, 2),
                "buffer_size": self.buffer_size,
            }


# Alias para total retrocompatibilidad
ThreadedOpenCVReader = ThreadedVideoReader


class AsyncVideoWriter:
    """
    VideoWriter multihilo con cola en segundo plano.
    - El bucle principal de inferencia llama a `write(frame)` sin bloquearse por I/O.
    - Un hilo de fondo consume la cola y escribe al disco.
    """

    def __init__(
        self,
        output_path: str,
        fourcc: int,
        fps: float,
        size: Tuple[int, int],
        buffer_size: int = 60,
    ):
        self.output_path = str(output_path)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(self.output_path, fourcc, fps, size)
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread = threading.Thread(target=self._write_loop, daemon=True, name="AsyncVideoWriter")
        self._thread.start()
        self._frame_count = 0
        logger.info(f"[AsyncVideoWriter] Iniciado → {output_path} ({size[0]}x{size[1]} @ {fps:.1f}fps, buffer={buffer_size})")

    def write(self, frame: np.ndarray):
        """Encola frame para escritura asíncrona. Bloquea si la cola está llena para no perder cuadros."""
        self._queue.put(frame, block=True)
        self._frame_count += 1

    def _write_loop(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._writer.write(frame)

    def release(self):
        """Vacía la cola pendiente, espera al hilo y cierra el VideoWriter."""
        self._queue.put(None)
        self._thread.join()
        self._writer.release()
        logger.info(f"[AsyncVideoWriter] ✅ Cerrado. Total frames escritos: {self._frame_count}")

    @property
    def pending_frames(self) -> int:
        return self._queue.qsize()
