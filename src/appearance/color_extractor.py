"""
Módulo de Extracción de Color de Camisetas (HSV / CIELAB)
Extrae el color representativo del torso superior descartando césped y tonos de piel.
"""

from typing import Tuple, Optional, List
import cv2
import numpy as np


def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    """Convierte una tupla (B, G, R) a string hexadecimal #RRGGBB."""
    b, g, r = bgr
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def lab_to_bgr(lab: np.ndarray) -> Tuple[int, int, int]:
    """Convierte un vector CIELAB [L, a, b] a tupla BGR uint8."""
    lab_pixel = np.uint8([[lab]])
    bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


class JerseyColorExtractor:
    """
    Extractor de color de camisetas para jugadores de fútbol.
    - Selecciona el ROI del torso superior (15% a 50% vertical, 15% a 85% horizontal).
    - Aplica máscaras HSV para filtrar césped y piel.
    - Calcula la mediana recortada (trimmed median) en espacio CIELAB.
    """

    def __init__(
        self,
        grass_hsv_lower: Tuple[int, int, int] = (30, 40, 40),
        grass_hsv_upper: Tuple[int, int, int] = (85, 255, 255),
        skin_hsv_lower: Tuple[int, int, int] = (0, 20, 70),
        skin_hsv_upper: Tuple[int, int, int] = (25, 255, 255),
    ):
        self.grass_lower = np.array(grass_hsv_lower, dtype=np.uint8)
        self.grass_upper = np.array(grass_hsv_upper, dtype=np.uint8)
        self.skin_lower = np.array(skin_hsv_lower, dtype=np.uint8)
        self.skin_upper = np.array(skin_hsv_upper, dtype=np.uint8)

    def extract_jersey_color(
        self,
        frame_bgr: np.ndarray,
        bbox: List[float],
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
        """
        Retorna (lab_vector, bgr_color) o None si el recorte es inválido.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        if bw < 8 or bh < 16:
            return None

        # ROI: Torso superior
        ry1 = max(0, int(y1 + 0.15 * bh))
        ry2 = min(h_img, int(y1 + 0.50 * bh))
        rx1 = max(0, int(x1 + 0.15 * bw))
        rx2 = min(w_img, int(x2 - 0.15 * bw))

        if ry2 <= ry1 or rx2 <= rx1:
            return None

        crop = frame_bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None

        # Máscaras HSV para descartar césped y piel
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv_crop, self.grass_lower, self.grass_upper)
        skin_mask = cv2.inRange(hsv_crop, self.skin_lower, self.skin_upper)
        invalid_mask = cv2.bitwise_or(grass_mask, skin_mask)
        valid_mask = cv2.bitwise_not(invalid_mask)

        # Conversión a CIELAB
        lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        valid_pixels = lab_crop[valid_mask > 0]

        if len(valid_pixels) < 10:
            # Fallback al centro si la máscara eliminó casi todo
            ch, cw = crop.shape[:2]
            valid_pixels = lab_crop[int(ch * 0.2):int(ch * 0.8), int(cw * 0.2):int(cw * 0.8)].reshape(-1, 3)

        if len(valid_pixels) == 0:
            return None

        # Mediana recortada en percentiles 10-90
        p10 = np.percentile(valid_pixels, 10, axis=0)
        p90 = np.percentile(valid_pixels, 90, axis=0)
        in_range = np.all((valid_pixels >= p10) & (valid_pixels <= p90), axis=1)
        trimmed = valid_pixels[in_range] if np.any(in_range) else valid_pixels

        lab_vector = np.median(trimmed, axis=0).astype(np.float32)
        bgr_color = lab_to_bgr(lab_vector)

        return lab_vector, bgr_color


class JerseyDescriptorExtractor:
    """
    Descriptor interno de apariencia del torso (solo para DIFERENCIAR grupos; no se exporta).

    - ROI del torso superior reducido a un tamaño fijo y pequeño (barato).
    - Césped descartado de forma ADAPTATIVA: el tono dominante del campo se estima por frame
      (cada `grass_refresh_frames`) en lugar de usar un rango HSV fijo.
    - Sin máscara de piel fija: el rango de piel (H 0-25) coincidía con camisetas rojas/naranjas
      y las destruía.
    - Histograma LAB cuantizado (L grueso para tolerar sombras; a/b más finos) normalizado y con
      raíz cuadrada (distancia euclídea == distancia de Hellinger). Maneja camisetas a rayas
      mejor que una mediana de color.
    """

    L_BINS = 4
    A_BINS = 8
    B_BINS = 8
    DESC_DIM = L_BINS * A_BINS * B_BINS

    def __init__(
        self,
        crop_size: Tuple[int, int] = (16, 24),
        grass_refresh_frames: int = 30,
        grass_hue_tolerance: int = 14,
        grass_min_saturation: int = 35,
        default_grass_hue: float = 60.0,
    ):
        self.crop_size = crop_size
        self.grass_refresh_frames = grass_refresh_frames
        self.grass_hue_tolerance = grass_hue_tolerance
        self.grass_min_saturation = grass_min_saturation
        self._grass_hue: Optional[float] = None
        self._default_grass_hue = default_grass_hue
        self._last_field_frame = -10 ** 9

    @property
    def grass_hue(self) -> float:
        return self._grass_hue if self._grass_hue is not None else self._default_grass_hue

    def update_field_model(self, frame_bgr: np.ndarray, frame_idx: int) -> None:
        """Estima el tono dominante del césped (pico del histograma H en 25-95)."""
        if self._grass_hue is not None and (frame_idx - self._last_field_frame) < self.grass_refresh_frames:
            return
        self._last_field_frame = frame_idx

        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sel = (hsv[..., 1] >= 40) & (hsv[..., 2] >= 40)
        n_sel = int(sel.sum())
        if n_sel == 0:
            return

        hist = np.bincount(hsv[..., 0][sel].ravel(), minlength=180).astype(np.float32)
        smooth = np.convolve(hist, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        lo, hi = 25, 95
        peak = lo + int(np.argmax(smooth[lo:hi]))
        share = float(hist[max(0, peak - 7):peak + 8].sum()) / float(n_sel)
        if share < 0.15:
            return

        if self._grass_hue is None:
            self._grass_hue = float(peak)
        else:
            self._grass_hue = 0.7 * self._grass_hue + 0.3 * float(peak)

    def extract(self, frame_bgr: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Retorna el descriptor (DESC_DIM,) float32 del torso, o None si el recorte es inválido."""
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw < 8 or bh < 16:
            return None

        ry1 = max(0, int(y1 + 0.15 * bh))
        ry2 = min(h_img, int(y1 + 0.50 * bh))
        rx1 = max(0, int(x1 + 0.15 * bw))
        rx2 = min(w_img, int(x2 - 0.15 * bw))
        if ry2 <= ry1 or rx2 <= rx1:
            return None

        crop = frame_bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None

        small = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)

        hue = hsv[..., 0].astype(np.int16)
        grass = (np.abs(hue - self.grass_hue) <= self.grass_hue_tolerance) & (hsv[..., 1] >= self.grass_min_saturation)
        valid = ~grass
        if int(valid.sum()) < 0.2 * valid.size:
            # Casi todo parece césped (p. ej. camiseta verde): usar todos los píxeles
            valid = np.ones_like(valid)

        px = lab[valid].astype(np.int32)
        lb = (px[:, 0] * self.L_BINS) >> 8
        ab = (px[:, 1] * self.A_BINS) >> 8
        bb = (px[:, 2] * self.B_BINS) >> 8
        idx = (lb * self.A_BINS + ab) * self.B_BINS + bb

        hist = np.bincount(idx, minlength=self.DESC_DIM).astype(np.float32)
        total = float(hist.sum())
        if total <= 0:
            return None
        hist /= total
        return np.sqrt(hist)
