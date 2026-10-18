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
