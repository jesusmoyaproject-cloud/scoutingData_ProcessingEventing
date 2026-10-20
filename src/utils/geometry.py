"""
Módulo de Utilidades Geométricas y Transformaciones de Coordenadas
Soporta conversiones entre espacio de imagen (píxeles) y coordenadas métricas de cancha (metros FIFA).
"""

from typing import Dict, Tuple, List, Optional, Any
import cv2
import numpy as np


def pixel_to_field(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Transforma coordenadas de píxeles (espacio de imagen) a coordenadas métricas de cancha en metros.
    """
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def field_to_pixel(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Transforma coordenadas de cancha en metros a coordenadas de imagen (píxeles).
    """
    _, H_inv = cv2.invert(H)
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H_inv)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Calcula la distancia euclidiana entre dos puntos (x1, y1) y (x2, y2)."""
    return float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))


def bbox_to_center_bottom(bbox: List[float]) -> Tuple[float, float]:
    """
    Calcula el punto de contacto con el suelo del jugador (centro en X, base inferior en Y).
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return float((x1 + x2) / 2.0), float(y2)


def bbox_to_center(bbox: List[float]) -> Tuple[float, float]:
    """Calcula el centro geométrico del bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def compute_pitch_keypoint_coordinates(
    length: float = 105.0,
    width: float = 68.0,
    penalty_box_length: float = 16.5,
    penalty_box_width: float = 40.32,
    goal_box_length: float = 5.5,
    goal_box_width: float = 18.32,
    centre_circle_radius: float = 9.15,
    penalty_spot_distance: float = 11.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Genera las coordenadas métricas (en metros) de los 30 keypoints canónicos del campo.
    El origen (0, 0) es la esquina superior izquierda.
    X aumenta longitudinalmente (largo), Y aumenta hacia abajo (ancho).
    """
    half_len = length / 2.0
    half_wid = width / 2.0

    dx_penalty = penalty_box_length - penalty_spot_distance
    dy_arc = float(np.sqrt(max(0.0, centre_circle_radius**2 - dx_penalty**2)))

    return {
        # 1-6: Línea lateral izquierda (x=0)
        "01": (0.0, 0.0),
        "02": (0.0, half_wid - penalty_box_width / 2.0),
        "03": (0.0, half_wid - goal_box_width / 2.0),
        "04": (0.0, half_wid + goal_box_width / 2.0),
        "05": (0.0, half_wid + penalty_box_width / 2.0),
        "06": (0.0, width),

        # 7-8: Esquinas del área chica izquierda
        "07": (goal_box_length, half_wid - goal_box_width / 2.0),
        "08": (goal_box_length, half_wid + goal_box_width / 2.0),

        # 10-13: Esquinas del área grande izquierda e intersecciones con la medialuna
        "10": (penalty_box_length, half_wid - penalty_box_width / 2.0),
        "11": (penalty_box_length, half_wid - dy_arc),
        "12": (penalty_box_length, half_wid + dy_arc),
        "13": (penalty_box_length, half_wid + penalty_box_width / 2.0),

        # 15-18: Línea central (x = length / 2)
        "15": (half_len, 0.0),
        "16": (half_len, half_wid - centre_circle_radius),
        "17": (half_len, half_wid + centre_circle_radius),
        "18": (half_len, width),

        # 20-23: Esquinas del área grande derecha e intersecciones con la medialuna
        "20": (length - penalty_box_length, half_wid - penalty_box_width / 2.0),
        "21": (length - penalty_box_length, half_wid - dy_arc),
        "22": (length - penalty_box_length, half_wid + dy_arc),
        "23": (length - penalty_box_length, half_wid + penalty_box_width / 2.0),

        # 25-26: Esquinas del área chica derecha
        "25": (length - goal_box_length, half_wid - goal_box_width / 2.0),
        "26": (length - goal_box_length, half_wid + goal_box_width / 2.0),

        # 27-32: Línea lateral derecha (x = length)
        "27": (length, 0.0),
        "28": (length, half_wid - penalty_box_width / 2.0),
        "29": (length, half_wid - goal_box_width / 2.0),
        "30": (length, half_wid + goal_box_width / 2.0),
        "31": (length, half_wid + penalty_box_width / 2.0),
        "32": (length, width),

        # 14 & 19: Intersecciones círculo central con eje transversal
        "14": (half_len - centre_circle_radius, half_wid),
        "19": (half_len + centre_circle_radius, half_wid),
    }


def get_pitch_template_specification(
    length: float = 105.0,
    width: float = 68.0,
    penalty_box_length: float = 16.5,
    penalty_box_width: float = 40.32,
    goal_box_length: float = 5.5,
    goal_box_width: float = 18.32,
    centre_circle_radius: float = 9.15,
    penalty_spot_distance: float = 11.0,
) -> Dict[str, Any]:
    """
    Retorna la especificación geométrica completa de la cancha y sus elementos vectoriales
    en metros para consumo directo por frontends 2D (Canvas / SVG / WebGL).
    """
    half_len = round(length / 2.0, 2)
    half_wid = round(width / 2.0, 2)
    pb_y1 = round(half_wid - penalty_box_width / 2.0, 2)
    pb_y2 = round(half_wid + penalty_box_width / 2.0, 2)
    gb_y1 = round(half_wid - goal_box_width / 2.0, 2)
    gb_y2 = round(half_wid + goal_box_width / 2.0, 2)

    return {
        "dimensions_m": {
            "length": float(length),
            "width": float(width),
            "penalty_box_length": float(penalty_box_length),
            "penalty_box_width": float(penalty_box_width),
            "goal_box_length": float(goal_box_length),
            "goal_box_width": float(goal_box_width),
            "centre_circle_radius": float(centre_circle_radius),
            "penalty_spot_distance": float(penalty_spot_distance),
        },
        "coordinate_system": {
            "origin": "top_left",
            "x_axis": "longitudinal_0_to_length",
            "y_axis": "transversal_0_to_width",
            "unit": "meters",
        },
        "vector_elements": {
            "boundary": [[0.0, 0.0], [float(length), 0.0], [float(length), float(width)], [0.0, float(width)], [0.0, 0.0]],
            "halfway_line": [[half_len, 0.0], [half_len, float(width)]],
            "center_circle": {
                "center": [half_len, half_wid],
                "radius": float(centre_circle_radius),
            },
            "center_spot": [half_len, half_wid],
            "penalty_box_left": [[0.0, pb_y1], [float(penalty_box_length), pb_y1], [float(penalty_box_length), pb_y2], [0.0, pb_y2]],
            "penalty_box_right": [[float(length), pb_y1], [round(length - penalty_box_length, 2), pb_y1], [round(length - penalty_box_length, 2), pb_y2], [float(length), pb_y2]],
            "goal_box_left": [[0.0, gb_y1], [float(goal_box_length), gb_y1], [float(goal_box_length), gb_y2], [0.0, gb_y2]],
            "goal_box_right": [[float(length), gb_y1], [round(length - goal_box_length, 2), gb_y1], [round(length - goal_box_length, 2), gb_y2], [float(length), gb_y2]],
            "penalty_spots": [
                [float(penalty_spot_distance), half_wid],
                [round(length - penalty_spot_distance, 2), half_wid],
            ],
        },
    }

