"""
Módulo de Métricas y Asociación Espacial: DIoU (Distance-IoU) y Algoritmo de Coincidencia.
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_diou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    r"""
    Calcula la matriz de costo DIoU entre dos conjuntos de bounding boxes [x1, y1, x2, y2].
    Retorna matriz de costos C donde:
    \mathcal{C}_{i,j} = 1 - \text{DIoU}(A_i, B_j) \in [0, 2]
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    # Coordenadas de intersección
    x1_inter = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1_inter = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2_inter = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2_inter = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    w_inter = np.maximum(0.0, x2_inter - x1_inter)
    h_inter = np.maximum(0.0, y2_inter - y1_inter)
    area_inter = w_inter * h_inter

    # Áreas de las cajas individuales
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    area_union = area_a[:, None] + area_b[None, :] - area_inter
    iou = area_inter / np.maximum(area_union, 1e-6)

    # Centros de masa de las cajas
    cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) / 2.0
    cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2.0
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0

    # Distancia euclídea al cuadrado entre centros (\rho^2)
    rho2 = (cx_a[:, None] - cx_b[None, :]) ** 2 + (cy_a[:, None] - cy_b[None, :]) ** 2

    # Caja mínima envolvente (enclosing box c)
    x1_c = np.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1_c = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2_c = np.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2_c = np.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    c2 = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2
    c2 = np.maximum(c2, 1e-4)

    # DIoU = IoU - (\rho^2 / c^2)
    diou = iou - (rho2 / c2)

    # Convertir a costo (1 - DIoU) donde menor costo significa mejor coincidencia
    cost_matrix = 1.0 - diou

    # Sanitización estricta para evitar NaN/Inf en solvers como scipy.optimize.linear_sum_assignment
    cost_matrix = np.nan_to_num(cost_matrix, nan=2.0, posinf=2.0, neginf=0.0)
    cost_matrix = np.clip(cost_matrix, 0.0, 2.0)
    return cost_matrix


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve el problema de asignación lineal óptima con umbral de descarte.
    Retorna (matches, unmatched_a, unmatched_b).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    # Asegurar valores estrictamente finitos y numéricos
    clean_cost = np.nan_to_num(cost_matrix, nan=1e5, posinf=1e5, neginf=0.0).astype(np.float64)

    try:
        row_ind, col_ind = linear_sum_assignment(clean_cost)
    except ValueError:
        # Fallback en caso de inconsistencia dimensional o numérico inesperado
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return (
        np.array(matches, dtype=int).reshape(-1, 2),
        np.array(unmatched_a, dtype=int),
        np.array(unmatched_b, dtype=int),
    )
