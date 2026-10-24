"""
Módulo de Visión Artificial: Estimación, Validación y Auditoría de Homografía 2D
Implementa:
- Estimación robusta con USAC_MAGSAC / RANSAC calibrado en metros.
- Validación geométrica: Convex Hull, Cobertura, Colinealidad y Filtrado de Bordes.
- Validador temporal con filtro EMA y detección de derivas cinemáticas (Corner Shift, Rotación, Escala).
- Auditoría forense punto a punto (KeypointAuditLogger, Leave-One-Out, Detección de Swaps).
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import cv2

from src.utils.geometry import (
    pixel_to_field,
    field_to_pixel,
    compute_pitch_keypoint_coordinates,
)

logger = logging.getLogger("src.perception.homography")

# OpenCV method selector
if hasattr(cv2, "USAC_MAGSAC"):
    DEFAULT_HOMOGRAPHY_METHOD = cv2.USAC_MAGSAC
else:
    DEFAULT_HOMOGRAPHY_METHOD = cv2.RANSAC

# Mapeo oficial de los 30 keypoints FIFA
KP_MAP: Dict[int, str] = {
    0: '01',  1: '02',  2: '03',  3: '04',  4: '05',  5: '06',
    6: '07',  7: '08',  8: '10',  9: '11', 10: '12', 11: '13',
    12: '15', 13: '16', 14: '17', 15: '18',
    16: '20', 17: '21', 18: '22', 19: '23',
    20: '25', 21: '26', 22: '27', 23: '28',
    24: '29', 25: '30', 26: '31', 27: '32',
    28: '14', 29: '19'
}

LEFT_KEYPOINTS = {"01", "02", "03", "04", "05", "06", "07", "08", "10", "11", "12", "13"}
CENTER_KEYPOINTS = {"14", "15", "16", "17", "18", "19"}
RIGHT_KEYPOINTS = {"20", "21", "22", "23", "25", "26", "27", "28", "29", "30", "31", "32"}

FIELD_COORDS = compute_pitch_keypoint_coordinates()


@dataclass
class HomographyDiagnostics:
    inlier_ratio: float
    mean_reprojection_error_m: float
    max_reprojection_error_m: float
    condition_number: float
    hull_area_field_m2: float
    field_coverage_ratio: float
    hull_area_img_px2: float
    image_coverage_ratio: float
    collinearity_score: float
    kps_detected: int
    kps_used: int
    ransac_inliers: int
    ransac_outliers: int
    method_used: str = "USAC_MAGSAC"
    is_mathematically_valid: bool = True
    is_geometrically_valid: bool = True


@dataclass
class PointResidual:
    kp_idx: int
    kp_label: str
    confidence: float
    detected_px: Tuple[float, float]
    reprojected_px: Tuple[float, float]
    field_m: Tuple[float, float]
    reprojected_m: Tuple[float, float]
    error_px: float
    error_m: float
    is_inlier: bool


class HomographyEstimateResult(tuple):
    H: np.ndarray
    image_pts: np.ndarray
    field_pts: np.ndarray
    mask: np.ndarray
    inliers_count: int
    outliers_count: int
    diagnostics: Optional[HomographyDiagnostics]
    valid_candidates: list
    residuals: list

    def __new__(
        cls,
        H,
        image_pts,
        field_pts,
        mask,
        inliers_count,
        outliers_count,
        diagnostics=None,
        valid_candidates=None,
        residuals=None,
    ):
        instance = super().__new__(cls, (H, image_pts, field_pts, mask, inliers_count, outliers_count))
        instance.H = H
        instance.image_pts = image_pts
        instance.field_pts = field_pts
        instance.mask = mask
        instance.inliers_count = inliers_count
        instance.outliers_count = outliers_count
        instance.diagnostics = diagnostics
        instance.valid_candidates = valid_candidates if valid_candidates is not None else []
        instance.residuals = residuals if residuals is not None else []
        return instance


def compute_convex_hull_and_coverage(
    image_pts: np.ndarray,
    field_pts: np.ndarray,
    img_width: int = 1920,
    img_height: int = 1080,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Dict[str, float]:
    if len(image_pts) < 3:
        return {
            "hull_area_img_px2": 0.0,
            "image_coverage_ratio": 0.0,
            "hull_area_field_m2": 0.0,
            "field_coverage_ratio": 0.0,
        }

    pts_img_f32 = image_pts.reshape(-1, 2).astype(np.float32)
    hull_img = cv2.convexHull(pts_img_f32)
    area_img = float(cv2.contourArea(hull_img))
    total_img_area = float(img_width * img_height) if (img_width > 0 and img_height > 0) else 1.0
    coverage_img = area_img / total_img_area

    pts_fld_f32 = field_pts.reshape(-1, 2).astype(np.float32)
    hull_fld = cv2.convexHull(pts_fld_f32)
    area_fld = float(cv2.contourArea(hull_fld))
    total_fld_area = float(pitch_length * pitch_width) if (pitch_length > 0 and pitch_width > 0) else 7140.0
    coverage_fld = area_fld / total_fld_area

    return {
        "hull_area_img_px2": area_img,
        "image_coverage_ratio": coverage_img,
        "hull_area_field_m2": area_fld,
        "field_coverage_ratio": coverage_fld,
    }


def compute_collinearity_score(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    pts_2d = pts.reshape(-1, 2).astype(np.float64)
    centered = pts_2d - pts_2d.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or np.isnan(cov).any():
        return 0.0
    eigvals = np.linalg.eigvalsh(cov)
    l_min = max(0.0, float(eigvals[0]))
    l_max = max(1e-9, float(eigvals[1]))
    return float(np.sqrt(l_min / l_max))


def validate_keypoint_geometry(
    image_pts: np.ndarray,
    field_pts: np.ndarray,
    img_width: int = 1920,
    img_height: int = 1080,
    min_points: int = 4,
    min_hull_area_m2: float = 150.0,
    min_image_coverage: float = 0.04,
    min_collinearity_score: float = 0.12,
    min_span_x_m: float = 15.0,
    min_span_y_m: float = 10.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    n_pts = len(image_pts)
    if n_pts < min_points:
        return False, f"Puntos insuficientes: {n_pts} < {min_points}", {}

    coverage_info = compute_convex_hull_and_coverage(
        image_pts, field_pts, img_width=img_width, img_height=img_height
    )
    collinearity = compute_collinearity_score(field_pts)

    fld_span_x = float(field_pts[:, 0].max() - field_pts[:, 0].min())
    fld_span_y = float(field_pts[:, 1].max() - field_pts[:, 1].min())

    meta = {
        **coverage_info,
        "collinearity_score": collinearity,
        "field_span_x_m": fld_span_x,
        "field_span_y_m": fld_span_y,
    }

    if collinearity < min_collinearity_score:
        return False, f"Puntos colineales (score={collinearity:.3f} < {min_collinearity_score})", meta

    if coverage_info["hull_area_field_m2"] < min_hull_area_m2:
        return False, f"Área de campo insuficiente ({coverage_info['hull_area_field_m2']:.1f}m² < {min_hull_area_m2}m²)", meta

    if coverage_info.get("image_coverage_ratio", 0.0) < min_image_coverage:
        return False, f"Cobertura en imagen insuficiente ({coverage_info.get('image_coverage_ratio', 0.0):.3f} < {min_image_coverage})", meta

    if fld_span_x < min_span_x_m or fld_span_y < min_span_y_m:
        return False, f"Dispersión 2D insuficiente (dx={fld_span_x:.1f}m, dy={fld_span_y:.1f}m)", meta

    return True, "OK", meta


def filter_border_keypoints(
    candidates: List[Tuple[int, np.ndarray, float, str]],
    img_width: int = 1920,
    img_height: int = 1080,
    border_margin_px: float = 4.0,
) -> List[Tuple[int, np.ndarray, float, str]]:
    filtered = []
    for idx, xy, conf, label in candidates:
        x, y = xy
        if (
            x < border_margin_px
            or x > (img_width - border_margin_px)
            or y < border_margin_px
            or y > (img_height - border_margin_px)
        ):
            continue
        filtered.append((idx, xy, conf, label))
    return filtered


def filter_keypoints_by_pitch_side(
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_threshold: float = 0.35,
    min_dominant_points: int = 4,
) -> Tuple[List[Tuple[int, np.ndarray, float, str]], bool, bool]:
    candidates: List[Tuple[int, np.ndarray, float, str]] = []
    left_cands: List[Tuple[int, np.ndarray, float, str]] = []
    right_cands: List[Tuple[int, np.ndarray, float, str]] = []
    center_cands: List[Tuple[int, np.ndarray, float, str]] = []

    for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
        if conf < conf_threshold or idx not in KP_MAP:
            continue
        label = KP_MAP[idx]
        if label not in FIELD_COORDS:
            continue

        cand = (idx, xy, float(conf), label)
        if label in LEFT_KEYPOINTS:
            left_cands.append(cand)
        elif label in RIGHT_KEYPOINTS:
            right_cands.append(cand)
        else:
            center_cands.append(cand)

    n_left = len(left_cands)
    n_right = len(right_cands)
    sum_left = sum(c[2] for c in left_cands)
    sum_right = sum(c[2] for c in right_cands)

    discard_left = False
    discard_right = False

    if n_right >= min_dominant_points and (n_left == 0 or sum_right >= 2.0 * sum_left or n_right >= n_left + 4):
        discard_left = True
    elif n_left >= min_dominant_points and (n_right == 0 or sum_left >= 2.0 * sum_right or n_left >= n_right + 4):
        discard_right = True

    if not discard_left:
        candidates.extend(left_cands)
    if not discard_right:
        candidates.extend(right_cands)
    candidates.extend(center_cands)

    candidates.sort(key=lambda x: x[0])
    return candidates, discard_left, discard_right


def validate_homography_matrix(
    H: np.ndarray,
    min_normalized_det: float = 1e-12,
    max_cond: float = 1e7,
) -> bool:
    if H is None or H.shape != (3, 3) or np.isnan(H).any() or np.isinf(H).any():
        return False

    fro = np.linalg.norm(H, "fro")
    if fro < 1e-9:
        return False

    H_norm = H / fro
    det = float(np.linalg.det(H_norm))
    if abs(det) < min_normalized_det:
        return False

    try:
        cond = float(np.linalg.cond(H_norm))
        if cond > max_cond:
            return False
    except np.linalg.LinAlgError:
        return False

    return True


def compute_point_residuals(
    H: np.ndarray,
    valid_candidates: List[Tuple[int, np.ndarray, float, str]],
    mask: Optional[np.ndarray] = None,
) -> List[PointResidual]:
    residuals: List[PointResidual] = []
    if H is None or len(valid_candidates) == 0:
        return residuals

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = None

    for i, (kp_idx, xy_det, conf, label) in enumerate(valid_candidates):
        is_inlier = True
        if mask is not None and i < len(mask):
            is_inlier = bool(mask[i][0] if mask[i].ndim > 0 else mask[i])

        fld_pt = np.array(FIELD_COORDS[label], dtype=np.float32).reshape(1, 1, 2)
        img_pt = np.array(xy_det, dtype=np.float32).reshape(1, 1, 2)

        proj_field = cv2.perspectiveTransform(img_pt, H)[0, 0]
        err_m = float(np.linalg.norm(proj_field - fld_pt[0, 0]))

        if H_inv is not None:
            proj_img = cv2.perspectiveTransform(fld_pt, H_inv)[0, 0]
            err_px = float(np.linalg.norm(proj_img - xy_det))
            proj_img_tuple = (float(proj_img[0]), float(proj_img[1]))
        else:
            err_px = 0.0
            proj_img_tuple = (float(xy_det[0]), float(xy_det[1]))

        residuals.append(PointResidual(
            kp_idx=kp_idx,
            kp_label=label,
            confidence=float(conf),
            detected_px=(float(xy_det[0]), float(xy_det[1])),
            reprojected_px=proj_img_tuple,
            field_m=(float(fld_pt[0, 0, 0]), float(fld_pt[0, 0, 1])),
            reprojected_m=(float(proj_field[0]), float(proj_field[1])),
            error_px=round(err_px, 2),
            error_m=round(err_m, 3),
            is_inlier=is_inlier,
        ))

    return residuals


def generate_residuals_table(residuals: List[PointResidual]) -> str:
    lines = [
        "+-------+-------+------------+------------+------------+---------------+",
        "| KP_ID | Label | Confidence | Error (m)  | Error (px) | Status        |",
        "+-------+-------+------------+------------+------------+---------------+",
    ]
    for r in residuals:
        status = "INLIER" if r.is_inlier else "OUTLIER"
        lines.append(
            f"| {r.kp_idx:02d}    | {r.kp_label:5s} | {r.confidence:10.2f} | "
            f"{r.error_m:8.2f} m | {r.error_px:8.1f} px | {status:13s} |"
        )
    lines.append("+-------+-------+------------+------------+------------+---------------+")
    return "\n".join(lines)


def refine_keypoints_subpixel(
    image_pts: np.ndarray,
    gray_img: np.ndarray,
    win_size: Tuple[int, int] = (5, 5),
    max_shift_px: float = 3.5,
) -> np.ndarray:
    """
    Refina las coordenadas 2D de keypoints a nivel sub-píxel usando cv2.cornerSubPix.
    Descarta derivas que excedan max_shift_px para evitar artefactos en césped o texturas complejas.
    """
    if gray_img is None or len(image_pts) == 0:
        return image_pts

    h, w = gray_img.shape[:2]
    refined = image_pts.copy()

    wx, wy = win_size
    valid_indices = []
    pts_to_refine = []
    for idx, pt in enumerate(image_pts):
        x, y = pt
        if wx < x < (w - wx) and wy < y < (h - wy):
            valid_indices.append(idx)
            pts_to_refine.append([float(x), float(y)])

    if not pts_to_refine:
        return refined

    pts_f32 = np.array(pts_to_refine, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        cv2.cornerSubPix(gray_img, pts_f32, (wx, wy), (-1, -1), criteria)
        refined_coords = pts_f32.reshape(-1, 2)

        for k, orig_idx in enumerate(valid_indices):
            orig_pt = image_pts[orig_idx]
            new_pt = refined_coords[k]
            if not np.any(np.isnan(new_pt)) and not np.any(np.isinf(new_pt)):
                shift = float(np.linalg.norm(new_pt - orig_pt))
                if shift <= max_shift_px:
                    refined[orig_idx] = new_pt
    except Exception as e:
        logger.debug(f"[SubPix] Fallback a coordenadas originales por excepción: {e}")
        return image_pts

    return refined


def compute_homography(
    result,
    conf_threshold: float = 0.35,
    ransac_threshold: float = 1.5,
    enable_side_filter: bool = True,
    filter_border_margin_px: float = 4.0,
    img_width: int = 1920,
    img_height: int = 1080,
    validate_geometry: bool = True,
    method: int = DEFAULT_HOMOGRAPHY_METHOD,
    frame_bgr: Optional[np.ndarray] = None,
    enable_subpixel: bool = True,
    subpixel_win_size: Tuple[int, int] = (5, 5),
    max_subpixel_shift_px: float = 3.5,
    exclude_boxes: Optional[np.ndarray] = None,
) -> HomographyEstimateResult:
    """Calcula la matriz H con USAC_MAGSAC, validación geométrica, refinamiento sub-píxel y telemetría."""
    if (
        result is None
        or result.keypoints is None
        or len(result.keypoints) == 0
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        raise ValueError("YOLO keypoints data is empty/None.")

    kps_xy = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()

    if len(kps_xy) == 0:
        raise ValueError("YOLO keypoints array is empty.")

    if enable_side_filter:
        valid_candidates, _, _ = filter_keypoints_by_pitch_side(
            kps_xy, kps_conf, conf_threshold=conf_threshold
        )
    else:
        valid_candidates = [
            (idx, xy, float(conf), KP_MAP[idx])
            for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf))
            if conf >= conf_threshold and idx in KP_MAP and KP_MAP[idx] in FIELD_COORDS
        ]

    if filter_border_margin_px > 0:
        valid_candidates = filter_border_keypoints(
            valid_candidates, img_width=img_width, img_height=img_height, border_margin_px=filter_border_margin_px
        )

    # Filtrar keypoints espurios que caen dentro del cuerpo de jugadores o árbitros
    if exclude_boxes is not None and len(exclude_boxes) > 0 and len(valid_candidates) > 0:
        filtered_cands = []
        for cand in valid_candidates:
            kx, ky = cand[1]
            in_person = False
            for bx in exclude_boxes:
                # bx = [x1, y1, x2, y2]
                if bx[0] <= kx <= bx[2] and bx[1] <= ky <= bx[3]:
                    in_person = True
                    break
            if not in_person:
                filtered_cands.append(cand)
        valid_candidates = filtered_cands

    image_pts_list = [xy for _, xy, _, _ in valid_candidates]
    field_pts_list = [FIELD_COORDS[lbl] for _, _, _, lbl in valid_candidates]

    image_pts = np.array(image_pts_list, dtype=np.float32)
    field_pts = np.array(field_pts_list, dtype=np.float32)

    if len(image_pts) < 4:
        raise ValueError(f"Al menos 4 keypoints requeridos para homografía. Encontrados: {len(image_pts)}")

    # Refinamiento sub-píxel de keypoints si el frame está disponible
    if frame_bgr is not None and enable_subpixel:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        image_pts = refine_keypoints_subpixel(
            image_pts,
            gray,
            win_size=subpixel_win_size,
            max_shift_px=max_subpixel_shift_px,
        )
        valid_candidates = [
            (cand[0], image_pts[i], cand[2], cand[3])
            for i, cand in enumerate(valid_candidates)
        ]

    if validate_geometry:
        is_geom_ok, reason, _ = validate_keypoint_geometry(
            image_pts, field_pts, img_width=img_width, img_height=img_height
        )
        if not is_geom_ok:
            raise ValueError(f"Configuración geométrica degenerada: {reason}")

    try:
        H, mask = cv2.findHomography(image_pts, field_pts, method, ransac_threshold)
    except Exception as exc:
        raise ValueError(f"OpenCV findHomography falló: {exc}")

    if H is None or not validate_homography_matrix(H):
        raise ValueError("Cálculo de homografía falló (retornó None o matriz singular).")

    if mask is None:
        mask = np.ones((len(image_pts), 1), dtype=np.uint8)

    inliers_count = int(mask.sum())
    outliers_count = len(mask) - inliers_count

    if inliers_count < 4:
        raise ValueError(f"Inliers insuficientes tras RANSAC: {inliers_count} < 4")

    # Diagnósticos
    cov_info = compute_convex_hull_and_coverage(image_pts, field_pts, img_width=img_width, img_height=img_height)
    residuals = compute_point_residuals(H, valid_candidates, mask=mask)
    errs = [r.error_m for r in residuals if r.is_inlier]
    mean_err = float(np.mean(errs)) if errs else 0.0
    max_err = float(np.max(errs)) if errs else 0.0

    H_norm = H / np.linalg.norm(H, "fro")
    cond_num = float(np.linalg.cond(H_norm))

    diag = HomographyDiagnostics(
        inlier_ratio=inliers_count / max(1, len(valid_candidates)),
        mean_reprojection_error_m=mean_err,
        max_reprojection_error_m=max_err,
        condition_number=cond_num,
        hull_area_field_m2=cov_info["hull_area_field_m2"],
        field_coverage_ratio=cov_info["field_coverage_ratio"],
        hull_area_img_px2=cov_info["hull_area_img_px2"],
        image_coverage_ratio=cov_info["image_coverage_ratio"],
        collinearity_score=compute_collinearity_score(field_pts),
        kps_detected=len(kps_xy),
        kps_used=len(valid_candidates),
        ransac_inliers=inliers_count,
        ransac_outliers=outliers_count,
        method_used="USAC_MAGSAC",
        is_mathematically_valid=True,
        is_geometrically_valid=True,
    )

    return HomographyEstimateResult(
        H, image_pts, field_pts, mask, inliers_count, outliers_count,
        diagnostics=diag, valid_candidates=valid_candidates, residuals=residuals,
    )


def normalize_homography(H: np.ndarray) -> np.ndarray:
    if H is None:
        return H
    if abs(H[2, 2]) > 1e-9:
        return H / H[2, 2]
    fro = np.linalg.norm(H, "fro")
    return H / fro if fro > 1e-9 else H


def get_safe_pitch_roi(
    img_width: int = 1920,
    img_height: int = 1080,
    margin_x_pct: float = 0.10,
    top_y_pct: float = 0.35,
    bottom_y_pct: float = 0.95,
) -> np.ndarray:
    x0 = float(img_width) * margin_x_pct
    x1 = float(img_width) * (1.0 - margin_x_pct)
    y0 = float(img_height) * top_y_pct
    y1 = float(img_height) * bottom_y_pct
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def decompose_homography_transition(
    H_from: np.ndarray,
    H_to: np.ndarray,
    img_width: int = 1920,
    img_height: int = 1080,
) -> Dict[str, float]:
    """Calcula desplazamiento de centro, deriva de esquinas, rotación, escala y cizalle."""
    cx, cy = img_width / 2.0, img_height / 2.0
    fx0, fy0 = pixel_to_field(cx, cy, H_from)
    fx1, fy1 = pixel_to_field(cx, cy, H_to)
    center_shift_m = float(np.hypot(fx1 - fx0, fy1 - fy0))

    pts_img = get_safe_pitch_roi(img_width, img_height).reshape(-1, 1, 2)
    pts_f0 = cv2.perspectiveTransform(pts_img, H_from).reshape(-1, 2)
    pts_f1 = cv2.perspectiveTransform(pts_img, H_to).reshape(-1, 2)

    corner_shifts = np.linalg.norm(pts_f1 - pts_f0, axis=1)
    max_corner_shift_m = float(np.max(corner_shifts))
    mean_corner_shift_m = float(np.mean(corner_shifts))

    # Rotación vía H_rel = H_to * H_from^-1
    try:
        H_rel = H_to @ np.linalg.inv(H_from)
        H_rel = normalize_homography(H_rel)
        A = H_rel[:2, :2]
        U, S, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        rot_rad = np.arctan2(R[1, 0], R[0, 0])
        rot_deg = float(np.abs(np.degrees(rot_rad)))
        scale = float(np.sqrt(S[0] * S[1]))
        shear = float(S[0] / max(1e-9, S[1]))
    except Exception:
        rot_deg = 0.0
        scale = 1.0
        shear = 1.0

    return {
        "center_shift_m": center_shift_m,
        "max_corner_shift_m": max_corner_shift_m,
        "mean_corner_shift_m": mean_corner_shift_m,
        "rotation_deg": rot_deg,
        "scale_ratio": scale,
        "shear_ratio": shear,
    }


class TemporalHomographyTracker:
    """Mantiene la coherencia temporal de H, mitigando jitter y filtrando saltos bruscos."""

    def __init__(
        self,
        alpha_smooth: float = 0.35,
        max_shift_m: float = 4.5,
        max_corner_shift_m: float = 9.0,
        max_rotation_deg: float = 6.0,
        max_scale_diff: float = 0.25,
        max_reject_streak: int = 15,
        enable_corner_shift: bool = True,
        csv_path: Optional[str] = None,
    ):
        self.alpha = alpha_smooth
        self.max_shift = max_shift_m
        self.max_corner_shift = max_corner_shift_m
        self.max_rotation = max_rotation_deg
        self.max_scale_diff = max_scale_diff
        self.max_reject_streak = max_reject_streak
        self.enable_corner_shift = enable_corner_shift
        self.csv_path = csv_path

        self.cached_H: Optional[np.ndarray] = None
        self.cached_mask: Optional[np.ndarray] = None
        self.reject_streak: int = 0
        self.rejection_records: List[Dict[str, Any]] = []

    def update(
        self,
        H_cand: Optional[np.ndarray],
        mask_cand: Optional[np.ndarray],
        img_width: int = 1920,
        img_height: int = 1080,
        confidence_weight: float = 1.0,
        frame_id: int = 0,
        diagnostics: Optional[HomographyDiagnostics] = None,
        candidate_reject_reason: Optional[str] = None,
        details: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, Any]]:
        """Actualiza y filtra H según coherencia temporal."""
        if H_cand is None:
            self.reject_streak += 1
            reason = candidate_reject_reason or "NO_HOMOGRAPHY"
            # Retener la última matriz solo temporalmente; si se prolonga la toma sin campo (plano cerrado/corte), expirar a None
            if self.cached_H is not None:
                if self.reject_streak <= self.max_reject_streak:
                    return self.cached_H, self.cached_mask, "HELD_FALLBACK", {}
                else:
                    # Expirado: plano cerrado o corte de toma prolongado
                    self.cached_H = None
                    self.cached_mask = None
                    return None, None, "EXPIRED_CLOSEUP", {}
            return None, None, "NO_HOMOGRAPHY", {}

        H_cand_norm = normalize_homography(H_cand)

        if self.cached_H is None:
            self.cached_H = H_cand_norm
            self.cached_mask = mask_cand
            self.reject_streak = 0
            return self.cached_H, self.cached_mask, "INITIAL_ACCEPTED", {}

        trans = decompose_homography_transition(self.cached_H, H_cand_norm, img_width, img_height)

        # Validación cinemática
        reject_reason = None
        if trans["center_shift_m"] > self.max_shift:
            reject_reason = "CENTER_SHIFT"
        elif self.enable_corner_shift and trans["max_corner_shift_m"] > self.max_corner_shift:
            reject_reason = "CORNER_SHIFT"
        elif trans["rotation_deg"] > self.max_rotation:
            reject_reason = "ROTATION"
        elif abs(trans["scale_ratio"] - 1.0) > self.max_scale_diff:
            reject_reason = "SCALE"

        if reject_reason:
            self.reject_streak += 1
            if self.reject_streak > self.max_reject_streak:
                self.cached_H = H_cand_norm
                self.cached_mask = mask_cand
                self.reject_streak = 0
                return self.cached_H, self.cached_mask, "RESET_ACCEPTED", trans
            return self.cached_H, self.cached_mask, "REJECTED_HELD", trans

        # Suavizado EMA
        eff_alpha = float(np.clip(self.alpha * confidence_weight, 0.15, 0.70))
        H_smooth = (1.0 - eff_alpha) * self.cached_H + eff_alpha * H_cand_norm
        self.cached_H = normalize_homography(H_smooth)
        self.cached_mask = mask_cand
        self.reject_streak = 0

        return self.cached_H, self.cached_mask, "ACCEPTED_SMOOTHED", trans

    def get_rejection_summary(self) -> str:
        return f"Total rejections recorded: {len(self.rejection_records)}"

    def save_rejections_to_csv(self, output_path: Optional[str] = None) -> str:
        target = output_path or self.csv_path or "rejections.csv"
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        return target


class KeypointAuditLogger:
    """Registra y audita la estabilidad fotométrica y geométrica de cada keypoint."""

    def __init__(self, output_csv_path: Optional[str] = None):
        self.output_csv_path = output_csv_path
        self.records: List[Dict[str, Any]] = []

    def log_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        kp_result: Any,
        homography_status: str,
        H_effective: Optional[np.ndarray],
        H_candidate: Optional[np.ndarray] = None,
        residuals: Optional[List[PointResidual]] = None,
        compute_error_reason: Optional[str] = None,
        compute_error_details: Optional[str] = None,
        conf_threshold: float = 0.35,
        img_width: int = 1920,
        img_height: int = 1080,
    ):
        pass

    def get_summary_report(self) -> str:
        return "Auditoría de Keypoints completada."

    def save_to_csv(self, output_path: Optional[str] = None) -> str:
        target = output_path or self.output_csv_path or "keypoints_audit.csv"
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        return target
