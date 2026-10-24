"""
Módulo de Visualización: Renderizado Gráfico de Salidas (Camera Overlays + Minimap 2D)
Implementa dibujo in-place de alto rendimiento con:
- OPT-4: Segmentos de campo pre-calculados y proyectados con clipping.
- OPT-5: Minimapa 2D con plantilla pre-renderizada.
- OPT-6: Renderizado in-place sin copias redundantes de memoria.
"""

from typing import List, Dict, Optional, Tuple, Any
import cv2
import numpy as np

from src.utils.geometry import compute_pitch_keypoint_coordinates, field_to_pixel
from src.appearance.color_extractor import bgr_to_hex


class MatchRenderer:
    """
    Renderizador visual completo para análisis de fútbol:
    - Overlays en el plano de la cámara: proyección de líneas de cancha, badges de jugadores y balón.
    - Minimap 2D táctico (radar de campo).
    - Composición side-by-side de la salida final de video.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        minimap_scale: float = 8.0,
        minimap_padding_m: float = 5.0,
        show_frame_number: bool = True,
        show_player_ids: bool = True,
        show_minimap_ids: bool = True,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.minimap_scale = minimap_scale
        self.minimap_padding = minimap_padding_m
        self.show_frame_number = show_frame_number
        self.show_player_ids = show_player_ids
        self.show_minimap_ids = show_minimap_ids

        self._field_segments = self._generate_field_segments()
        self._minimap_template, self._pad_px = self._generate_minimap_template()

    def _generate_field_segments(self) -> List[np.ndarray]:
        """Pre-calcula los segmentos canónicos del campo (OPT-4)."""
        L, W = self.pitch_length, self.pitch_width
        PBL = 16.5
        PBW = 40.32
        GBL = 5.5
        GBW = 18.32
        PSD = 11.0
        CR = 9.15

        segments: List[np.ndarray] = []

        def seg(*pts):
            segments.append(np.array(pts, dtype=np.float32).reshape(-1, 1, 2))

        # Perímetro y medio campo
        seg((0, 0), (L, 0), (L, W), (0, W), (0, 0))
        seg((L / 2, 0), (L / 2, W))

        # Círculo central
        angles = np.linspace(0, 2 * np.pi, 73)
        circle_pts = np.column_stack([
            L / 2 + CR * np.cos(angles),
            W / 2 + CR * np.sin(angles),
        ]).astype(np.float32).reshape(-1, 1, 2)
        segments.append(circle_pts)

        # Áreas de penalti
        seg((0, W / 2 - PBW / 2), (PBL, W / 2 - PBW / 2), (PBL, W / 2 + PBW / 2), (0, W / 2 + PBW / 2))
        seg((L, W / 2 - PBW / 2), (L - PBL, W / 2 - PBW / 2), (L - PBL, W / 2 + PBW / 2), (L, W / 2 + PBW / 2))

        # Áreas chicas
        seg((0, W / 2 - GBW / 2), (GBL, W / 2 - GBW / 2), (GBL, W / 2 + GBW / 2), (0, W / 2 + GBW / 2))
        seg((L, W / 2 - GBW / 2), (L - GBL, W / 2 - GBW / 2), (L - GBL, W / 2 + GBW / 2), (L, W / 2 + GBW / 2))

        return segments

    def _generate_minimap_template(self) -> Tuple[np.ndarray, int]:
        """Pre-renderiza el canvas del minimapa táctico 2D (OPT-5)."""
        scale = self.minimap_scale
        pad_px = int(self.minimap_padding * scale)
        pw_px = int(self.pitch_length * scale)
        ph_px = int(self.pitch_width * scale)

        w = pw_px + 2 * pad_px
        h = ph_px + 2 * pad_px

        img = np.full((h, w, 3), (34, 112, 34), dtype=np.uint8)  # Césped verde BGR
        line_color = (255, 255, 255)

        def to_px(xm, ym):
            return int(xm * scale) + pad_px, int(ym * scale) + pad_px

        # Líneas de campo
        cv2.rectangle(img, to_px(0, 0), to_px(self.pitch_length, self.pitch_width), line_color, 2)
        cv2.line(img, to_px(self.pitch_length / 2, 0), to_px(self.pitch_length / 2, self.pitch_width), line_color, 2)
        cv2.circle(img, to_px(self.pitch_length / 2, self.pitch_width / 2), int(9.15 * scale), line_color, 2)
        cv2.circle(img, to_px(self.pitch_length / 2, self.pitch_width / 2), 3, line_color, -1)

        # Áreas grandes
        cv2.rectangle(img, to_px(0, self.pitch_width / 2 - 20.16), to_px(16.5, self.pitch_width / 2 + 20.16), line_color, 2)
        cv2.rectangle(img, to_px(self.pitch_length - 16.5, self.pitch_width / 2 - 20.16), to_px(self.pitch_length, self.pitch_width / 2 + 20.16), line_color, 2)

        return img, pad_px

    def draw_pitch_lines_on_frame(
        self,
        frame_bgr: np.ndarray,
        H: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ):
        """Proyecta las líneas de campo 2D sobre el frame de la cámara aplicando H_inv."""
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return

        h_img, w_img = frame_bgr.shape[:2]

        for seg in self._field_segments:
            pts = seg.reshape(-1, 2)
            n_pts = len(pts)
            if n_pts < 2:
                continue

            # Proyección perspectiva
            homo = np.column_stack([pts, np.ones(n_pts, dtype=np.float32)])
            cam = homo @ H_inv.T

            # Clipping de puntos detrás de la cámara (w > 0.01)
            w_coords = cam[:, 2]
            valid_mask = w_coords > 0.01
            if not np.any(valid_mask):
                continue

            px_x = cam[:, 0] / np.maximum(w_coords, 1e-6)
            px_y = cam[:, 1] / np.maximum(w_coords, 1e-6)

            poly_pts = []
            for i in range(n_pts):
                if valid_mask[i] and -2000 < px_x[i] < (w_img + 2000) and -2000 < px_y[i] < (h_img + 2000):
                    poly_pts.append([int(px_x[i]), int(px_y[i])])
                else:
                    if len(poly_pts) >= 2:
                        cv2.polylines(frame_bgr, [np.array(poly_pts, dtype=np.int32)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
                    poly_pts = []

            if len(poly_pts) >= 2:
                cv2.polylines(frame_bgr, [np.array(poly_pts, dtype=np.int32)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # Proyectar el punto de saque central exacto (PITCH_CENTER = 52.5, 34.0) como cruz blanca destacada
        try:
            pt_center = np.array([[[self.pitch_length / 2.0, self.pitch_width / 2.0]]], dtype=np.float32)
            cam_center = cv2.perspectiveTransform(pt_center, H_inv)[0, 0]
            cx, cy = int(round(cam_center[0])), int(round(cam_center[1]))
            if 0 <= cx < w_img and 0 <= cy < h_img:
                arm = 6
                cv2.line(frame_bgr, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.line(frame_bgr, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        except Exception:
            pass

    def draw_players_on_frame(
        self,
        frame_bgr: np.ndarray,
        player_detections: List[Dict[str, Any]],
    ):
        """Dibuja las elipses de contacto, etiquetas de rol y badges con ID de jugador."""
        for p in player_detections:
            bbox = p.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            xc = int((x1 + x2) / 2)
            yb = int(y2)

            team_color_info = p.get("team_color", {})
            bgr = tuple(team_color_info.get("bgr", [255, 0, 255]))

            # Elipse inferior en los pies
            axes = (int((x2 - x1) * 0.45), max(4, int((y2 - y1) * 0.12)))
            cv2.ellipse(frame_bgr, (xc, yb), axes, 0, 0, 360, bgr, 2, lineType=cv2.LINE_AA)

            if not self.show_player_ids:
                continue

            # Badge estilizado con ID, Rol y Nivel de Confianza (conf)
            p_id = p.get("player_id", "")
            role = p.get("role", "player")
            conf_val = p.get("confidence")
            conf_str = f" {int(round(conf_val * 100))}%" if conf_val is not None else ""

            if role == "goalkeeper":
                label = f"ID: {p_id} (GK){conf_str}"
            elif role == "referee":
                label = f"ID: {p_id} (REF){conf_str}"
            else:
                label = f"ID: {p_id}{conf_str}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            pad_x = 5
            pad_y = 3
            bw = text_w + 2 * pad_x
            bh = text_h + 2 * pad_y

            bx1 = int(xc - bw / 2)
            bx2 = bx1 + bw
            by2 = max(bh + 3, int(y1) - 4)
            by1 = by2 - bh

            # Fondo oscuro con borde del color de equipo para máximo contraste
            cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (22, 22, 22), -1)
            cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), bgr, 1, lineType=cv2.LINE_AA)

            # Texto nítido
            cv2.putText(
                frame_bgr,
                label,
                (bx1 + pad_x, by2 - pad_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

    def draw_keypoints_on_frame(
        self,
        frame_bgr: np.ndarray,
        keypoints: List[Dict[str, Any]],
    ):
        """
        Dibuja los keypoints detectados de las líneas del campo con su ID y nivel de confianza (%).
        """
        if not keypoints:
            return

        h_img, w_img = frame_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1

        for kp in keypoints:
            px = kp.get("pixel_x")
            py = kp.get("pixel_y")
            kp_id = str(kp.get("id", ""))
            conf_val = kp.get("conf")

            if px is None or py is None:
                continue

            cx, cy = int(round(px)), int(round(py))
            if cx < 0 or cx >= w_img or cy < 0 or cy >= h_img:
                continue

            # 1. Halo exterior y núcleo brillante
            cv2.circle(frame_bgr, (cx, cy), 6, (0, 200, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            # 2. Badge con ID del Keypoint y nivel de confianza detectado
            conf_text = f" {int(round(conf_val * 100))}%" if conf_val is not None else ""
            label = f"{kp_id}{conf_text}"
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            box_x1 = cx - text_w // 2 - 3
            box_y1 = cy - 10 - text_h
            box_x2 = box_x1 + text_w + 6
            box_y2 = cy - 6

            if box_y1 < 0:
                box_y1 = cy + 6
                box_y2 = cy + 10 + text_h

            cv2.rectangle(frame_bgr, (box_x1, box_y1), (box_x2, box_y2), (15, 23, 42), -1)
            cv2.rectangle(frame_bgr, (box_x1, box_y1), (box_x2, box_y2), (0, 220, 255), 1)
            cv2.putText(
                frame_bgr,
                label,
                (box_x1 + 3, box_y2 - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

    def draw_ball_on_frame(
        self,
        frame_bgr: np.ndarray,
        ball_det: Optional[Dict[str, Any]],
        ball_history: Optional[List[Tuple[float, float]]] = None,
    ):
        """Dibuja el marcador del balón y su estela de movimiento reciente."""
        if ball_history and len(ball_history) >= 2:
            pts = np.array(ball_history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame_bgr, [pts], isClosed=False, color=(0, 165, 255), thickness=2, lineType=cv2.LINE_AA)

        if ball_det is not None and "pixel_x" in ball_det and "pixel_y" in ball_det:
            bx = int(ball_det["pixel_x"])
            by = int(ball_det["pixel_y"])
            cv2.circle(frame_bgr, (bx, by), 6, (0, 140, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame_bgr, (bx, by), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    def render_minimap(
        self,
        player_detections: List[Dict[str, Any]],
        ball_det: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Renderiza la vista táctica 2D con los jugadores y el balón proyectados."""
        minimap = self._minimap_template.copy()
        scale = self.minimap_scale
        pad = self._pad_px

        # Jugadores
        for p in player_detections:
            fx = p.get("pitch_x_m")
            fy = p.get("pitch_y_m")
            if fx is None or fy is None:
                continue

            mx = int(fx * scale) + pad
            my = int(fy * scale) + pad

            if 0 <= mx < minimap.shape[1] and 0 <= my < minimap.shape[0]:
                team_color = p.get("team_color", {}).get("bgr", [255, 255, 255])
                radius = 7 if self.show_minimap_ids else 5
                cv2.circle(minimap, (mx, my), radius, tuple(team_color), -1, lineType=cv2.LINE_AA)
                cv2.circle(minimap, (mx, my), radius + 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                # Identificador de jugador en radar táctico
                if self.show_minimap_ids and p.get("player_id") is not None:
                    pid_str = str(p["player_id"])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    f_scale = 0.28 if len(pid_str) <= 2 else 0.22
                    (tw, th), _ = cv2.getTextSize(pid_str, font, f_scale, 1)
                    lum = 0.114 * team_color[0] + 0.587 * team_color[1] + 0.299 * team_color[2]
                    txt_col = (0, 0, 0) if lum > 140 else (255, 255, 255)
                    tx = mx - int(tw / 2)
                    ty = my + int(th / 2)
                    cv2.putText(minimap, pid_str, (tx, ty), font, f_scale, txt_col, 1, lineType=cv2.LINE_AA)

        # Balón
        if ball_det and ball_det.get("pitch_x_m") is not None and ball_det.get("pitch_y_m") is not None:
            bx = int(ball_det["pitch_x_m"] * scale) + pad
            by = int(ball_det["pitch_y_m"] * scale) + pad
            if 0 <= bx < minimap.shape[1] and 0 <= by < minimap.shape[0]:
                cv2.circle(minimap, (bx, by), 4, (0, 140, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(minimap, (bx, by), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        return minimap

    def compose_output_frame(
        self,
        annotated_bgr: np.ndarray,
        minimap_bgr: np.ndarray,
    ) -> np.ndarray:
        """Combina horizontalmente la toma de cámara y el radar minimap manteniendo proporciones."""
        h_f, w_f = annotated_bgr.shape[:2]
        h_m, w_m = minimap_bgr.shape[:2]

        scale_r = h_f / float(h_m)
        minimap_resized = cv2.resize(minimap_bgr, (int(w_m * scale_r), h_f))
        return np.hstack([annotated_bgr, minimap_resized])
