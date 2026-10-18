"""
Paleta fija de colores asignados a cada grupo (Team A, Team B, Árbitro).

El color NO se deriva de la camiseta real: es solo una etiqueta visual estable que el
front puede sustituir. Lo único relevante para el análisis es la clasificación
(team_id), que se resuelve con descriptores internos en `clustering.py`.
"""

from typing import Dict, Tuple

# team_id -> color hex asignado
TEAM_PALETTE_HEX: Dict[int, str] = {
    0: "#E63946",   # Team A
    1: "#1D78C1",   # Team B
    -1: "#2B2B2B",  # Árbitro
}

UNKNOWN_COLOR_BGR: Tuple[int, int, int] = (255, 0, 255)


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


TEAM_PALETTE_BGR: Dict[int, Tuple[int, int, int]] = {
    tid: _hex_to_bgr(hx) for tid, hx in TEAM_PALETTE_HEX.items()
}


def team_color_bgr(team_id: int) -> Tuple[int, int, int]:
    """Color BGR asignado a un team_id (magenta si el id es desconocido)."""
    return TEAM_PALETTE_BGR.get(team_id, UNKNOWN_COLOR_BGR)


def team_color_hex(team_id: int) -> str:
    """Color hex asignado a un team_id."""
    bgr = team_color_bgr(team_id)
    return f"#{bgr[2]:02X}{bgr[1]:02X}{bgr[0]:02X}"
