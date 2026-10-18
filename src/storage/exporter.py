"""
Módulo de Persistencia y Exportación de Datos (JSON, CSV, SQLite)
Serializa y exporta tracking, eventos tácticos, métricas y metadatos del partido.
"""

import json
import csv
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger("src.storage.exporter")


class DataExporter:
    """
    Gestiona la persistencia de resultados del análisis:
    - JSON Contract oficial (tracking espacial frame a frame, metadatos, equipos).
    - CSV de eventos de juego (pases, recuperaciones, remates).
    - SQLite persistente para consultas analíticas relacionales y visualización en dashboards.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        dataset: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> str:
        """Exporta el dataset completo en formato JSON contract."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"[DataExporter] 💾 Tracking JSON exportado: {p}")
        return str(p)

    def export_metadata_json(
        self,
        metadata: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> str:
        """Exporta el archivo independiente de metadatos del partido (equipos, colores, video)."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[DataExporter] 📋 Match Metadata JSON exportado: {p}")
        return str(p)

    def export_events_csv(
        self,
        events: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> str:
        """Exporta la lista de eventos tácticos a CSV."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["frame_idx", "timestamp_sec", "timestamp_fmt", "event_type", "team_id", "confidence", "details"]

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for evt in events:
                ts = evt.get("timestamp_sec", 0.0)
                mins = int(ts // 60)
                secs = int(ts % 60)
                ts_fmt = f"{mins:02d}:{secs:02d}"

                details = []
                for k, v in evt.items():
                    if k not in ("frame_idx", "timestamp_sec", "event_type", "team_id", "confidence"):
                        details.append(f"{k}={v}")

                writer.writerow([
                    evt.get("frame_idx"),
                    ts,
                    ts_fmt,
                    evt.get("event_type"),
                    evt.get("team_id"),
                    evt.get("confidence", 1.0),
                    "; ".join(details),
                ])

        logger.info(f"[DataExporter] 📄 CSV de eventos exportado: {p}")
        return str(p)

    def export_sqlite(
        self,
        dataset: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> str:
        """Exporta los datos a una base de datos relacional SQLite."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(p))
        cursor = conn.cursor()

        # Tabla de Metadatos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        meta = dataset.get("metadata", {})
        for k, v in meta.items():
            val_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (k, val_str))

        # Tabla de Eventos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_idx INTEGER,
                timestamp_sec REAL,
                event_type TEXT,
                team_id INTEGER,
                confidence REAL,
                details TEXT
            )
        """)
        for evt in dataset.get("events", []):
            cursor.execute("""
                INSERT INTO events (frame_idx, timestamp_sec, event_type, team_id, confidence, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                evt.get("frame_idx"),
                evt.get("timestamp_sec"),
                evt.get("event_type"),
                evt.get("team_id"),
                evt.get("confidence", 1.0),
                json.dumps(evt),
            ))

        # Tabla de Posición de Jugadores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_tracking (
                frame_idx INTEGER,
                timestamp_sec REAL,
                track_id INTEGER,
                team_id INTEGER,
                role TEXT,
                pixel_x REAL,
                pixel_y REAL,
                pitch_x REAL,
                pitch_y REAL,
                is_interpolated INTEGER
            )
        """)
        for fr in dataset.get("frames", []):
            f_idx = fr.get("frame")
            ts = fr.get("timestamp", 0.0)
            for pl in fr.get("players", []):
                cursor.execute("""
                    INSERT INTO player_tracking (frame_idx, timestamp_sec, track_id, team_id, role, pixel_x, pixel_y, pitch_x, pitch_y, is_interpolated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f_idx,
                    ts,
                    pl.get("track_id"),
                    pl.get("team_id"),
                    pl.get("role"),
                    pl.get("pixel_x"),
                    pl.get("pixel_y"),
                    pl.get("pitch_x"),
                    pl.get("pitch_y"),
                    1 if pl.get("is_interpolated") else 0,
                ))

        conn.commit()
        conn.close()

        logger.info(f"[DataExporter] 🗄️ SQLite Database exportada: {p}")
        return str(p)
