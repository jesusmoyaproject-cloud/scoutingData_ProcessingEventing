"""
Pruebas unitarias para la exportación de metadatos del partido y visualización de IDs de jugadores.
"""

import json
import unittest
import numpy as np
from pathlib import Path
import tempfile
import cv2

from src.appearance.clustering import TeamClassifier
from src.storage.exporter import DataExporter
from src.visualization.renderer import MatchRenderer


class TestMetadataAndVisualization(unittest.TestCase):

    def test_team_keys_and_metadata_structure(self):
        classifier = TeamClassifier(min_samples_to_cluster=2)
        
        # Test helper get_team_key
        self.assertEqual(classifier.get_team_key(0), "team_a")
        self.assertEqual(classifier.get_team_key(1), "team_b")
        self.assertEqual(classifier.get_team_key(-1), "referee")
        
        metadata = classifier.get_teams_metadata()
        
        # Validate canonical team keys exist
        self.assertIn("team_a", metadata)
        self.assertIn("team_b", metadata)
        self.assertIn("referee", metadata)
        
        # Validate color structures
        for key in ("team_a", "team_b", "referee"):
            team = metadata[key]
            self.assertIn("color_hex", team)
            self.assertTrue(team["color_hex"].startswith("#"))
            self.assertIn("color_rgb", team)
            self.assertEqual(len(team["color_rgb"]), 3)
            self.assertIn("color_bgr", team)
            self.assertEqual(len(team["color_bgr"]), 3)

    def test_export_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            exporter = DataExporter(output_dir=tmp_dir)
            target_file = Path(tmp_dir) / "match_metadata.json"
            
            sample_metadata = {
                "match_info": {
                    "video_name": "test_video.mp4",
                    "fps": 25.0,
                    "total_frames": 100,
                },
                "teams": {
                    "team_a": {"color_hex": "#FF0000", "name": "Team A"},
                    "team_b": {"color_hex": "#FFFFFF", "name": "Team B"},
                    "referee": {"color_hex": "#282828", "name": "Referee"},
                }
            }
            
            out_path = exporter.export_metadata_json(sample_metadata, target_file)
            self.assertTrue(Path(out_path).exists())
            
            with open(out_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["match_info"]["video_name"], "test_video.mp4")
            self.assertEqual(loaded["teams"]["team_a"]["color_hex"], "#FF0000")

    def test_renderer_with_player_and_minimap_ids(self):
        renderer = MatchRenderer(show_player_ids=True, show_minimap_ids=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        player_dets = [
            {
                "player_id": 7,
                "role": "player",
                "team_id": 0,
                "bbox": [100.0, 100.0, 150.0, 220.0],
                "pixel_x": 125.0,
                "pixel_y": 220.0,
                "pitch_x_m": 50.0,
                "pitch_y_m": 34.0,
                "team_color": {"bgr": [0, 0, 255], "hex": "#FF0000"},
            },
            {
                "player_id": 1,
                "role": "goalkeeper",
                "team_id": 0,
                "bbox": [50.0, 300.0, 90.0, 420.0],
                "pixel_x": 70.0,
                "pixel_y": 420.0,
                "pitch_x_m": 5.0,
                "pitch_y_m": 34.0,
                "team_color": {"bgr": [0, 255, 255], "hex": "#FFFF00"},
            },
        ]
        
        # Debe ejecutarse sin error y dibujar sobre el frame
        renderer.draw_players_on_frame(frame, player_dets)
        
        # Frame debe contener píxeles modificados
        self.assertTrue(np.any(frame > 0))
        
        # Minimapa táctico
        minimap = renderer.render_minimap(player_dets, ball_det={"pitch_x_m": 50.0, "pitch_y_m": 34.0})
        self.assertIsNotNone(minimap)
        self.assertEqual(len(minimap.shape), 3)


if __name__ == "__main__":
    unittest.main()
