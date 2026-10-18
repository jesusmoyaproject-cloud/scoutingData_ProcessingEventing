"""
Pruebas Unitarias para el Módulo de Apariencia (Extractor de Color y Clasificador de Equipos)
"""

import unittest
import numpy as np
from src.appearance.color_extractor import JerseyColorExtractor
from src.appearance.clustering import TeamClassifier


class TestColorClustering(unittest.TestCase):

    def setUp(self):
        self.extractor = JerseyColorExtractor()
        self.classifier = TeamClassifier(min_samples_to_cluster=4, history_window=10, lock_threshold=5)

    def test_jersey_color_extraction_synthetic(self):
        # Frame sintético con fondo de césped verde (BGR: [34, 139, 34])
        h, w = 720, 1280
        frame = np.full((h, w, 3), [34, 139, 34], dtype=np.uint8)

        # Torso rojo (BGR: [20, 20, 220])
        bbox_red = [200.0, 200.0, 260.0, 360.0]
        ry1, ry2 = int(200 + 0.15 * 160), int(200 + 0.50 * 160)
        rx1, rx2 = int(200 + 0.15 * 60), int(260 - 0.15 * 60)
        frame[ry1:ry2, rx1:rx2] = [20, 20, 220]

        extracted = self.extractor.extract_jersey_color(frame, bbox_red)
        self.assertIsNotNone(extracted)
        lab, bgr = extracted
        self.assertEqual(len(lab), 3)
        self.assertEqual(len(bgr), 3)

    def test_team_classifier_calibration_and_roles(self):
        # Muestras sintéticas LAB: Equipo 1 (Rojo) vs Equipo 2 (Blanco)
        lab_red = np.array([120.0, 50.0, 40.0], dtype=np.float32)
        bgr_red = (20, 20, 220)

        lab_white = np.array([230.0, 0.0, 0.0], dtype=np.float32)
        bgr_white = (240, 240, 240)

        # Entrenar calibración K-Means
        for _ in range(5):
            self.classifier.classify_player(lab_red, bgr_red, track_id=1, pitch_x_m=50.0, pitch_y_m=30.0)
            self.classifier.classify_player(lab_white, bgr_white, track_id=2, pitch_x_m=55.0, pitch_y_m=35.0)

        self.assertTrue(self.classifier._is_calibrated)
        self.assertIsNotNone(self.classifier.centroids_lab)
        self.assertEqual(len(self.classifier.centroids_lab), 2)

        # Locking temporal tras observaciones consistentes
        for _ in range(8):
            res_locked = self.classifier.classify_player(lab_red, bgr_red, track_id=1, pitch_x_m=50.0, pitch_y_m=30.0)
        self.assertTrue(res_locked["is_locked"])

        # Rol contextual: Portero (Color outlier cerca del arco en x=5m)
        lab_gk = np.array([200.0, 20.0, 180.0], dtype=np.float32)
        bgr_gk = (0, 255, 255)
        gk_res = self.classifier.classify_player(lab_gk, bgr_gk, track_id=99, pitch_x_m=5.0, pitch_y_m=34.0)
        self.assertEqual(gk_res["role"], "goalkeeper")

        # Rol contextual: Árbitro (Color outlier en el medio campo x=52m cuando initial_role no está definido)
        ref_res = self.classifier.classify_player(lab_gk, bgr_gk, track_id=100, pitch_x_m=52.0, pitch_y_m=34.0)
        self.assertEqual(ref_res["role"], "referee")
        self.assertEqual(ref_res["team_id"], -1)

    def test_player_role_preservation_and_no_false_referee(self):
        # Cuando initial_role es 'player' (detectado por YOLO), nunca debe convertirse en 'referee'
        # incluso si su color difiere en el medio campo
        lab_red = np.array([120.0, 50.0, 40.0], dtype=np.float32)
        bgr_red = (20, 20, 220)
        lab_blue = np.array([150.0, -20.0, -30.0], dtype=np.float32)
        bgr_blue = (220, 100, 20)

        for _ in range(5):
            self.classifier.classify_player(lab_red, bgr_red, track_id=10, initial_role="player")
            self.classifier.classify_player(lab_blue, bgr_blue, track_id=20, initial_role="player")

        # Jugador rojo en el medio campo (x=50m) con initial_role="player"
        p_res = self.classifier.classify_player(lab_red, bgr_red, track_id=11, pitch_x_m=50.0, initial_role="player")
        self.assertEqual(p_res["role"], "player")
        self.assertNotEqual(p_res["team_id"], -1)
        self.assertEqual(p_res["team_id"], 0)


if __name__ == "__main__":
    unittest.main()
