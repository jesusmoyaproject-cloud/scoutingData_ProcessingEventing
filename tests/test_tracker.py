"""
Pruebas Unitarias para el Módulo de Tracking (BallTracker y PlayerKinematicTracker)
"""

import unittest
import numpy as np
from src.tracking.botsort import BallTracker, PlayerKinematicTracker


class TestTracker(unittest.TestCase):

    def test_ball_tracker_rejection_and_interpolation(self):
        tracker = BallTracker(max_history=5, max_missing_frames=3, max_distance_jump=100.0)

        # 1. Inserción normal de dos detecciones
        det1 = {"pixel_x": 100.0, "pixel_y": 100.0, "confidence": 0.9}
        res1, hist1 = tracker.update(det1)
        self.assertIsNotNone(res1)
        self.assertEqual(len(hist1), 1)

        det2 = {"pixel_x": 110.0, "pixel_y": 110.0, "confidence": 0.85}
        res2, hist2 = tracker.update(det2)
        self.assertIsNotNone(res2)
        self.assertEqual(len(hist2), 2)

        # 2. Detección espuria con salto anormal (> 100 px) -> Debe ser rechazada
        det_jump = {"pixel_x": 500.0, "pixel_y": 500.0, "confidence": 0.9}
        res_jump, _ = tracker.update(det_jump)
        # La detección espuria se descarta y se procede a interpolar o None
        if res_jump is not None:
            self.assertTrue(res_jump.get("interpolated", False))

        # 3. Frame sin detección -> Interpolación cinemática
        res_missing, _ = tracker.update(None)
        self.assertIsNotNone(res_missing)
        self.assertTrue(res_missing.get("interpolated", False))

    def test_player_kinematic_extrapolation(self):
        kin_tracker = PlayerKinematicTracker()

        # Frame 0: Jugador en (200, 300)
        dets_f0 = [
            {"player_id": 10, "pixel_x": 200.0, "pixel_y": 300.0, "bbox": [180.0, 250.0, 220.0, 300.0]}
        ]
        kin_tracker.update_from_detections(dets_f0)

        # Frame 1: Jugador avanza a (205, 302) -> velocidad (+5, +2)
        dets_f1 = [
            {"player_id": 10, "pixel_x": 205.0, "pixel_y": 302.0, "bbox": [185.0, 252.0, 225.0, 302.0]}
        ]
        kin_tracker.update_from_detections(dets_f1)

        # Frame 2 (omitido por stride): Extrapolación debe dar (210, 304)
        extrapolated = kin_tracker.extrapolate_next_frame(H=None)
        self.assertEqual(len(extrapolated), 1)
        p = extrapolated[0]
        self.assertEqual(p["player_id"], 10)
        self.assertEqual(p["pixel_x"], 210.0)
        self.assertEqual(p["pixel_y"], 304.0)
        self.assertTrue(p["is_interpolated"])


if __name__ == "__main__":
    unittest.main()
