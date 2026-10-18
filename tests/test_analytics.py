"""
Pruebas Unitarias para el Módulo de Analítica Táctica (Posesión y Eventos)
"""

import unittest
from src.analytics.ball_possession import BallPossessionEngine
from src.analytics.events import MatchEventDetector


class TestAnalytics(unittest.TestCase):

    def test_ball_possession_and_stats(self):
        engine = BallPossessionEngine(possession_radius_m=2.0, min_possession_frames=2)

        # Balón en (50.0, 34.0)
        ball = {"pitch_x_m": 50.0, "pitch_y_m": 34.0}

        # Jugador 10 (Equipo 0) en (50.5, 34.2) -> Distancia ~ 0.54m (dentro de 2.0m)
        players = [
            {"player_id": 10, "team_id": 0, "pitch_x_m": 50.5, "pitch_y_m": 34.2, "role": "player"},
            {"player_id": 20, "team_id": 1, "pitch_x_m": 60.0, "pitch_y_m": 40.0, "role": "player"},
        ]

        # Frame 1: Primer contacto (candidato detectado, aún no confirmado)
        res1 = engine.update(ball, players)
        self.assertEqual(res1["candidate_id"], 10)
        self.assertIsNone(res1["possessor_id"])
        self.assertFalse(res1["is_confirmed"])

        # Frame 2: Confirmación tras 2 frames consecutivos
        res2 = engine.update(ball, players)
        self.assertEqual(res2["possessor_id"], 10)
        self.assertEqual(res2["team_id"], 0)
        self.assertTrue(res2["is_confirmed"])

        stats = engine.get_possession_stats()
        self.assertEqual(stats["team_0_pct"], 100.0)
        self.assertEqual(stats["team_1_pct"], 0.0)

    def test_pass_and_turnover_events(self):
        detector = MatchEventDetector()

        # 1. Posesión inicial: Jugador 10 (Equipo 0)
        detector.update(
            frame_idx=10,
            timestamp_sec=0.33,
            possession_info={"possessor_id": 10, "team_id": 0, "is_confirmed": True},
            ball_det=None,
        )

        # 2. Pase completado a Jugador 11 (Mismo Equipo 0) en Frame 15
        events_pass = detector.update(
            frame_idx=15,
            timestamp_sec=0.50,
            possession_info={"possessor_id": 11, "team_id": 0, "is_confirmed": True},
            ball_det=None,
        )
        self.assertEqual(len(events_pass), 1)
        self.assertEqual(events_pass[0]["event_type"], "PASS")
        self.assertEqual(events_pass[0]["from_player_id"], 10)
        self.assertEqual(events_pass[0]["to_player_id"], 11)

        # 3. Intercepción / Pérdida: Jugador 25 (Equipo 1) recupera el balón en Frame 20
        events_turnover = detector.update(
            frame_idx=20,
            timestamp_sec=0.66,
            possession_info={"possessor_id": 25, "team_id": 1, "is_confirmed": True},
            ball_det=None,
        )
        self.assertEqual(len(events_turnover), 1)
        self.assertEqual(events_turnover[0]["event_type"], "TURNOVER")
        self.assertEqual(events_turnover[0]["lost_by_team"], 0)
        self.assertEqual(events_turnover[0]["won_by_team"], 1)

    def test_ball_possession_rejects_fast_moving_ball(self):
        """Verifica que un balón disparado o pasado a alta velocidad cerca de un jugador no se asigne erróneamente."""
        engine = BallPossessionEngine(
            possession_radius_m=1.8,
            min_possession_frames=2,
            max_relative_speed_mps=6.0,
        )

        players = [
            {"player_id": 15, "team_id": 0, "pitch_x_m": 50.0, "pitch_y_m": 30.0, "role": "player"}
        ]

        # Frame 1 (t=0.0): Balón en (48.0, 30.0) -> a 2m de distancia (fuera de radio)
        ball_f1 = {"pitch_x_m": 48.0, "pitch_y_m": 30.0}
        res1 = engine.update(ball_f1, players, timestamp_sec=0.0)
        self.assertIsNone(res1["candidate_id"])

        # Frame 2 (t=0.04): Balón cruza a (51.0, 30.0) -> a 1.0m del jugador (dentro de 1.8m)
        # Velocidad del balón: 3m en 0.04s = 75 m/s (> 6.0 m/s)
        ball_f2 = {"pitch_x_m": 51.0, "pitch_y_m": 30.0}
        res2 = engine.update(ball_f2, players, timestamp_sec=0.04)

        # Debe descartar la posesión porque el balón pasó volando
        self.assertIsNone(res2["candidate_id"])
        self.assertIsNone(res2["possessor_id"])

    def test_ball_possession_directional_cone(self):
        """Verifica que el cono anatómico reduzca el radio efectivo a la espalda de un jugador en carrera."""
        engine = BallPossessionEngine(
            possession_radius_m=1.8,
            min_possession_frames=1,
            enable_directional_cone=True,
        )

        # Jugador 10 corriendo hacia la derecha (+X) a 5 m/s
        # Frame 1: Jugador en (50.0, 30.0), Balón adelante en (51.5, 30.0)
        p_f1 = [{"player_id": 10, "team_id": 0, "pitch_x_m": 50.0, "pitch_y_m": 30.0, "role": "player"}]
        ball_f1 = {"pitch_x_m": 51.0, "pitch_y_m": 30.0}
        engine.update(ball_f1, p_f1, timestamp_sec=0.0)

        # Frame 2: Jugador avanzó a (50.2, 30.0) (velocidad ~5 m/s hacia +X)
        # Balón está a su espalda a 1.4m (en 48.8, 30.0). Dentro de 1.8m circular pero fuera del cono posterior (1.8 * 0.55 = ~0.99m)
        p_f2 = [{"player_id": 10, "team_id": 0, "pitch_x_m": 50.2, "pitch_y_m": 30.0, "role": "player"}]
        ball_f2 = {"pitch_x_m": 48.8, "pitch_y_m": 30.0}
        res2 = engine.update(ball_f2, p_f2, timestamp_sec=0.04)

        # Debe ser rechazado por estar a la espalda mientras corre hacia adelante
        self.assertIsNone(res2["candidate_id"])


if __name__ == "__main__":
    unittest.main()
