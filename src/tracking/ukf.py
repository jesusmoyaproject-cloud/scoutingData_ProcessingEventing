"""
Módulo Unscented Kalman Filter (UKF) para Tracking Deportivo.
Implementa un filtro de Kalman no lineal utilizando Puntos Sigma (Van der Merwe)
y modelo cinemático CTRV (Constant Turn Rate and Velocity) para modelar de forma
fiel aceleraciones, frenadas y giros cerrados de jugadores en el campo.
"""

from typing import Tuple, Optional
import numpy as np


class UnscentedKalmanFilter2D:
    r"""
    UKF para el seguimiento de bounding boxes y centroides de jugadores en el plano de la imagen.
    Estado \mathbf{x}: [xc, yc, s, r, vx, vy, v, yaw, yaw_rate]^T
    donde:
      - (xc, yc): centro del bounding box (pixels)
      - s: escala/área aproximada (w * h)
      - r: aspect ratio (w / h)
      - vx, vy: componentes de velocidad cartesiana
      - v: velocidad escalar
      - yaw: ángulo de dirección
      - yaw_rate: velocidad angular (giro)
    """

    def __init__(
        self,
        dim_x: int = 7,
        dim_z: int = 4,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
        dt: float = 1.0 / 30.0,
    ):
        """
        dim_x = 7: [xc, yc, s, r, vx, vy, yaw_rate]
        dim_z = 4: [xc, yc, s, r] (medición directa del bounding box)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x

        # Pesos para la media y covarianza de los puntos sigma
        self.weights_m = np.zeros(2 * self.dim_x + 1)
        self.weights_c = np.zeros(2 * self.dim_x + 1)

        self.weights_m[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.weights_c[0] = self.lambda_ / (self.dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)

        w = 1.0 / (2.0 * (self.dim_x + self.lambda_))
        self.weights_m[1:] = w
        self.weights_c[1:] = w

        # Estado x y covarianza P
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 10.0

        # Ruido de proceso Q
        self.Q = np.eye(self.dim_x)
        self.Q[0:2, 0:2] *= 1.0   # posición
        self.Q[2, 2] *= 10.0      # escala
        self.Q[3, 3] *= 0.1       # ratio
        self.Q[4:6, 4:6] *= 10.0  # velocidades
        self.Q[6, 6] *= 1.0       # giro

        # Ruido de medición R
        self.R = np.eye(self.dim_z)
        self.R[0:2, 0:2] *= 1.0
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 1.0

    def init_from_bbox(self, bbox: Tuple[float, float, float, float]):
        """Inicializa el estado x a partir de [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        w = max(1e-2, float(x2 - x1))
        h = max(1e-2, float(y2 - y1))
        xc = float(x1 + w / 2.0)
        yc = float(y1 + h / 2.0)
        s = w * h
        r = w / h

        self.x = np.zeros(self.dim_x)
        self.x[0] = xc
        self.x[1] = yc
        self.x[2] = s
        self.x[3] = r
        self.x[4] = 0.0  # vx
        self.x[5] = 0.0  # vy
        self.x[6] = 0.0  # yaw_rate

        self.P = np.eye(self.dim_x)
        self.P[0:2, 0:2] *= 10.0
        self.P[2, 2] *= 100.0
        self.P[3, 3] *= 10.0
        self.P[4:6, 4:6] *= 1000.0
        self.P[6, 6] *= 10.0

    def generate_sigma_points(self) -> np.ndarray:
        """Calcula 2n + 1 puntos sigma a partir del estado actual x y covarianza P."""
        sigmas = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigmas[0] = self.x

        # Matriz raíz cuadrada vía Cholesky con regularización para estabilidad numérica
        P_reg = (self.P + self.P.T) / 2.0
        P_reg += np.eye(self.dim_x) * 1e-6
        try:
            U = np.linalg.cholesky((self.dim_x + self.lambda_) * P_reg)
        except np.linalg.LinAlgError:
            # Fallback vía descomposición en valores singulares/eigenvalores
            eigval, eigvec = np.linalg.eigh(P_reg)
            eigval = np.maximum(eigval, 1e-6)
            U = eigvec @ np.diag(np.sqrt((self.dim_x + self.lambda_) * eigval))

        for i in range(self.dim_x):
            sigmas[i + 1] = self.x + U[:, i]
            sigmas[self.dim_x + i + 1] = self.x - U[:, i]

        return sigmas

    def state_transition(self, sigma_point: np.ndarray) -> np.ndarray:
        """
        Función no lineal de transición de estado f(x, dt):
        Incorpora velocidad angular y traslación con aceleración centrípeta.
        """
        x_next = np.copy(sigma_point)
        xc, yc, s, r, vx, vy, yaw_rate = sigma_point

        # Si hay rotación/curvatura significativa
        if abs(yaw_rate) > 1e-3:
            d_theta = yaw_rate * self.dt
            cos_t = np.cos(d_theta)
            sin_t = np.sin(d_theta)
            # Rotar vector de velocidad
            n_vx = vx * cos_t - vy * sin_t
            n_vy = vx * sin_t + vy * cos_t
            x_next[0] = xc + (vx + n_vx) * 0.5 * self.dt
            x_next[1] = yc + (vy + n_vy) * 0.5 * self.dt
            x_next[4] = n_vx
            x_next[5] = n_vy
        else:
            x_next[0] = xc + vx * self.dt
            x_next[1] = yc + vy * self.dt

        # La escala y aspect ratio se mantienen casi constantes con amortiguación
        x_next[2] = max(1.0, s)
        x_next[3] = max(0.1, min(5.0, r))
        x_next[6] = yaw_rate * 0.98  # decaimiento natural de giro

        return x_next

    def predict(self):
        """Paso de predicción temporal del UKF."""
        sigmas = self.generate_sigma_points()
        sigmas_f = np.zeros_like(sigmas)

        for i in range(len(sigmas)):
            sigmas_f[i] = self.state_transition(sigmas[i])

        # Media predicha
        self.x = np.sum(self.weights_m[:, np.newaxis] * sigmas_f, axis=0)

        # Covarianza predicha
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(len(sigmas_f)):
            y = sigmas_f[i] - self.x
            self.P += self.weights_c[i] * np.outer(y, y)
        self.P += self.Q

    def measurement_function(self, sigma_point: np.ndarray) -> np.ndarray:
        """Proyección del espacio de estados al espacio de medición z = [xc, yc, s, r]."""
        return sigma_point[0:4]

    def update(self, bbox: Tuple[float, float, float, float]):
        """Paso de corrección/actualización con nueva detección [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        w = max(1e-2, float(x2 - x1))
        h = max(1e-2, float(y2 - y1))
        z = np.array([float(x1 + w / 2.0), float(y1 + h / 2.0), w * h, w / h])

        # Generar puntos sigma con la distribución predicha
        sigmas = self.generate_sigma_points()
        sigmas_h = np.zeros((len(sigmas), self.dim_z))

        for i in range(len(sigmas)):
            sigmas_h[i] = self.measurement_function(sigmas[i])

        # Media de medición z_pred
        z_pred = np.sum(self.weights_m[:, np.newaxis] * sigmas_h, axis=0)

        # Covarianza de innovación S
        S = np.zeros((self.dim_z, self.dim_z))
        for i in range(len(sigmas_h)):
            y = sigmas_h[i] - z_pred
            S += self.weights_c[i] * np.outer(y, y)
        S += self.R

        # Covarianza cruzada T (Estado x Medición)
        T = np.zeros((self.dim_x, self.dim_z))
        for i in range(len(sigmas)):
            diff_x = sigmas[i] - self.x
            diff_z = sigmas_h[i] - z_pred
            T += self.weights_c[i] * np.outer(diff_x, diff_z)

        # Ganancia de Kalman K
        K = T @ np.linalg.pinv(S)

        # Corrección de estado y covarianza
        residual = z - z_pred
        self.x += K @ residual
        self.P -= K @ S @ K.T

        # Blindaje de estabilidad numérica contra divergencia
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=1e4, neginf=-1e4)
        self.P = np.nan_to_num(self.P, nan=1.0, posinf=1e3, neginf=0.0)
        self.P = (self.P + self.P.T) / 2.0  # Forzar simetría

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Convierte el estado actual [xc, yc, s, r] de vuelta a [x1, y1, x2, y2]."""
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=1e4, neginf=-1e4)
        xc, yc, s, r = float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])
        s = max(1.0, float(s))
        r = max(0.1, min(5.0, float(r)))

        w = float(np.sqrt(s * r))
        h = float(s / max(1e-3, w))

        x1 = float(xc - w / 2.0)
        y1 = float(yc - h / 2.0)
        x2 = float(xc + w / 2.0)
        y2 = float(yc + h / 2.0)
        return x1, y1, x2, y2
