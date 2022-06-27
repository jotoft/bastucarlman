import math

import numpy as np


class Model:
    """
    :param Δt The time step of our measurements.
    """

    def __init__(self, Δt):
        self.Δt = Δt
        # State: Temperature and dT/dt, initial guess 20 degrees, zero temp velocity.
        self.m = np.array([20.0, 0.0])

        # Sensor measurement is only affected by the temperature. Velocity has no impact on sensor.
        self.H = np.array([1.0, 0.0]).reshape(1, 2)

        # Temperature sensor standard deviation
        self.R = .25

        # Update function. Next temperature is
        self.A = np.array([[1.0, Δt], [0.0, 1.0]])

        # Covariance
        self.Q = np.array([[0.01, 0.0], [0.00, 0.00001]])

        self.P_0 = np.array([[10, 0.0], [0.00, 0.1]])
        self.P_0 *= self.P_0
        # self.P = self.P_0
        self.p = self.P_0

    def probability(self, y_k):
        distance = abs(self.m_[0] - y_k)
        deviation = math.sqrt(self.p_[0][0])
        return distance / deviation

    def predict(self, Δt):
        # Update function. Next temperature is
        self.A = np.array([[1.0, Δt], [0.0, 1.0]])
        self.m_ = self.A @ self.m
        self.p_ = (self.A @ self.p @ self.A.T) + self.Q

        return self.m_, self.p_

    def update(self, y_k):
        S_k = self.H @ self.p_ @ self.H.T + self.R
        K_k = (self.p_ @ self.H.T) * (1.0 / S_k)
        self.m = self.m_ + K_k @ (y_k - (self.H @ self.m_))
        self.p = self.p_ - S_k * (K_k @ K_k.T)
        return self.m, self.p


def test():
    model = Model(12.0)

    for i in range(10):
        m_, p_ = model.predict(12.0)
        measurement = 30 + i * 0.1

        print(f"Prediction: T {m_[0]:2} dT/12s {m_[1] * 12}")
        print(f"Measurement: {measurement}", f"Probability {model.probability(measurement)}")
        m, p = model.update(measurement)
        print(f"Update T {m[0]:2f} dT/12s {m[1] * 12}")
