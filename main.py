import numpy as np


class Model:
    def __init__(self, Δt):


        self.Δt = Δt
        # State
        self.x = np.array([20.0, 0.0])

        #
        self.H = np.array([1.0, 0.0]).reshape(1,2)
        self.R = 0.25

        self.A = np.array([[1.0, Δt],[0.0, 1.0]])

        # Covariance
        self.Q = np.array([[0.01, 0.0],[0.0, 0.033]])

        self.P_0 = np.array([[10, 0.0], [0.0, 0.033]])
        self.P = self.P_0

        self.m = self.x
        self.p = self.P_0



    def predict(self):
        self.m_ = self.A @ self.m

        self.p_ = self.A @ self.p @ self.A.T + self.Q

        return self.m_, self.p_

    def update(self, y_k):
        S_k = self.H @ self.p_ @ self.H.T + self.R
        S_k_debug = self.H @ self.p @ self.H.T
        K_k = (self.p_ @ self.H.T) * (1.0/S_k)
        self.m = self.m_ + K_k * (y_k - (self.H @ self.m_))
        self.p = self.p_ - S_k*(K_k @ K_k.T)
        return self.m, self.p


m = Model(12.0)

for i in range(1000):
    m.predict()
    print(m.update(20.0+0.1*i))
