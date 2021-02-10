import numpy as np
class KalmanFilter:
    def __init__(self, x,std_meas):
        self.dt = 1
        self.A = np.array([[1, self.dt,0,0],
                            [0, 1,0,0],
                            [0,0,1,self.dt],
                            [0,0,0,1]])
        self.B = np.array([[(self.dt**2)/2,0,0,0],[0,self.dt,0,0],[0,0,(self.dt**2)/2,0],[0,0,0,self.dt]]) 
        self.H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.Q = np.array([[(self.dt**4)/4, (self.dt**3)/2,0,0],
                            [(self.dt**3)/2, self.dt**2,0,0],
                            [0,0,(self.dt**4)/4, (self.dt**3)/2],
                            [0,0,(self.dt**3)/2, self.dt**2]])
        self.R = [[std_meas**2,0,0,0],[0,std_meas**2,0,0],[0,0,std_meas**2,0],[0,0,0,std_meas**2]]
        self.P = np.eye(self.A.shape[1])
        self.x = x
        self.u = np.array([[0],[0],[0],[0]])
    def predict(self):
        # Ref :Eq.(9) and Eq.(10)

        # Update time state
        self.x = np.dot(self.A, self.x)
        #self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x
    def update(self, z):
            # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  # Eq.(12)

        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)
