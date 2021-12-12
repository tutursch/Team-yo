import numpy as np
from numpy.lib.function_base import select


# Initialization of all the matrixes and vectors for the kalman filter 
class KalmanFilter(object):
    def __init__(self, dt):

        self.dt = dt 

        # Matrix that links the previous state to the current one 
        self.A = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

        # State model noise covariance matrix
        self.Q = np.array([[10,  0,   0],
                           [  0, 10,   0],
                           [  0,   0, 10]])

        # Covariance matrix 
        self.R = np.array([[1,   0,    0],
                            [  0, 1,    0],
                            [  0,    0, 1]])

        # Measurement matrix
        self.H = np.array([[1.0,  0,   0],
                            [  0,1.0,   0],
                            [  0,  0, 1.0]])

        # Predicted covariance matrix of the state estimate
        self.P = np.array([[1.0,   0,   0],
                            [  0, 1.0,   0],
                            [  0,   0, 1.0]])

        # Observation vector
        self.E = np.array([[0], [0], [0]])

    def predict(self, pos_1, angle_1, motor_left, motor_right):

        # Matrix that links the robots input to its state
        self.B = np.array([[np.cos(angle_1)*self.dt, 0],
                           [np.sin(angle_1)*self.dt, 0],
                           [0, self.dt]])

        self.E = np.array([[pos_1[0]],
                            [pos_1[1]],
                            [angle_1]])

        # Radial speed
        self.omega = ((motor_left - motor_right)*np.pi/180)*0.38
        # Linear speed
        self.v = ((motor_right+motor_left)/2)/2.46
        # speed vector
        self.V = np.array([[self.v],
                           [self.omega]])

        self.E = np.dot(self.A, self.E) + np.dot(self.B, self.V)


        # Computation of the covariance error
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + (self.Q)

        state = [self.E[0][0], self.E[1][0], self.E[2][0]]

        return state

    def update(self, z):

        # Calculation of the measurement residual covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculatation of the near-optimal Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Computation of an updated state estimate for time k
        self.E = self.E+np.dot(K, (z-np.dot(self.H, self.E)))
        # Update of the state covariance estimate for time k
        I=np.eye(self.H.shape[1])
        self.P = (I-(K*self.H))*self.P

        state = [self.E[0][0], self.E[1][0], self.E[2][0]]
        return state