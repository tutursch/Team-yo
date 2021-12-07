import numpy as np
from numpy.lib.function_base import select

SPEED_THYMIO_TO_CM = 0.0326 
DISTANCE_WHEEL = 9.5
CM_TO_PIXEL = 5.854

class KalmanFilter(object):
    def __init__(self, dt):

        self.dt = dt 

        self.A = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.Q = np.matrix([[1.0,   0,   0],
                           [  0, 1.0,   0],
                           [  0,   0, 1.0]])

        self.R = np.matrix([[1.0,   0,    0],
                            [  0, 1.0,    0],
                            [  0,    0, 1.0]])

        self.H = np.matrix([[1.0,  0,   0],
                           [  0,1.0,   0],
                           [  0,  0, 1.0]])

        self.ERROR = np.matrix([0.07], [0.07], [0.04])

    def predict(self, pos_1, angle_1, motor_left, motor_right, P_1):

        self.B = np.matrix([[np.cos(angle_1)*self.dt, 0],
                           [np.sin(angle_1)*self.dt, 0],
                           [0, self.dt]])

        self.E = np.matrix([[pos_1[0]],
                           [pos_1[1]],
                           [angle_1]])

        #vitesse angulaire 
        self.omega = (motor_left + motor_right)*SPEED_THYMIO_TO_CM/DISTANCE_WHEEL
        #radial speed
        self.v = (motor_right+motor_left)/2*SPEED_THYMIO_TO_CM*CM_TO_PIXEL
        #vecteur vitesse
        self.V = np.matrix([[self.v],
                           [self.omega]])
        self.E = np.dot(self.A, self.E) + np.dot(self.B, self.V) + self.ERROR

        # Calcul de la covariance de l'erreur
        self.P = np.dot(np.dot(self.A, P_1), self.A.T) + (self.Q)

        return self.E, self.P

    def update(self, z):

        # Calculate the measurement residual covariance
        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Calculate the near-optimal Kalman gain
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))

        # Calculate an updated state estimate for time k
        self.E = np.round(self.E+np.dot(self.K, (z-np.dot(self.H, self.E))))
        # Update the state covariance estimate for time k
        I=np.eye(self.H.shape[1])
        self.P = (I-(self.K*self.H))*self.P

        return self.P


