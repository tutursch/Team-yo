import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from local_occupancy import sensor_measurements, sensor_distances

%matplotlib inline

def sensor_val_to_cm_dist(val):
    """
    Returns the distance corresponding to the sensor value based 
    on the sensor characteristics
    :param val: the sensor value that you want to convert to a distance
    :return: corresponding distance in cm
    """
    if val == 0:
        return np.inf
    
    f = interp1d(sensor_measurements, sensor_distances)
    return np.asscalar(f(val))

def checkObstacleTentative():
	global prox_horizontal, motor_left_target, motor_right_target, state

    trigger_distance = 5;
    
    avoid = [False,False,False,False,False,False,False]
        
    if state != 0:
        
        for i in range(len(avoid)):
            if sensor_val_to_cm_dist(prox_horizontal[i]) < trigger_distance:
                avoid[i] = True

        for i in range(len(avoid)):
            if avoid[i]:
                setMotors(i)

def setMotors(case):
    if case == 0:
        #turn right
    elif case == 1:
        #turn right more
    elif case == 2:
        #l'obstacle est en face, peut-être comparer les valeurs des sensors 2 et 4
    elif case == 3:
        #turn left more
    elif case == 4:
        #turn left


def checkObstacle(sensor):
    #global prox_horizontal, motor_left_target, motor_right_target, button_center, state

    basic_speed = 50;
    w_l = [40,  20, -20, -20, -40,  30, -10]
    w_r = [-40, -20, -20,  20,  40, -10,  30]

    # Scale factors for sensors and constant factor
    sensor_scale = 200
    constant_scale = 20
    
    motor = [0,0]
         
    for i in range(len(sensor)):
        # Get and scale inputs
        sensor[i] = prox_horizontal[i] // sensor_scale
            
        # Compute outputs of neurons and set motor powers
        motor[0] = basic_speed + sensor[i] * w_l[i]
        motor[1] = basic_speed + sensor[i] * w_r[i]
    
    # Set motor powers
    #motor_left_target = motor[0]
    #motor_right_target = motor[1]
    
    return motor
