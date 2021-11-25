import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

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

def checkObstacle(sensor):

    obstacleTrigger = 50
    obstacle = False

    for i in range(len(sensor)-2) :
        if sensor[i] > obstacleTrigger :
            obstacle = True

    return obstacle

def avoidObstacle(sensor, motorsInit):
    w_l = [40,  20, -20, -20, -40,  30, -10]
    w_r = [-40, -20, -20,  20,  40, -10,  30]

    # Scale factors for sensors and constant factor
    sensor_scale = 1000
    constant_scale = 20
    
    motorsOut = motorsInit

    for i in range(len(sensor)-2) :
        # Get and scale inputs
        sensor[i] = sensor[i] // sensor_scale
            
        # Compute outputs of neurons and set motor powers
        motorsOut[0] = motorsOut[0] + sensor[i] * w_r[i]
        motorsOut[1] = motorsOut[1] + sensor[i] * w_l[i]
    
    return motorsOut