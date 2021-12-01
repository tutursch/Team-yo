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
    print("checkObstacle")    

    obstacle_trigger = 10
    obstacle = False

    for i in range(len(sensor)-2) :
        if sensor[i] > obstacle_trigger :
            obstacle = True

    print("obstacle = ")
    print(obstacle)
    return obstacle

def avoidObstacle(sensor):
    print("avoidObstacle")
    w = [40,  20, -20, -20, -40]

    # Scale factors for sensors and constant factor
    sensor_scale = 800
    avoidance_speed = 0

    for i in range(len(sensor)-2) :
        # Get and scale inputs
        sensor[i] = sensor[i] // sensor_scale
            
        # Compute outputs of neurons and set motor powers
        avoidance_speed += sensor[i] * w[i]

    print("avoidance_speed = ")
    print(avoidance_speed)
    return avoidance_speed

def pdController(orientation, old_orientation, goal_orientation) :
    print("pdController")
    kp = 10
    kd = 1

    error = goal_orientation-orientation
    old_error = goal_orientation-old_orientation

    control_speed = kp * error + kd * (error-old_error)

    print("control_speed = ")
    print(control_speed)
    return control_speed


def localNavigation(theta, old_theta, goal_theta, sensors) :
    print("localNavigation")
    basic_speed = 100

    control_speed = 0
    control_speed = pdController(theta, old_theta, goal_theta)

    avoidance_speed = 0
    obstacle = False
    obstacle = checkObstacle(sensors)
    if obstacle : 
        avoidance_speed = avoidObstacle(sensors)

    left_speed = basic_speed - control_speed + avoidance_speed
    right_speed = basic_speed + control_speed - avoidance_speed

    print("left_speed = ")
    print(left_speed)
    print("right_speed = ")
    print(right_speed)
    return left_speed, right_speed
