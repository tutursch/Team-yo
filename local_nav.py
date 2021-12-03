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

def calculation_distance_and_angle(position_thymio, position_goal):
    t_x = position_thymio[0]
    t_y = position_thymio[1]
    g_x = position_goal[0]
    g_y = position_goal[1]
    distance = np.around(np.sqrt(np.power(t_x - g_x, 2) + np.power(t_y - g_y, 2)), 3)
    angle = np.around(np.arctan((position_goal[1] - position_thymio[1]) / (position_goal[0] - position_thymio[0])), 3)
    if (position_thymio[0]>position_goal[0] and position_thymio[1]<position_goal[1]):
        angle = angle + np.pi
    if (position_thymio[0]>position_goal[0] and position_thymio[1]>position_goal[1]):
        angle = angle + np.pi
    if (position_thymio[0]<position_goal[0] and position_thymio[1]>position_goal[1]):
        angle = angle + 2*np.pi
    return distance, angle

def checkObstacle(sensor):
    print("checkObstacle")    

    obstacle_trigger = 10
    obstacle = False

    for i in range(len(sensor)-2) :
        if sensor[i] > obstacle_trigger :
            obstacle = True


    return obstacle

def avoidObstacle(sensor):

    w = [40,  20, -20, -20, -40]

    # Scale factors for sensors and constant factor
    sensor_scale = 800
    avoidance_speed = 0

    for i in range(len(sensor)-2) :
        # Get and scale inputs
        sensor[i] = sensor[i] // sensor_scale
            
        # Compute outputs of neurons and set motor powers
        avoidance_speed += sensor[i] * w[i]


    return avoidance_speed

def pdController(orientation, old_orientation, goal_orientation) :

    kp = 10
    kd = 5

    error = goal_orientation-orientation
    if (error > np.pi): 
        error = error - 2*np.pi
    if (error < - np.pi):
        error = error + 2*np.pi

    old_error = goal_orientation-old_orientation
    if (old_error > np.pi): 
        old_error = old_error - 2*np.pi
    if (old_error < - np.pi):
        old_error = old_error + 2*np.pi
         
    print("error = ")
    print(error)

    control_speed = kp * error + kd * (error-old_error)


    return control_speed


def localNavigation(theta, old_theta, goal_theta, sensors) :

    basic_speed = 100

    control_speed = pdController(theta, old_theta, goal_theta)

    avoidance_speed = 0

    obstacle = checkObstacle(sensors)
    if obstacle : 
        avoidance_speed = avoidObstacle(sensors)

    left_speed = basic_speed + control_speed + avoidance_speed
    right_speed = basic_speed - control_speed - avoidance_speed

    return left_speed, right_speed
