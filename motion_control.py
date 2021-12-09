import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time 

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

def calculation_distance_and_angle(position_thymio, position_goal):
    t_x = position_thymio[0]
    t_y = position_thymio[1]
    g_x = position_goal[0]
    g_y = position_goal[1]
    distance = np.around(np.sqrt(np.power(t_x - g_x, 2) + np.power(t_y - g_y, 2)), 3)
    if ((position_goal[0] == position_thymio[0]) and (position_goal[1] - position_thymio[1] > 0)):
        angle = np.pi / 2
    if ((position_goal[0] == position_thymio[0]) and (position_goal[1] - position_thymio[1] < 0)):
        angle = (np.pi * 3) / 2
    angle = np.around(np.arctan((position_goal[1] - position_thymio[1]) / (position_goal[0] - position_thymio[0])), 3)
    if (position_thymio[0]>position_goal[0] and position_thymio[1]<position_goal[1]):
        angle = angle + np.pi
    if (position_thymio[0]>position_goal[0] and position_thymio[1]>position_goal[1]):
        angle = angle + np.pi
    if (position_thymio[0]<position_goal[0] and position_thymio[1]>position_goal[1]):
        angle = angle + 2*np.pi
    return distance, angle

def check_obstacle(sensor):
    print("checkObstacle")    

    obstacle_trigger = 0
    obstacle = False

    for i in range(len(sensor)-2) :
        if sensor[i] > obstacle_trigger :
            obstacle = True

    return obstacle

def local_avoidance(sensor):

    w = [80,  80, -40, -80, -80]

    # Scale factors for sensors and constant factor
    sensor_scale = 100
    avoidance_speed = 0

    for i in range(len(sensor)-2) :
        # Get and scale inputs
        sensor[i] = sensor[i] // sensor_scale
            
        # Compute outputs of neurons and set motor powers
        avoidance_speed += sensor[i] * w[i]


    return avoidance_speed

def pd_controller(orientation, old_orientation, goal_orientation) :

    kp = 15
    kd = 7

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
         

    control_speed = kp * error + kd * (error-old_error)


    return control_speed


def motion_control(theta, old_theta, goal_theta, sensors) :

    basic_speed = 150
    avoidance_scale = 0.2
    control_speed = pd_controller(theta, old_theta, goal_theta)

    avoidance_speed = 0

    obstacle = check_obstacle(sensors)
    if obstacle : 
        avoidance_speed = local_avoidance(sensors)

    
    left_speed = basic_speed + 10 * control_speed + avoidance_speed*avoidance_scale
    right_speed = basic_speed - 10 * control_speed - avoidance_speed*avoidance_scale

    return left_speed, right_speed
