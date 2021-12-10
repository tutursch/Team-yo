import numpy as np
import cv2
from numpy.lib.twodim_base import mask_indices
from init_map import *
from motion_control import *
from kalman_filter import *

def detect_thymio(pos_1, angle_1, motor_left, motor_right, KF, frame):
    top=[]
    bottom=[]
    middle = [0, 0]
    i = 0
    angle = 0
    while(i < 2 and (len(bottom) == 0 or len(top) == 0)):
        i = i+1
        color_top = 13
        color_bottom = 175
        lo_top = np.array([color_top-10, 50, 80])
        hi_top = np.array([color_top+10, 110, 165])
        lo_bottom = np.array([color_bottom-10, 110, 70])
        hi_bottom = np.array([color_bottom+10, 200, 165])
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        image = cv2.blur(image, (5, 5))
        mask_top = cv2.inRange(image, lo_top, hi_top)
        mask_bottom = cv2.inRange(image, lo_bottom, hi_bottom)
        mask_top = cv2.erode(mask_top, None, iterations=2)
        mask_top = cv2.dilate(mask_top, None, iterations=2)
        mask_bottom = cv2.erode(mask_bottom, None, iterations=2)
        mask_bottom = cv2.dilate(mask_bottom, None, iterations=2)
        elements_top = cv2.findContours(mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        elements_top = sorted(elements_top, key=lambda x:cv2.contourArea(x), reverse=True)
        elements_bottom = cv2.findContours(mask_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        elements_bottom = sorted(elements_bottom, key=lambda x:cv2.contourArea(x), reverse=True)
        if len(elements_top) > 0:
            c=max(elements_top, key=cv2.contourArea)
            ((x, y), radius)=cv2.minEnclosingCircle(c)
            if(len(top)==0):
                top.append(np.array([int(x), int(y)]))
        if len(elements_bottom) > 0:
            c=max(elements_bottom, key=cv2.contourArea)
            ((x, y), radius)=cv2.minEnclosingCircle(c)
            if(len(bottom)==0):
                bottom.append(np.array([int(x), int(y)]))
        if (len(bottom)>0 and len(top)>0):
            delta_x = top[0][0] - bottom[0][0]
            delta_y = top[0][1] - bottom[0][1]
            middle[0] = (bottom[0][0]+top[0][0])/2
            middle[1] = (bottom[0][1]+top[0][1])/2

            if (delta_x !=0):
                angle = np.arctan2(delta_y, delta_x)
            else:
                angle = np.pi/2
            if(angle < 0):
                angle = angle + 2*np.pi

    state = KF.predict(pos_1, angle_1, motor_left, motor_right)
    pos = [state[0], state[1]]
    angle_kalman = state[2]

    if not(len(bottom) == 0 or len(top) == 0):
        z = [[middle[0]], [middle[1]], [angle]]
        state_up = KF.update(z)
        pos_up = [state_up[0], state_up[1]]
        angle_up = state_up[2]
        return pos_up, angle_up

    return pos, angle_kalman