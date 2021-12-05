import numpy as np
import cv2
from numpy.lib.twodim_base import mask_indices
from l42 import *
from local_nav import *

VideoCap = cv2.VideoCapture(0)

def detect_thymio():
    top=[]
    bottom=[]
 
    while(len(bottom) == 0 or len(top) == 0):
        print('No thymio')
        ret, frame = VideoCap.read()
        color_top = 13
        color_bottom = 175
        lo_top = np.array([color_top-10, 50, 110])
        hi_top = np.array([color_top+10, 110, 155])
        lo_bottom = np.array([color_bottom-10, 140, 90])
        hi_bottom = np.array([color_bottom+10, 200, 155])
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

            if (delta_x !=0):
                angle = np.arctan2(delta_y, delta_x)
            else:
                angle = np.pi/2
            if(angle < 0):
                angle = angle + 2*np.pi
           
            return bottom[0], angle

# while(True):
#     bottom, top, angle_thymio = detect_thymio()
#     start, stop = detect_start_stop()
#     distance, angle = calculation_distance_and_angle(bottom, stop[0])
#     #left_speed, right_speed = localNavigation(, , , 0) :
#     print("STOP = ")
#     print(stop)
# #    print("START = ")
# #    print(start)
# #    print("BOTTOM = ")
# #    print(bottom)
# #    print("TOP = ")
# #    print(top)
#     print("ANGLE THYMIO = :")
#     print(angle_thymio)
#     print("DISTANCE = ")
#     print(distance)
#     print("Angle = ")
#     print(angle)
#     time.sleep(5)