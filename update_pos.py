import numpy as np
import cv2
from numpy.lib.twodim_base import mask_indices

VideoCap = cv2.VideoCapture(0)

def detect_thymio():
    top=[]
    bottom=[]
    ret, frame = VideoCap.read()
    color_top = 130
    color_bottom = 20
    lo_top = np.array([color_top-5, 100, 50])
    hi_top = np.array([color_top+5, 255,255])
    lo_bottom = np.array([color_bottom-5, 100, 50])
    hi_bottom = np.array([color_bottom+5, 255,255])
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
        top.append(np.array([int(x), int(y)]))
    if len(elements_bottom) > 0:
        c=max(elements_bottom, key=cv2.contourArea)
        ((x, y), radius)=cv2.minEnclosingCircle(c)
        bottom.append(np.array([int(x), int(y)]))
    delta_x = top[0][0] - bottom[0][0]
    delta_y = top[0][1] - bottom[0][1]
    angle = np.arctan2(delta_y/delta_x)

    return top, bottom, angle
