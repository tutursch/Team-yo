import numpy as np
import cv2
from numpy.lib.twodim_base import mask_indices
from vis_graph import *


# Initialization coordinates of the obstacles' corners
def init_corner(frame):

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    corners = cv2.goodFeaturesToTrack(gray_img, 45, 0.1, 10)
    corners = np.int0(corners)

    corner_pos  = []

    for corner in corners:
        x, y = corner.ravel()
        corner_pos.append(np.array([x,y]))

    return corner_pos


# Association of each corner to its corresponding polygon
def polygon(corner_pos, frame):

    threshold = 30
    safety_margin_coeff = 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_polys = []
    index_ok = []

    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    k = 0

    for i in range(len(contours)):
        polys = []
        M = cv2.moments(contours[i])
        if (M['m00']!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        for j in range (len(contours[i])):
                for l in range(len(corner_pos)):
                    if((corner_pos[l][0]>(contours[i][j][0][0]-threshold)) and (corner_pos[l][0]<(contours[i][j][0][0]+threshold))):
                         if((corner_pos[l][1]>(contours[i][j][0][1]-threshold)) and (corner_pos[l][1]<(contours[i][j][0][1]+threshold))):

                            k = l
                            
                            # Shift ot the corners' position by a safety margin that is at least half Thymio's width 
                            if(k not in index_ok):
                                index_ok.append(l)
                                corner_pos[l][0] = (corner_pos[l][0]-cx)*safety_margin_coeff+cx
                                corner_pos[l][1] = (corner_pos[l][1]-cy)*safety_margin_coeff+cy
                                polys.append(corner_pos[l])
                               
        if (len(polys) != 0):
            all_polys.append(polys)
    
    return all_polys

# Computation the coordinates of our start and stop
def detect_start_stop (frame):
    start=[]
    stop=[]
    color_start = 105
    color_stop = 70
    lo_start = np.array([color_start-10, 135-25, 65-15])
    hi_start = np.array([color_start+10, 135+25, 65+15])
    lo_stop = np.array([color_stop-20, 100-15, 50-10])
    hi_stop = np.array([color_stop+20, 100+15, 50+10])
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image = cv2.blur(image, (5, 5))
    mask_start = cv2.inRange(image, lo_start, hi_start)
    mask_stop = cv2.inRange(image, lo_stop, hi_stop)
    mask_start=cv2.erode(mask_start, None, iterations=2)
    mask_start=cv2.dilate(mask_start, None, iterations=2)
    mask_stop=cv2.erode(mask_stop, None, iterations=2)
    mask_stop=cv2.dilate(mask_stop, None, iterations=2)
    elements_start=cv2.findContours(mask_start, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elements_start=sorted(elements_start, key=lambda x:cv2.contourArea(x), reverse=True)
    elements_stop=cv2.findContours(mask_stop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elements_stop=sorted(elements_stop, key=lambda x:cv2.contourArea(x), reverse=True)
    if len(elements_start) > 0:
        c=max(elements_start, key=cv2.contourArea)
        ((x, y), radius)=cv2.minEnclosingCircle(c)
        if (len(start)==0):
            start.append(np.array([int(x), int(y)]))
    if len(elements_stop) > 0:
        c=max(elements_stop, key=cv2.contourArea)
        ((x, y), radius)=cv2.minEnclosingCircle(c)
        if (len(stop)==0):
            stop.append(np.array([int(x), int(y)]))
    return start, stop


def initialisation(frame, start, stop):
    corners_pos = init_corner(frame)

    poly = polygon(corners_pos, frame)

    all_polys_point = []

    # transformation into the class point
    for i in range(len(poly)):
        polys_point = []
        for j in range(len(poly[i])):
            polys_point.append(Point(poly[i][j][0], poly[i][j][1]))
        all_polys_point.append(polys_point)
    start_point = Point(start[0][0], start[0][1])  
    stop_point = Point(stop[0][0], stop[0][1]) 

    #computation of the shortest path
    g = VisGraph()
    g.build(all_polys_point)
    shortest = g.shortest_path(start_point, stop_point)

    for corner in corners_pos:
        x, y = corner.ravel()
        cv2.circle(frame, (x,y), 3, 255, -1)

    for i in range (len(poly)) :
        for j in range (len(poly[i])) :
            cv2.circle(frame, (poly[i][j][0], poly[i][j][1]), 3, 255, -1)
        
    for i in range(len(shortest)):
        cv2.circle(frame, (int(shortest[i].return_x()), int(shortest[i].return_y())), 10, (0, 0, 255), 2)

    cv2.imshow('image', frame)
    cv2.waitKey(0)

    #transformation into numpy
    for i in range (len(shortest)):
        shortest[i] = [shortest[i].return_x(), shortest[i].return_y()]

    return shortest
