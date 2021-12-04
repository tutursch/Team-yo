import numpy as np
import cv2
from numpy.lib.twodim_base import mask_indices
from VisGraph import *
import time

VideoCap = cv2.VideoCapture(0)

class KalmanFilter(object):
    def __init__(self, dt, point):
        self.dt=dt

        # Vecteur d'etat initial
        self.E=np.matrix([[point[0]], [point[1]], [0], [0]])

        # Matrice de transition
        self.A=np.matrix([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        # Matrice d'observation, on observe que x et y
        self.H=np.matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0]])

        self.Q=np.matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        self.R=np.matrix([[1, 0],
                          [0, 1]])

        self.P=np.eye(self.A.shape[1])

    def predict(self):
        self.E=np.dot(self.A, self.E)
        # Calcul de la covariance de l'erreur
        self.P=np.dot(np.dot(self.A, self.P), self.A.T)+self.Q
        return self.E

    def update(self, z):
        # Calcul du gain de Kalman
        S=np.dot(self.H, np.dot(self.P, self.H.T))+self.R
        K=np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Correction / innovation
        self.E=np.round(self.E+np.dot(K, (z-np.dot(self.H, self.E))))
        I=np.eye(self.H.shape[1])
        self.P=(I-(K*self.H))*self.P

        return self.E


#lo_blue=np.array([80, 50, 50])
#hi_blue=np.array([100, 255, 255])
#green = 70
#lo_green = np.array([green-10, 100, 50])
#hi_green = np.array([green+10, 255, 255])

#def detect_inrange(image, surface, lo, hi):
#    points=[]
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#    image=cv2.blur(image, (5, 5))
#    mask=cv2.inRange(image, lo, hi)
#    mask=cv2.erode(mask, None, iterations=2)
#    mask=cv2.dilate(mask, None, iterations=2)
#    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#    elements=sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
#    for element in elements:
#        if cv2.contourArea(element)>surface:
#            ((x, y), rayon)=cv2.minEnclosingCircle(element)
#            points.append(np.array([int(x), int(y)]))
#        else:
#            break
   

#    return points, mask

def init_corner():

    ret, frame = VideoCap.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    corners = cv2.goodFeaturesToTrack(gray_img, 45, 0.1, 10)
    corners = np.int0(corners)

    corner_pos  = []

    for corner in corners:
        x, y = corner.ravel()
        corner_pos.append(np.array([x,y]))

    return corner_pos

#KF = KalmanFilter(0.1, [0,0])

#def open_cam():
#    ret, frame = VideoCap.read()
    
#    points_b, mask_b = detect_inrange(frame, 800, lo_blue, hi_blue)
#    points_g, mask_g = detect_inrange(frame, 800, lo_green, hi_green)  
        
#    etat = KF.predict().astype(np.int32)

#    cv2.circle(frame, (int(etat[0]), int(etat[1])), 2, (0, 255, 0), 5)
#    #cv2.arrowedLine(frame, 
#    #                (etat[0], etat[1]), (etat[0]+etat[2], etat[1]+etat[3]),
#    #                color = (0, 255, 0),
#    #                thickness=3,
#    #                tipLength=0.2)
#    if(len(points_b)>0):
#        cv2.circle(frame, (points_b[0][0], points_b[0][1]), 10, (0, 0, 255), 2)
#        KF.update(np.expand_dims(points_b[0], axis=-1))

#    if(len(points_g)>0):
#        cv2.circle(frame, (points_g[0][0], points_g[0][1]), 10, (255, 0, 0), 2)

#    cv2.imshow('image', frame)
#    cv2.imshow('mask blue', mask_b)
#    #cv2.imshow('mask green', mask_g)
#    cv2.waitKey(0)
#    return 1

def polygon(corner_pos):

    ret, frame = VideoCap.read()
    threshold = 30
    #Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_polys = []
    index_ok = []

    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    poly_found = 0
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
                            if(k not in index_ok):
                                index_ok.append(l)
                                corner_pos[l][0] = (corner_pos[l][0]-cx)*2+cx
                                corner_pos[l][1] = (corner_pos[l][1]-cy)*2+cy
                                polys.append(corner_pos[l])
                               
        if (len(polys) != 0):
            all_polys.append(polys)
    
    return all_polys

def detect_start_stop ():
    start=[]
    stop=[]
    while(len(stop)==0 or (len(start)==0)): #remettre len(start)
        ret, frame = VideoCap.read()
        print("No detection...")
        color_start = 25
        color_stop = 70
        lo_start = np.array([color_start-10, 80, 50])
        hi_start = np.array([color_start+10, 120,110])
        lo_stop = np.array([color_stop-15, 70, 30])
        hi_stop = np.array([color_stop+15, 140,100])
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
 
def initialisation():
    ret, frame = VideoCap.read()

    corners_pos = init_corner()

    poly = polygon(corners_pos)

    start, stop = detect_start_stop()

    all_polys_point = []

    #transformation into the class point
    for i in range(len(poly)):
        polys_point = []
        for j in range(len(poly[i])):
            polys_point.append(Point(poly[i][j][0], poly[i][j][1]))
        all_polys_point.append(polys_point)
    start_point = Point(start[0][0], start[0][1])  
    stop_point = Point(stop[0][0], stop[0][1]) 

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
