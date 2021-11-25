import numpy as np
import cv2

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


lo_blue=np.array([80, 50, 50])
hi_blue=np.array([100, 255, 255])
green = 70
lo_green = np.array([green-10, 100, 50])
hi_green = np.array([green+10, 255, 255])

def detect_inrange(image, surface, lo, hi):
    points=[]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image=cv2.blur(image, (5, 5))
    mask=cv2.inRange(image, lo, hi)
    mask=cv2.erode(mask, None, iterations=2)
    mask=cv2.dilate(mask, None, iterations=2)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elements=sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
    for element in elements:
        if cv2.contourArea(element)>surface:
            ((x, y), rayon)=cv2.minEnclosingCircle(element)
            points.append(np.array([int(x), int(y)]))
        else:
            break
   

    return points, mask

def init_corner():

    VideoCapInit = cv2.VideoCapture(1)
    ret, frame = VideoCapInit.read()
    print("test")


    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    corners = cv2.goodFeaturesToTrack(gray_img, 20, 0.1, 10)
    corners = np.int0(corners)

    corner_pos  = []

    for corner in corners:
        x, y = corner.ravel()
        corner_pos.append(np.array([x,y]))
        cv2.circle(frame, (x,y), 3, 255, -1)
    return corner_pos

VideoCap = cv2.VideoCapture(1)
KF = KalmanFilter(0.1, [0,0])

def open_cam():
    ret, frame = VideoCap.read()
    
    points_b, mask_b = detect_inrange(frame, 800, lo_blue, hi_blue)
    points_g, mask_g = detect_inrange(frame, 800, lo_green, hi_green)  
        
    etat = KF.predict().astype(np.int32)

    cv2.circle(frame, (int(etat[0]), int(etat[1])), 2, (0, 255, 0), 5)
    #cv2.arrowedLine(frame, 
    #                (etat[0], etat[1]), (etat[0]+etat[2], etat[1]+etat[3]),
    #                color = (0, 255, 0),
    #                thickness=3,
    #                tipLength=0.2)
    if(len(points_b)>0):
        cv2.circle(frame, (points_b[0][0], points_b[0][1]), 10, (0, 0, 255), 2)
        KF.update(np.expand_dims(points_b[0], axis=-1))

    if(len(points_g)>0):
        cv2.circle(frame, (points_g[0][0], points_g[0][1]), 10, (255, 0, 0), 2)

    cv2.imshow('image', frame)
    cv2.imshow('mask blue', mask_b)
    cv2.imshow('mask green', mask_g)
    if cv2.waitKey(100)==ord('q'):
        VideoCap.release()
        cv2.destroyAllWindows()
        return 0
    return 1

