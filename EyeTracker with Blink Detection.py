# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 18:58:24 2018

@author: Faysal
"""

from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import imutils
import time
import dlib
import cv2
import numpy as np

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 10
COUNTER = 0
numerator = 0
denominator = 0
cx = 0
cy = 0

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

def get_face_shape(rects, frame):
    try:
        if len(rects)>0:
            shape = predictor(frame, rects[0])       
            shape = face_utils.shape_to_np(shape)
            
            return shape
        else:
            print("No faces")
            pass
        
    except Exception as e:
        print(e)
        pass

def get_eye_box(frame, shape, i):
    #Adjustment factor
    epsilon = 5
    
    #Eye Bounding Box
    eyeX = shape[i][0] - epsilon
    eyeY = int((shape[i+1][1] + shape[i+2][1])/2) - epsilon
    eyeW = shape[i+3][0] - shape[i][0] + 2*epsilon
    eyeH = int((shape[i+4][1]+shape[i+5][1])/2 - (shape[i+1][1] + shape[i+2][1])/2) + 2*epsilon
            
    eye_box = frame[eyeY:eyeY + eyeH, eyeX:eyeX + eyeW]
    
    #Drawing bounding box
    cv2.rectangle(frame, (eyeX, eyeY), (eyeX + eyeW, eyeY + eyeH), (0, 255, 0), 1)
    
    return eye_box

def find_centroid(eye):
    gray = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    thres = cv2.inRange(equ,0,20)
    kernel = np.ones((3,3),np.uint8)
    global cx, cy
    global numerator, denominator
    
    #Denoising the ROI
    dilation = cv2.dilate(thres,kernel,iterations = 2)
    #Decreasing the size of white region
    erosion = cv2.erode(dilation,kernel,iterations = 3)
    
    #Finding Contours. In this case the iris area
    image, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    if len(contours)==2 :
        numerator += 1
        M = cv2.moments(contours[1])
        if M['m00']!=0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.line(eye,(cx,cy),(cx,cy),(0,0,255),3)

    elif len(contours)==1:
        numerator += 1              
        M = cv2.moments(contours[0])
        if M['m00']!=0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.line(eye,(cx,cy),(cx,cy),(0,0,255),3)
                         
    else:
        denominator += 1
        
    return cx, cy

    
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4]) 
	C = dist.euclidean(eye[0], eye[3])
 
	ear = (A + B) / (2.0 * C)
    
	return ear


def check_blink(eye_points):
    global COUNTER
    EAR = eye_aspect_ratio(eye_points)
    
    if EAR < EYE_AR_THRESH:
        COUNTER += 1
    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            print("Eye blinked")
            COUNTER = 0


def main():
    vs = VideoStream(src=1).start()
    time.sleep(1.0)
    
    while True:
        frame = vs.read()
        frame = cv2.flip(frame, 1)        
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 0)
        
        shape = get_face_shape(rects, gray)
        
        if shape is not None:
            l_eye = get_eye_box(frame, shape, 36)
#            r_eye = get_eye_box(frame, shape, 42)
            
            cnx, cny = find_centroid(l_eye)   
            print(cnx, cny)
            
            leftEye = shape[rStart:rEnd]
#            rightEye = shape[lStart:lEnd]
            check_blink(leftEye)

            
        cv2.imshow("Frame", frame)        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    print ("accurracy=",(float(numerator)/float(numerator+denominator))*100)
    vs.stop()
    cv2.destroyAllWindows()
       
if __name__ == "__main__":
    main()