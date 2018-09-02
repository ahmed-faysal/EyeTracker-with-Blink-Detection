# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 18:58:24 2018

@author: Faysal
"""

from imutils.video import VideoStream
from imutils import face_utils
import imutils
import pyautogui
from scipy.spatial import distance as dist
import time
import dlib
import cv2
import numpy as np

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10
EAR = 0
COUNTER = 0
successful = 0
unsuccessful = 0
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
    eyeY = int(shape[i][1] - (abs(shape[i+3][0] - shape[i][0]))/4) - epsilon
    eyeW = shape[i+3][0] - shape[i][0] + 2*epsilon
    eyeH = int((abs(shape[i+3][0] - shape[i][0]))/2) + 2*epsilon
            
    eye_box = frame[eyeY:eyeY + eyeH, eyeX:eyeX + eyeW]
    
    #Drawing bounding box
    cv2.rectangle(frame, (eyeX, eyeY), (eyeX + eyeW, eyeY + eyeH), (0, 255, 0), 1)
    
    return eye_box, eyeW, eyeH


def find_centroid(eye):
    gray = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    thres = cv2.inRange(equ,0,20)
    kernel = np.ones((3,3),np.uint8)
    global cx, cy, successful, unsuccessful
    
    #Denoising the ROI
    dilation = cv2.dilate(thres,kernel,iterations = 2)
    #Decreasing the size of white region
    erosion = cv2.erode(dilation,kernel,iterations = 3)
    
    #Finding Contours. In this case the iris area
    image, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    if len(contours)==2 :
        successful += 1
        M = cv2.moments(contours[1])
        if M['m00']!=0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.line(eye,(cx,cy),(cx,cy),(0,0,255),3)

    elif len(contours)==1:
        successful += 1              
        M = cv2.moments(contours[0])
        if M['m00']!=0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.line(eye,(cx,cy),(cx,cy),(0,0,255),3)
                         
    else:
        unsuccessful += 1
        
    return cx, cy

    
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4]) 
	C = dist.euclidean(eye[0], eye[3])
 
	ear = (A + B) / (2.0 * C)
    
	return ear


def check_blink(eye_points, direction):
    global COUNTER, EAR
    EAR = eye_aspect_ratio(eye_points)
    
    if EAR < EYE_AR_THRESH:
        COUNTER += 1
    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            print("Eye blinked on ", direction)

            if direction == 1:
                pyautogui.press('left')
                
            elif direction == 2:
                pyautogui.press('right')
                
            else:
                pyautogui.press('space')
                
            COUNTER = 0
        else:
            COUNTER = 0


def move_cursor(cx, cy, eyeW, eyeH):
    
    direction = 0
    window_width, window_height = pyautogui.size()
    
    width_ratio = window_width/eyeW
    height_ratio = window_height/eyeH
       
    screenX = int(width_ratio * cx)
    screenY = int(height_ratio * cy)
    
    if( screenX < window_width * 0.4):
        print("looking left", screenX, screenY)
        direction = 1
        
    elif( screenX > window_width * 0.6):
        print("looking right", screenX, screenY)
        direction = 2
        
    else:
        print("Centered", screenX, screenY)
        direction = 0
        
#    print(eyeW, eyeH, screenX, screenY)   
#    print(window_width, window_height, width_ratio, height_ratio)
    
    return direction


def main():
    vs = VideoStream(src=1).start()
    time.sleep(1.0)
    direction = 0
    
    while True:
        frame = vs.read()
        frame = cv2.flip(frame, 1)        
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 0)
        
        shape = get_face_shape(rects, gray)
        
        if shape is not None:
            l_eye, eyeW, eyeH = get_eye_box(frame, shape, 36)
#            r_eye = get_eye_box(frame, shape, 42)
            
            leftEye = shape[lStart:lEnd]
#            rightEye = shape[lStart:lEnd]
            check_blink(leftEye, direction)
#            print(COUNTER)
            
            if(EAR > EYE_AR_THRESH):
                cnx, cny = find_centroid(l_eye)
    #            print(cnx, cny)
                
                direction = move_cursor(cnx, cny, eyeW, eyeH)
    #            control(cnx, cny, eyeW, eyeH)
            
            
        cv2.imshow("Frame", frame)        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        
    print ("Accurracy = ",(float(successful)/float(successful+unsuccessful))*100)
    vs.stop()
    cv2.destroyAllWindows()

      
if __name__ == "__main__":
    main()