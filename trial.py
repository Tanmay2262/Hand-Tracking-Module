import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModule as htm

prev_time = 0
curr_time = 0
cap = cv.VideoCapture(0)
detector = htm.handDetector()


while True:
    ret,img = cap.read()
    img = detector.findHands(img,draw = False)
    
    lm_list = detector.findPosition(img,draw=False)
    if len(lm_list) != 0:
        print(lm_list[4])

    # frame rate
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_TRIPLEX,3,(255,0,255),3)
    
    cv.imshow("Image",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break