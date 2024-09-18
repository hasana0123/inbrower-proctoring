import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame = cv.flip(frame,1)
    
    cv.putText(frame,'Press q to quit',(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv.LINE_AA)
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()