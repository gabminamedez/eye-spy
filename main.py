import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_eye.xml')
left_eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_righteye_2splits.xml')

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

if not cam.isOpened():
    raise IOError('Cannot open webcam!')
  
while True:
    ret, frame = cam.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.putText(frame, 'Face', (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)

    cv.imshow('Eye Spy', frame) 
    
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

cam.release()
cv.destroyAllWindows()