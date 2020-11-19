import numpy as np
import cv2 as cv
from gtts import gTTS
import random
import playsound
import os


face_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
left_eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_righteye_2splits.xml')

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

if not cam.isOpened():
    raise IOError('Cannot open webcam!')

tts = gTTS(text='Welcome to Eye Spy!', lang='en')
rand = random.randint(1,20000000)
audio_file = 'audio' + str(rand) + '.mp3'
tts.save(audio_file)
playsound.playsound(audio_file)
os.remove(audio_file)
  
while True:
    ret, frame = cam.read()

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    if len(faces) >= 1:
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv.putText(frame, 'Face', (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)

        left_face = frame[y:y+h, x+int(w/2):x+w]
        left_face_gray = frame_gray[y:y+h, x+int(w/2):x+w]
        left_eye = left_eye_cascade.detectMultiScale(left_face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (ex, ey, ew, eh) in left_eye:
            cv.rectangle(left_face, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)

        right_face = frame[y:y+h, x:x+int(w/2)]
        right_face_gray = frame_gray[y:y+h, x:x+int(w/2)]
        right_eye = right_eye_cascade.detectMultiScale(right_face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (ex, ey, ew, eh) in right_eye:
            cv.rectangle(right_face, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)

    cv.imshow('Eye Spy', frame) 
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()