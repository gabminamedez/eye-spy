import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise IOError("Cannot open webcam!")
  
while True:
    ret, frame = cam.read() 
    cv2.imshow('Eye Spy', frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cam.release()
cv2.destroyAllWindows()