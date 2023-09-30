import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultHumanDetector())

cv2.startWindowThread()
cap = cv2.VideoCapture(0)

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('*MJPG'), 15., (640,480))

while True:
  ret, frame = cap.read()
  frame = cv2.resize(frame, (640,480))
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  box, weight = hog.detectMultiScale(frame, winStride=(8,8))
  box = np.array([[x,y,x+w,y+h] for x,y,w,h in box])
  for (xA, yA, xB, yB) in box:
    cv2.rectangle(frame, (xA, yA), (xB, yB), (0,255,0), 2)
  out.write(frame.astype('uint8'))
  cv2.imshow('frame',frame)
  if cv2.WaitKey(1) & 0xFF == ord('q'):
    break

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.WaitKey(1)
