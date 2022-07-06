import numpy as np
import cv2

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier =cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if len (faces) == 0 :
    print("Wajah Tidak Terdeksi")
    
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('Wajah Terdeteksi',img)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    
    cv2.waitKey(0)
    
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow('Wajah Terdeteksi',img)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()
cv2.waitKey(0)

