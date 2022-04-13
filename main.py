import cv2
import cvzone
import numpy as np
from tkinter import Frame
from charset_normalizer import detect
from keras.models import load_model
from cvzone.FaceMeshModule import FaceMeshDetector
 

model=load_model("model2-001.model")
results={0:'With Mask',1:'Without Mask'}
GR_dict={0:(0,255,0),1:(0,0,255)}
rect_size = 4
cap = cv2.VideoCapture(0) 
 
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = FaceMeshDetector(maxFaces=100) 

while True:
    (rval, Frame) = cap.read()
    Frame=cv2.flip(Frame,1,1) 
    Frame, detect = detector.findFaceMesh(Frame, draw=False)
    rerect_size = cv2.resize(Frame, (Frame.shape[1] // rect_size, Frame.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    
    if detect:
        face = detect[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        
        W = 6.3
        f = 840
        d = (W * f)/ w

    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
         
        face_img = Frame[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
 
        label=np.argmax(result,axis=1)[0]
        depth = f'Depth: {int(d)}cm'
        text = "{}: {:.0f}%".format(results[label], np.max(result) * 100)
        print(text, depth)

        cv2.rectangle(Frame,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(Frame,(x,y),(x+w,y),GR_dict[label],-1)
 
        cv2.putText(Frame, depth, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(Frame, text, (x, y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
 
    cv2.imshow('LIVE', Frame)
     
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
 
cap.release()
 
cv2.destroyAllWindows()