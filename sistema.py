from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


classificador_face = cv2.CascadeClassifier(r'C:\Users\luan-\OneDrive\Documentos\TCC\TCC_sistema\haarcascade_frontalface_default.xml')
classificador = load_model(r'C:\Users\luan-\OneDrive\Documentos\TCC\TCC_sistema\model.h5')
emoções = ['Raiva','Nojo','Medo','Feliz','Neutro', 'Triste', 'Surpreso']
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rostos = classificador_face.detectMultiScale(gray)

    for (x,y,w,h) in rostos:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        face_gray = gray[y:y+h,x:x+w]
        face_gray = cv2.resize(face_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([face_gray])!=0:
            roi = face_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classificador.predict(roi)[0]
            label=emoções[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),3)

    cv2.imshow('Detectando emoções',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
