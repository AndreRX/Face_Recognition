import cv2
import numpy as np
import time

def redim(img, largura):
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation = cv2.INTER_AREA)
    return img


df = cv2.CascadeClassifier('haarcascade_eye.xml')

camera = cv2.VideoCapture(0)

test1 = []
t_end = time.time() + 60*0.5

while time.time()<t_end:
    (sucesso, frame) = camera.read()
    if not sucesso:
        break
    
    frame = redim(frame, 300)
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = df.detectMultiScale(frame_pb, scaleFactor = 1.3, minNeighbors=3)
    frame_temp = frame.copy()

    if faces != []:
        test = cv2.resize(frame_pb, (28, 28), interpolation = cv2.INTER_AREA)
        resize = test.flatten().reshape((1, 784))
       
        test1.append(resize)




    for (x, y, lar, alt) in faces:
        cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
    
    cv2.imshow("Encontrando faces...", redim(frame_temp, 640))
    
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break


print(test1[0][0])
camera.release()
cv2.destroyAllWindows()