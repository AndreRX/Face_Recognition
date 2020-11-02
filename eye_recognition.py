import cv2
import numpy as np
import time

def redim(img, largura):
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation = cv2.INTER_AREA)
    return img

df = cv2.CascadeClassifier('haarcascade_eye.xml')
img_path = 'C:\\Users\\andre\\Desktop\\Face_Recognition\\images\\img'
aux = []
t_end = time.time() + 60*0.5  # 60 * minutes program will run

camera = cv2.VideoCapture(0)  #Opens WebCam

while time.time()<t_end:
    (sucess, frame) = camera.read()
    if not sucess:
        break
    
    frame = redim(frame, 300)
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = df.detectMultiScale(frame_bw, scaleFactor = 1.3, minNeighbors=3)
    frame_temp = frame.copy()

    if faces != []:   #if an eye is recognized 
        #test = cv2.resize(frame_bw, (28, 28), interpolation = cv2.INTER_AREA)  # makes the image into 784 pixels, just with wanted converted and not images
        #resize = test.flatten().reshape((1, 784))   # making the image into an array
        aux.append(frame)

    for (x, y, lar, alt) in faces:
        cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0, 255, 0), 2)   # make the rectangle in the eye
    
    cv2.imshow("finding faces", redim(frame_temp, 640))
    
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

camera.release()
cv2.destroyAllWindows()
cont = 0
for img in aux:
    cv2.imwrite(img_path +str(cont)+'.jpg',img)
    cont+=1