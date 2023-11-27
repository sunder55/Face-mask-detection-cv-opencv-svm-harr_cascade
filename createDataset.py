import cv2 
import numpy as np
import os
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# import matplotlib.pyplot as plt

# print(img.shape) # height , width , color channel(RGB)
# print(img[0]) #first row of image

# while True:
#     cv2.imshow('result',img)
#     # 27 is ASCII of escape key
#     if cv2.waitKey(2) == 27: 
#         break
# cv2.destroyAllWindows()

haar_data =  cv2.CascadeClassifier(cascPathface)

# haar_data.detectMultiScale(img)

# cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)

# while True:
#     faces = harr_data.detectMultiScale(img)
#     for x,y,w,h in faces:
#         cv2.rectangle(img,(x,y),(x+w, y+h))
#     cv2.imshow('result',img)
#     # 27 is ASCII of escape key
#     if cv2.waitKey(2) == 27: 
#         break
# cv2.destroyAllWindows()

capture = cv2. VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y),(x+w, y+h),(255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data) < 400:
                data.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data) >=400:
            break
        
capture.release()
cv2.destroyAllWindows()
        
np.save('400Nomask.npy',data)
