import numpy as np
import cv2
from math import sqrt
from keras.models import model_from_json

with open('model.json','r') as m:
    model = model_from_json(m.read())

model.load_weights('model.h5')



img = cv2.imread("images/image.jpg")

im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (3, 3), 0)
im_th = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,7)


c,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
rects = [cv2.boundingRect(ctr) for ctr in ctrs]    


print("Number of digits : ",len(rects))
for rect in rects:
    shift = int(sqrt(max(rect[3],rect[2])))
    pred_img = im_th[rect[1]-shift:(rect[1]+ rect[3]+shift),rect[0]-shift:(rect[0]+ rect[2]+shift)]
    pred_img = cv2.resize(pred_img,(28,28))
    pred_img = pred_img/255.0

    
    img = cv2.rectangle(img, (rect[0]-shift, rect[1]-shift), (rect[0] + rect[2]+shift, rect[1] + rect[3]+shift), (0, 255, 0), 1)
    prediction = model.predict(pred_img.reshape(1,28,28,1))
    
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img,str(np.argmax(prediction)),(rect[0],rect[1]),font,0.7,(0,0,255),2 )
    

cv2.imwrite('images2/rectangle_images.jpg',img)


