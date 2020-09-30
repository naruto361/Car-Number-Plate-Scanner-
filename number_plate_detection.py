import cv2 as cv
import numpy as np
path='resources/car2.png'

img=cv.imread(path)
img=cv.resize(img,(640,400))
minArea=2000
nPlateCascade=cv.CascadeClassifier('resources\haarcascade_russian_plate_number.xml')
#while 1:
    
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
numberPlates=nPlateCascade.detectMultiScale(gray)
for (x,y,w,h) in numberPlates:
    area=w*h
    if area>minArea:
        cv.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
        cv.putText(img,'Number Plate',(x,y-10),cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
        imgRoi=img[y:y+h,x:x+w]
cv.imshow('Result',imgRoi)       
            
cv.imshow('image',img)
cv.waitKey(0)
'''    k=cv.waitKey(1)
    if k==27:
        break'''
cv.destroyAllWindows()