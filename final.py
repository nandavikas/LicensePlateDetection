from __future__ import print_function
from PIL import Image
from pytesseract import image_to_string
import numpy as np
import imutils
import cv2 
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,help = "Enter car image to detect car number")
args = vars(ap.parse_args())
audi = cv2.imread(args["image"])
#dim = (1566, 491)
#resize = imutils.resize(audi,width=500)
#resize= cv2.resize(audi, dim, interpolation=cv2.INTER_AREA)
img = audi
gray = cv2.cvtColor(audi, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("Image", gray)
#cv2.waitKey(0)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
top = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,rectKernel)
#cv2.imshow("black hat",top)
#cv2.waitKey(0)

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
#cv2.imshow("tophat",tophat)
#cv2.waitKey(0)




(T, threshInv) = cv2.threshold(top, 30, 255, cv2.THRESH_BINARY)
#cv2.imshow("Threshold Binary Inverse", threshInv)
#cv2.waitKey(0)


im, cnts, hierarchy = cv2.findContours(threshInv.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	#cv2.rectangle(audi,(x,y),(x+w,y+h),(0,0,255))
	#cv2.imshow("suma", resize)
	#cv2.waitKey(0)
	#print (w,h)
	if (w<321 and h<70 and w>315 and h >62):
		crop = audi[y:y+h,x:x+w
]		license_plate = img[y:y+h,x:x+w]
		#cv2.imshow("License Plate", crop)
		#cv2.waitKey(0)
		
	


grayplate = cv2.cvtColor(license_plate,cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
tophat = cv2.morphologyEx(grayplate,cv2.MORPH_TOPHAT,kernel)
(T,thr) = cv2.threshold(grayplate,100,150,cv2.THRESH_BINARY_INV)
#tight = cv2.Canny(thr, 10, 250)
auto = imutils.auto_canny(thr)
#cv2.imshow("threshold number plate",thr)
#cv2.waitKey(0)
cv2.imwrite("license.jpg",thr)

result = image_to_string(Image.open("license.jpg"))
f = open('file.txt','w')
output = str()
for y in (str(result.encode('utf-8'))).split("\n"):
	output = output + str(y)
output = output[4:13]
print(output)
#print(len(output))


f.write(output)