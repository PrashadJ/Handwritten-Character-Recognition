# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:49:28 2018

@author: Prashad
"""

import numpy as np
import cv2
##import tensorflow
#import statsmodels.api as sm
image=cv2.imread('F:\handwritten.PNG')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small=cv2.pyrDown(image)

cv2.imshow('Digits image',small)
cv2.waitKey(0)
cv2.destroyAllWindows()

cells=[np.hsplit(row,100) for row in np.vsplit(gray,50)]

x=np.array(cells)
##print(x)
print('the shape of our cell array is'+str(x.shape))

train=x[:,:70].reshape(-1,400).astype(np.float32)

test=x[:,70:100].reshape(-1,400).astype(np.float32)

k=[0,1,2,3,4,5,6,7,8,9]
train_labels=np.repeat(k,350)[:,np.newaxis]
test_labels=np.repeat(k,150)[:,np.newaxis]
#model1=sm.OLS(train)
#model2=sm.OLS(train_labels)
knn=cv2.ml.KNearest_create()


knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,distance=knn.findNearest(test,k=3)
matches=result==test_labels
correct=np.count_nonzero(matches)
accuracy=correct*(100.0/result.size)
print('accuracy is=%.2f '% accuracy+'%')


def x_cord_contour(contour):
    
    #if cv2.contourArea(contour) > 10:
    M=cv2.moments(contour)
    print(int(M['m10']/M['m00']))
    return (int(M['m10']/M['m00']) )

def makeSquare(not_square):

    BLACK=[0,0,0]
    img_dim=not_square.shape
    height=img_dim[0]
    width=img_dim[1]
    if(height==width):
        square=not_square
        return square
    else:
        doublesize=cv2.resize(not_square,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
        print(doublesize)
        height=height*2
        width=width*2
        
        if(height>width):
            pad=(height-width)/2
            doublesize_square=cv2.copyMakeBorder(doublesize,0,0,int(pad),\
                                                 int(pad),cv2.BORDER_CONSTANT,value=BLACK)
        else:
           pad=(width-height)/2
           doublesize_square=(cv2.copyMakeBorder(doublesize,int(pad),int(pad),0,0,\
                                                    cv2.BORDER_CONSTANT,value=BLACK))
    doublesize_square_dim=doublesize_square.shape
    return doublesize_square                     
        
def resize_to_pixel(dimension,image):
    buffer_pix=4
    dimension=dimension-buffer_pix
    squared=image
    r=float(dimension)/squared.shape[1]
    dim=(dimension,int(squared.shape[0]*r))
    resized=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    img_dim2=resized.shape
    height_r=img_dim2[0]
    width_r=img_dim2[1]
    BLACK=[0,0,0]
    if(height_r>width_r):
        resized=cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if(height_r<width_r):
        resized=cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p=2
    ReSizedImg=cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim=ReSizedImg.shape
    height=img_dim[0]
    width=img_dim[1]
    
    return ReSizedImg

image=cv2.imread('F:\HD3.PNG')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow('image',image)
cv2.waitKey(0)
##print('the number is '+' '.join(full_number))
cv2.destroyAllWindows()

cv2.imshow('gray',gray)

blurred=cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow('blurred',blurred)

edged=cv2.Canny(blurred,30,150)
cv2.imshow('edged',edged)

_,contours,_=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


##contours=sorted(contours,key=x_cord_contour,reverse=False)
##n = len(contours) - 1  ### For white background, the whole are is taken as the first biggest contour
cnts=[]

for cnt in contours:
    if cv2.contourArea(cnt) > 10:
        cnts.append(cnt)

contours = sorted(cnts, key = x_cord_contour, reverse = False)

full_number=[]

for c in contours:
    (x,y,w,h)=cv2.boundingRect(c)
    
    if w>=5 and h>=25:
        roi=blurred[y:y+h,x:x+w]
        ret,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
        squared=makeSquare(roi)
        final=resize_to_pixel(20,squared)
        ##cv2.imshow('final',final)
        final_array=final.reshape((1,400))
        final_array=final_array.astype(np.float32)
        ret,result,neighbour,dist=knn.findNearest(final_array,k=1)
        number=str(int(float(result[0])))
        full_number.append(number)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(image,number,(x,y+155),
        cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
        cv2.imshow('image',image)
        ##cv2.waitKey(0)

cv2.waitKey(0)
##print('the number is '+' '.join(full_number))
cv2.destroyAllWindows()