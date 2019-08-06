# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 06:40:29 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:32:36 2019

@author: HP
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2

#get RGBs
X = [0,0,0]; Y = [0]
for pic in range(46):
    var1 = str(pic+1)+'.png'
    var2 = 'm'+str(pic+1)+'.jpg'
    img = cv2.imread(var1)
    mimg = cv2.imread(var2)
    B,G,R = cv2.split(img)
    img = cv2.merge([R,G,B])
    Xtemp = np.column_stack((np.reshape(np.array(R),(-1,1)),np.reshape(np.array(G),(-1,1)),np.reshape(np.array(B),(-1,1))))
    Xtemp = np.array(Xtemp)
    X = np.vstack((X,Xtemp))
    Ytemp = cv2.split(mimg)
    Ytemp = np.reshape(np.array(Ytemp[0]),[-1,1])
    Y = np.vstack((Y,Ytemp))
X = X[1:len(X)]
y = Y[1:len(Y)]
y[y<=100] = 0
y[y>100] = 1
'''
y = -1 * np.ones([np.size(Y),1])
for i in range(len(Y)):
    if Y[i] > 100:
        y[i] = 1
# D = (X,y)
'''

#Logistic Regression
w_MLE = 0 * np.ones([3,1]); a = 1/len(y) #initial Max Likelihood Esti.
for j in range(7):
    Sum = 0
    for i in range(len(X)):
        if y[i] == 0:
            continue
        else:
            temp = y[i] * (1 - 1/(1 + np.exp(-y[i] * np.dot(X[i],w_MLE)))) * X[i]
            #print(temp)
            #print(i)
            Sum += temp 
    #print(Sum)
    print(w_MLE)
    w_MLE += a * np.reshape(Sum,[-1,1])


    