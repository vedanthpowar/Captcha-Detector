import numpy as np
import numpy
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
model=tf.keras.models.load_model(r"C:\Users\hp pavelion\OneDrive\Desktop\Deep_Dive_Round1\model.h5")
def main(img_path):
    img=cv2.imread(img_path,0)
#     print(img)
    img_stack=segmentation(img)
    t=""
    for i in img_stack:
        # plt.imshow(i,cmap="gray")
        plt.show()
        x=np.reshape(i,(28,28,1))/255
        y=[x]
        y=np.array(y)
        result=np.argmax(model.predict(y))
        # print(class_mapping[result])
        t+=class_mapping[result]
    print(t)
def segmentation(photo):
    blur = cv2.GaussianBlur(photo,(3,3),2)
#     blur=photo
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plt.imshow(blur,cmap="gray")
    my_segmented_stack=[]
    start=0
    for i in range(0,len(th3[0])):
        flag=0
        count=0
        for j in range(0,len(th3)):
            if(th3[j][i]==0):
                count=count+1
        if(count>6):
            flag=1
        if(flag==1 and start==0):
            xi=i
            start=1
        if(flag==0 and start==1):
            xf=i
    #         print(xi,xf)
            start=0
    #         print(xf,xi)
            if(xf-xi>len(th3[0])/50):
                margin=20
                cropped = th3[0:len(th3[0])-2,max(0,xi-margin):min(len(th3[0]-1),xf+margin)]
                # print(cropped)
                count2=0
                for k in range(0,len(cropped)):
                    for l in range(0,len(cropped[0])):
                        if(cropped[k][l]==0):
                            count2=count2+1
                    if(count2>0):
                        yi=k
                        break
                count2=0
                for k in range(len(cropped)-1,-1,-1):
                    for l in range(0,len(cropped[0])):
                        if(cropped[k][l]==0):
                            count2=count2+1
                    if(count2>0):
                        yf=k
                        break
    #             print(yi,yf)
                
                cropped = th3[max(yi-margin,0):min(yf+margin,len(th3)-1),  max(0,xi-margin):min(len(th3[0]-1),xf+margin)]
#                 cropped = th3[yi-20:yf+20,xi-20:xf+20]
                print(yi,yf,xi,xf)
                kernel = np.ones((5,5), np.uint8)
                cropped = cv2.erode(cropped, kernel, iterations=1)
                cropped=cv2.resize(cropped,(28,28))
                cropped=cv2.bitwise_not(cropped)
#                 cropped = cv2.GaussianBlur(cropped,(3,3),0)
                my_segmented_stack.append(cropped)
    return my_segmented_stack
        
            # start=0
def segmentation(photo):
    blur = cv2.GaussianBlur(photo,(3,3),2)
#     blur=photo
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plt.imshow(blur,cmap="gray")
    my_segmented_stack=[]
    start=0
    for i in range(0,len(th3[0])):
        flag=0
        count=0
        for j in range(0,len(th3)):
            if(th3[j][i]==0):
                count=count+1
        if(count>6):
            flag=1
        if(flag==1 and start==0):
            xi=i
            start=1
        if(flag==0 and start==1):
            xf=i
    #         print(xi,xf)
            start=0
    #         print(xf,xi)
            if(xf-xi>len(th3[0])/50):
                margin=20
                cropped = th3[0:len(th3[0])-2,max(0,xi-margin):min(len(th3[0]-1),xf+margin)]
                # print(cropped)
                count2=0
                for k in range(0,len(cropped)):
                    for l in range(0,len(cropped[0])):
                        if(cropped[k][l]==0):
                            count2=count2+1
                    if(count2>0):
                        yi=k
                        break
                count2=0
                for k in range(len(cropped)-1,-1,-1):
                    for l in range(0,len(cropped[0])):
                        if(cropped[k][l]==0):
                            count2=count2+1
                    if(count2>0):
                        yf=k
                        break
    #             print(yi,yf)
                
                cropped = th3[max(yi-margin,0):min(yf+margin,len(th3)-1),  max(0,xi-margin):min(len(th3[0]-1),xf+margin)]
#                 cropped = th3[yi-20:yf+20,xi-20:xf+20]
#                 print(yi,yf,xi,xf)
                kernel = np.ones((5,5), np.uint8)
                cropped = cv2.erode(cropped, kernel, iterations=1)
                cropped=cv2.resize(cropped,(28,28))
                cropped=cv2.bitwise_not(cropped)
#                 cropped = cv2.GaussianBlur(cropped,(3,3),0)
                my_segmented_stack.append(cropped)
    return my_segmented_stack
        
            # start=0
class_mapping=["A","1","2","3","4","5","6","7","B","E","F","G","H","I","N","O","P","R","S","T","Z","Y","g","h","t","w"]

main(r"C:\Users\hp pavelion\Downloads\IMG_20220412_193053.jpg")