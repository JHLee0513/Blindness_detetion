import os
import pandas as pd
import numpy as np 
import time

import cv2
from PIL import Image

#https://www.kaggle.com/jtbontinck/cnn-xgb-end-to-end-0-11
#inspired by https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

def crop_image(img,tol=7):
    w, h = img.shape[1],img.shape[0]
    gray_img = img
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    gray_img = (quantile_transform(gray_img, n_quantiles=256, random_state=0, copy=True)*256).astype(int)
    mask2 = gray_img<tol
    xp = (gray_img.mean(axis=0)>tol)
    yp = (gray_img.mean(axis=1)>tol)
    x1, x2 = np.argmax(xp), w-np.argmax(np.flip(xp))
    y1, y2 = np.argmax(yp), h-np.argmax(np.flip(yp))
    if x1 >= x2 or y1 >= y2 : # something wrong with the crop
        return img # return original image
    else:
        img1=img[y1:y2,x1:x2,0]
        img2=img[y1:y2,x1:x2,1]
        img3=img[y1:y2,x1:x2,2]
        img = np.stack([img1,img2,img3],axis=-1)
    return img

def process_image(image, size=512):
    image = cv2.resize(image, (size,int(size*image.shape[0]/image.shape[1])))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        image = crop_image(image, tol=7)
    except Exception as e:
        image = image
        print( str(e) )
    return image