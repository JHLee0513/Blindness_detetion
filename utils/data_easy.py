'''
@BrianLee

This generates subset of data from original train/val split, filtered via a threhold
on dirt composition in the whole image. 

TODO: use symlink instead of copy

'''
from glob import glob
import numpy as np
import shutil
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

#directories of train/val dataset
train_images1 = "/Users/JoonH/Downloads/unet_data/unet_data/train_image/dirt/"
val_images1 = "/Users/JoonH/Downloads/unet_data/unet_data/val_image/dirt/"
train_labels1 = "/Users/JoonH/Downloads/unet_data/unet_data/train_mask/dirt/"
val_labels1 = "/Users/JoonH/Downloads/unet_data/unet_data/val_mask/dirt/"

#destination directory for filtered dataset
train_images = "/Users/JoonH/Downloads/unet_data/unet_data/train_image_easy/dirt/"
val_images = "/Users/JoonH/Downloads/unet_data/unet_data/val_image_easy/dirt/"
train_labels = "/Users/JoonH/Downloads/unet_data/unet_data/train_mask_easy/dirt/"
val_labels = "/Users/JoonH/Downloads/unet_data/unet_data/val_mask_easy/dirt/"

shutil.rmtree(train_images)
shutil.rmtree(train_labels)
shutil.rmtree(val_images)
shutil.rmtree(val_labels)

os.mkdir(train_images)
os.mkdir(train_labels)
os.mkdir(val_images)
os.mkdir(val_labels)

threshold = 0.0

train_count = 0 
val_count = 0

for f in tqdm(glob(train_images1 + "*.bmp")):
    
    f = f.split(".bmp")[0]
    f = f.split("\\")[-1]
    f = f.split("_")[1]
    mask_tmp = cv2.imread(train_labels1 + "image_" + f + ".bmp")
    
    #ignore too sparse of a mask, or bad mask
    if (np.count_nonzero(mask_tmp) < mask_tmp.shape[0] *
     mask_tmp.shape[1] * mask_tmp.shape[2] * threshold or f == 1959 or f == 1960):
        continue

    if (np.random.randn() < -0.5):
        continue
    
    img_adr =  train_images + "image_" + f + ".bmp"
    mask_adr = train_labels + "image_" + f + ".bmp"
    shutil.copy(train_images1 + "image_" + f + ".bmp", img_adr)
    shutil.copy(train_labels1 + "image_" + f + ".bmp", mask_adr)

    #binarize masks
    label = cv2.imread(mask_adr)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = np.where(label > 125, 255, 0)
    cv2.imwrite(mask_adr, label)

    train_count += 1

#REDUNDANCE REDUNDANCE
for f in tqdm(glob(val_images1 + "*.bmp")):
    
    f = f.split(".bmp")[0]
    f = f.split("\\")[-1]
    f = f.split("_")[1]
    mask_tmp = cv2.imread(val_labels1 + "image_" + f + ".bmp")
    if (np.count_nonzero(mask_tmp) < mask_tmp.shape[0] *
     mask_tmp.shape[1] * mask_tmp.shape[2] * threshold):
        continue
    
    img_adr =  val_images + "image_" + f + ".bmp"
    mask_adr = val_labels + "image_" + f + ".bmp"

    shutil.copy(val_images1 + "image_" + f + ".bmp", img_adr)
    shutil.copy(val_labels1 + "image_" + f + ".bmp", mask_adr)
    
    #binarize masks
    label = cv2.imread(mask_adr)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = np.where(label > 125, 255, 0)
    cv2.imwrite(mask_adr, label)
    
    val_count += 1

print("copy complete! %d training images and %d val images saved." % (train_count, val_count))

