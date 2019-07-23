'''
@BrianLee
Randomly splits entire data into train set, valdiation set. Test split is not implemented yet.
'''
from glob import glob
import numpy as np
import shutil
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from skimage.transform import resize

images = "/Users/JoonH/Downloads/unet_data/unet_data/images/"
labels = "/Users/JoonH/Downloads/unet_data/unet_data/labels/"

train_images = "/Users/JoonH/Downloads/unet_data/unet_data/train_image/dirt/"
val_images = "/Users/JoonH/Downloads/unet_data/unet_data/val_image/dirt/"
test_images = "/Users/JoonH/Downloads/unet_data/unet_data/test_image/dirt/"
train_labels = "/Users/JoonH/Downloads/unet_data/unet_data/train_mask/dirt/"
val_labels = "/Users/JoonH/Downloads/unet_data/unet_data/val_mask/dirt/"
test_labels = "/Users/JoonH/Downloads/unet_data/unet_data/test_mask/dirt/"

filenames = []
def assign_images(names_list, images_dir, labels_dir, crop = False, mat = False, gray = True):
    count = 0
    for name in tqdm(names_list):
        image_adr = images_dir + "image_" + str(count) + ".bmp"
        label_adr = labels_dir + "image_" + str(count) + ".bmp"
        shutil.copy(images + "image_" + name + ".bmp", image_adr)
        shutil.copy(labels + "label_" + name + ".bmp", label_adr)
        
        #crop the images based on mask
        img = cv2.imread(image_adr)
        label = cv2.imread(label_adr)
        #https://stackoverflow.com/questions/25710356/numpy-given-the-nonzero-indices-of-a-matrix-how-to-extract-the-elements-into-a
        x,y = np.nonzero(label[:,:,0])
        pad = 0
        
        if (crop):
            img_crop = np.zeros((x.max()+1-x.min() + pad * 2, y.max()+1-y.min() + pad * 2,3))
            label_crop = np.zeros((x.max()+1-x.min() + pad * 2, y.max()+1-y.min() + pad * 2,3))
        
            for channel in range(3):
                img_crop[:,:,channel] = img[x.min():x.max()+1, y.min():y.max()+1,channel]
                label_crop[:,:,channel] = label[x.min():x.max()+1, y.min():y.max()+1,channel]

            black = np.all(img_crop == 0, axis = -1)
            img_crop[black] = 182 #avg color of the table (background)

            #binarizing label mask
            non_dirt = np.all(label_crop < 125, axis = -1)
            label_crop[non_dirt] = 0
            
            if (mat):
                if gray:
                    #matting with grayscale only background
                    ma = np.random.choice(np.arange(0,255))
                    mat = [ma, ma, ma]
                else:
                    #matting with random RGB background
                    mat = np.random.choice(np.arange(0,255), size = 3)

                img_crop[non_dirt] = mat
        else:
            img_crop = img
            label_crop = label

        cv2.imwrite(image_adr, img_crop)
        cv2.imwrite(label_adr, label_crop)
        count += 1
   
        '''if (oversample): # oversampling
            for i in range(3):
            #matting with grayscale only background
                ma = np.random.choice(np.arange(0,255))
                mat = [ma, ma, ma]
                image_adr = images_dir + "image_" + str(count) + ".bmp"
                label_adr = labels_dir + "image_" + str(count) + ".bmp"
                img_crop[non_dirt] = mat
                cv2.imwrite(image_adr, img_crop)
                cv2.imwrite(label_adr, label_crop)
                count += 1'''

for f in glob(images + "/*.bmp"):
    f = f.split(".bmp")[0]
    f = f.split("\\")[-1]
    f = f.split("_")[1]
    
    if (f == 1959 or f == 1960):
        continue

    filenames.append(f)

shutil.rmtree(train_images)
shutil.rmtree(train_labels)
shutil.rmtree(val_images)
shutil.rmtree(val_labels)
shutil.rmtree(test_images)
shutil.rmtree(test_labels)

os.mkdir(train_images)
os.mkdir(train_labels)
os.mkdir(val_images)
os.mkdir(val_labels)
os.mkdir(test_images)
os.mkdir(test_labels)

train_names, val_names = train_test_split(filenames, test_size = 0.4)#, random_state = 50)

val_names, test_names = train_test_split(val_names, test_size = 0.5)

assign_images(train_names, train_images, train_labels)
assign_images(val_names, val_images, val_labels)
assign_images(test_names, test_images, test_labels)

print("split complete!")

