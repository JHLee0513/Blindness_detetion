'''
@BrianLee
Offline data augmentation script. Ideally online augmentation also to be implemented for better variance.
'''
from glob import glob
import numpy as np
import shutil
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from skimage.transform import resize

train_images = "/Users/JoonH/Downloads/unet_data/unet_data/train_image_easy/dirt/"
val_images = "/Users/JoonH/Downloads/unet_data/unet_data/val_image_easy/dirt/"
train_labels = "/Users/JoonH/Downloads/unet_data/unet_data/train_mask_easy/dirt/"
val_labels = "/Users/JoonH/Downloads/unet_data/unet_data/val_mask_easy/dirt/"

filenames = []
def assign_images(names_list, images_dir, labels_dir, repeat = 3):
    count = 0
    for name in tqdm(names_list):

        image_adr = images_dir + "image_" + str(count) + "_aug.bmp"
        label_adr = labels_dir + "image_" + str(count) + "_aug.bmp"

        img_o = cv2.imread(images_dir + "image_" + name + ".bmp")
        label_o = cv2.imread(labels_dir + "label_" + name + ".bmp")

        for i in range(repeat):
            img = np.copy(img_o)
            label = np.copy(label_o)
            if (np.random.rand() < 0.5):
                img, label = augment_crop(img, label)
            if (np.random.rand() < 0.5):
                img, label = augment_flip(img, label)
            #if (np.random.rand() < 0.5):
            #    img, label = augment_brightness(img, label)
            #if (np.random.rand() < 0.5):
            #    img, label = augment_rotation(img, label)
        cv2.imwrite(image_adr, img)
        cv2.imwrite(label_adr, label)
        count += 1   

# randomly crops image and mask at same location
def augment_crop(image, mask):
    max_width = image.shape[0]
    max_height = image.shape[1]

    size = int(np.random.choice(np.arange(int(np.min([max_width, max_height]) / 4))))
    idx = int(np.random.choice(np.arange(int(np.min([max_width, max_height]) / 2))))
    print(size, idx)
    image = image[idx:idx+size, idx:idx+size, :]
    mask = mask[idx:idx+size, idx:idx+size, :]

    return np.uint8(image), np.uint8(mask) 

# either shifts the background ground color or sets it with a preset background
def augment_background(image, mask, color_not_tile = True):
    if (color_not_tile):
        #shifting color, either mono or gradient
        return True
    else:
        #use tile
        return False

#tis one bit more difficult to figure out:
# images: array of set of images to jigsaw
# masks: array of set of masks to jigsaw
#def augment_jigsaw(images, masks):

def augment_flip(image, mask):
    flip = np.random.choice([0,1])
    # horizontal flip
    if (flip == 0):
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        return image, mask
    # vertical flip
    else:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        return image, mask

def augment_brightness(image, mask, diff = 15):
    brightness = np.random.choice(np.arange(-diff,diff))
    return image + brightness, mask + brightness

def augment_rotation(image, mask):
    degree = np.random.choice(np.arange(0, 360, 15))
    rows, cols, _ = image.shape
    rot_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    image = cv2.warpAffine(image, rot_matrix, (cols, rows))
    mask = cv2.warpAffine(mask, rot_matrix, (cols, rows))
    return image, mask



def get_names(address):
    for f in glob(address + "/*.bmp"):
        f = f.split(".bmp")[0]
        f = f.split("\\")[-1]
        f = f.split("_")[1]
    
        if (f == 1959 or f == 1960):
            continue

        filenames.append(f)

    return filenames


names = get_names(train_images)
assign_images(names, train_images, train_labels)
names = get_names(val_images)
assign_images(names, val_images, val_labels)
print("split complete!")

#shutil.rmtree(train_images)
#shutil.rmtree(train_labels)
#shutil.rmtree(val_images)
#shutil.rmtree(val_labels)

#os.mkdir(train_images)
#os.mkdir(train_labels)
#os.mkdir(val_images)
#os.mkdir(val_labels)

#train_names, val_names = train_test_split(filenames, test_size = 0.3, random_state = 50)



