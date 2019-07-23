'''
custom data generator as keras does quite bad job on augmentation for image/mask generator for segmentation task.
citation: https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a
'''

from cv2 import imread
import numpy as np
import cv2

def get_image(path):
    #return cv2.cvtColor(imread(path), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)

def get_mask(path):
    return cv2.cvtColor(imread(path), cv2.COLOR_BGR2GRAY)


def flip(image, label):

    if (np.random.randn() < 0.5):
        image = np.fliplr(image)
        label = np.fliplr(label)
    else:
        image = np.flipud(image)
        label = np.flipud(label)

    return image, label

def rotate(image, label):
    return image, label

def crop(image, label):
    return image, label
    
def bright(image, label):
    val = np.random.choice(np.arange(-3,3))
    image += np.uint8(val)
    label += np.uint8(val)
    image = np.clip(image, 0, 255)
    label = np.clip(label, 0, 255)
    return image, label

def preprocess_input(image, label):
    
    if (np.random.randn() < 0.2):
        image, label = flip(image, label)
    #if (np.random.randn() < 0.25):
    #    image, label = crop(image, label)
    #if (np.random.randn() < 0.25):
    #    image, label = rotate(image, label)
    if (np.random.randn() < 0.2):
        image, label = bright(image, label)
    
    
    return image, label

#figure out argument and logic for grayscale vs rgb
def image_generator(filenames, image_path,label_path, batch_size, shape):
    
    while True:
        # Select files (paths/indices) for the batch 
        batch_paths = np.random.choice(a = filenames, 
                                         size = batch_size)
        #batch_input = np.empty((batch_size, shape, shape, 1))
        batch_input = np.empty((batch_size, shape, shape, 3))
        batch_output = np.empty((batch_size, shape, shape, 1)) 
          
        # Read in each input, perform preprocessing and get labels
        for i, input_path in enumerate(batch_paths):
            img_path = image_path + "image_" + input_path + ".bmp"
            input1 = get_image(img_path)
            path = label_path + "image_" + input_path + ".bmp"
            output1 = get_mask(path)
            input1, output1 = preprocess_input(input1, output1)
            input2 = cv2.resize(input1, (shape, shape)) / 255.
            output2 = cv2.resize(output1, (shape, shape)) / 255.

            #if grayscale, if rgb no expanding dim
            #batch_input[i,] = np.expand_dims(input2, -1)
            batch_input[i,] = input2
            batch_output[i,] = np.expand_dims(output2, -1)

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield(batch_x, batch_y)

