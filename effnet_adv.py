import numpy as np
import pandas as pd
import gc
import cv2
from utils.clr_callback import *
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras import backend as K
from keras.utils import Sequence, to_categorical
from keras.callbacks import Callback
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy, categorical_crossentropy
from efficientnet import EfficientNetB0
from keras.utils import multi_gpu_model
import scipy
from imgaug import augmenters as iaa
import imgaug as ia
import gc
import math
gc.enable()
gc.collect()

img_target = 228
SIZE = 228
IMG_SIZE = 228
batch = 72
IMAGE_SIZE = 228
train_df = pd.read_csv("/nas-homes/joonl4/blind/train_balanced.csv")
train_df['diagnosis'] = 0
test_df = pd.read_csv("/nas-homes/joonl4/blind/test.csv")
test_df['diagnosis'] = 1
test_df['id_code'] += '.png'

comb_df = pd.concat([train_df, test_df], axis = 0)

train, val = train_test_split(comb_df, test_size = 0.2, random_state = 69420, stratify = comb_df['diagnosis'])

#https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping#3.-Further-improve-by-auto-cropping

def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def load_ben_color(image, sigmaX=10):
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

class My_Generator(Sequence):

    def __init__(self, image_filenames, labels,
                 batch_size, is_train=True,
                 mix=False, augment=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if(self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        if(self.is_train):
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass
    
    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread('/nas-homes/joonl4/blind/train_images/'+sample)
            img = load_ben_color(img)
            if(self.is_augment):
                img = seq.augment_image(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        if(self.is_mix):
            batch_images, batch_y = self.mix_up(batch_images, batch_y)
        return batch_images, batch_y

    def valid_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread('/nas-homes/joonl4/blind/train_images/'+sample)
            img = load_ben_color(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        return batch_images, batch_y

class Test_Generator(Sequence):

    def __init__(self, image_filenames,
                 batch_size, is_train=True,
                 mix=False, augment=False):
        self.image_filenames= image_filenames
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if(self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.valid_generate(batch_x)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass
    
    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread('/nas-homes/joonl4/blind/train_images/'+sample)
            img = load_ben_color(img)
            if(self.is_augment):
                img = seq.augment_image(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        if(self.is_mix):
            batch_images, batch_y = self.mix_up(batch_images, batch_y)
        return batch_images, batch_y

    def valid_generate(self, batch_x):
        batch_images = []
        for sample inbatch_x:
            img = cv2.imread('/nas-homes/joonl4/blind/train_images/'+sample)
            img = load_ben_color(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        return batch_images


def build_model(freeze = False):
    model = EfficientNetB0(input_shape = (img_target, img_target, 3), weights = 'imagenet', include_top = False, pooling = None)
    inputs = model.input
    x = model.output
    x = GlobalAveragePooling2D()(x)
    out_layer = Dense(1, activation = 'sigmoid', name = 'normal_regressor') (Dropout(0.2)(x))
    model = Model(inputs, out_layer)
    return model

for cv_index in range(1):
    fold = cv_index
    train_x = train['id_code']
    train_y = train['diagnosis']
    val_x = val['id_code']
    val_y = val['diagnosis']
    with tf.device('/cpu:0'):
        model = build_model(freeze = False)
    parallel_model = multi_gpu_model(model, gpus=3)
    save_model_name = '/nas-homes/joonl4/blind_weights/eff_adversarial.hdf5'
    model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                    mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)
    train_generator = My_Generator(train_x, train_y, batch, is_train=True, augment=False)
    val_generator = My_Generator(val_x, val_y, batch, is_train=False)
    parallel_model.compile(loss='binary_crossentropy', optimizer = Adamax(1e-3),
                metrics= ['accuracy'])
    cycle = len(train_y)/batch * 10
    cyclic = CyclicLR(mode='exp_range', base_lr = .5e-4, max_lr = 1e-3, step_size = cycle)  
    # parallel_model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=len(train_y)/batch,
    #     epochs=20,
    #     verbose = 1,
    #     callbacks = [model_checkpoint, cyclic],
    #     validation_data = val_generator,
    #     validation_steps = len(val_y)/batch,
    #     workers=1, use_multiprocessing=False)

    parallel_model.load_weights(save_model_name)
    test_generator = Test_Generator(train_df['id_code'], 1, is_train=False)
    predictions = model.predict_generator(generator=test_generator,steps =np.ceil(train.shape[0]), verbose = 1)
    train_df['is_test'] = predictions
    train_df = train_df.sort_values(by=['is_test'])
    train_df.to_csv("/nas-homes/joonl4/blind/adv_list.csv", index=False)

