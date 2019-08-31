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
from efficientnet import EfficientNetB3, EfficientNetB1
from keras.utils import multi_gpu_model
import scipy
from imgaug import augmenters as iaa
import imgaug as ia
import gc
gc.enable()
gc.collect()

img_target = 300
SIZE = 300
IMG_SIZE = 300
batch = 40

train_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
train_df['id_code'] += '.png'
val_2019_list = pd.read_csv("/nas-homes/joonl4/blind/adv_val.csv")
val_2019_list = val_2019_list[val_2019_list['name'].str.contains(".png")]
print(val_2019_list.head())

val = train_df[train_df['id_code'].isin(val_2019_list['name'])]
train = train_df[~train_df['id_code'].isin(val_2019_list['name'])]

print(val.head())
print(train.head())
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
            # img = cv2.resize(img, (SIZE, SIZE))
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
            # img = cv2.resize(img, (SIZE, SIZE))
            # img = val_seq.augment_image(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        return batch_images, batch_y


class QWKEvaluation(Callback):
    def __init__(self, validation_data=(), batch_size=64, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(generator=self.valid_generator,
                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                                                  workers=1, use_multiprocessing=True,
                                                  verbose=1)
            def flatten(pred):
                #print(np.argmax(y,axis = 1).astype(int))
                #return np.argmax(y, axis=1).astype(int)
                return np.rint(pred).astype(int)
                #return np.rint(np.sum(y,axis=1)).astype(int)
            
            score = cohen_kappa_score(flatten(self.y_val),
                                      flatten(y_pred),
                                      labels=[0,1,2,3,4],
                                      weights='quadratic')
#             print(flatten(self.y_val)[:5])
#             print(flatten(y_pred)[:5])
            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
            self.history.append(score)
            if score >= max(self.history):
                print('save checkpoint: ', score)
                # self.model.save(qwk_ckpt_name)
                #log.write(str(log_fold) + ": " + str(score) + "\n")

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-40, 40), # rotate by -180 to +180 degrees
            # shear=(-5, 5), # shear by -16 to +16 degrees
            # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        sometimes(iaa.contrast.LinearContrast(alpha = (0.85, 1.15))),
        sometimes(iaa.size.Crop(percent = (0.05, 0.4), keep_size = True))
    ],
    random_order=True)

def build_model(freeze = False):
    model = EfficientNetB3(input_shape = (img_target, img_target, 3), weights = 'imagenet', include_top = False, pooling = None)
    for layers in model.layers:
        layers.trainable= not freeze
    inputs = model.input
    x = model.output
    x = GlobalAveragePooling2D()(x)
    out_layer = Dense(1, activation = None, name = 'normal_regressor') (Dropout(0.3)(x))
    model = Model(inputs, out_layer)
    return model

# for cv_index in range(1,6):
for cv_index in range(1):
    fold = cv_index
    train_x = train['id_code']
    train_y = train['diagnosis'].astype(int)
    val_x = val['id_code']
    val_y = val['diagnosis'].astype(int)
    train_generator = My_Generator(train_x, train_y, batch, is_train=True, augment=True)
    val_generator = My_Generator(val_x, val_y, batch, is_train=False)
    qwk = QWKEvaluation(validation_data=(val_generator, val_y),
                        batch_size=batch, interval=1)
    with tf.device('/cpu:0'):
        model = build_model(freeze = False)
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.load_weights('/nas-homes/joonl4/blind_weights/effnet_adversarial_B3.hdf5')
    parallel_model.compile(loss='mse', optimizer = Adam(lr=1e-3),
                metrics= ['accuracy'])
    save_model_name = '/nas-homes/joonl4/blind_weights/effnet_adversarial_B3_tuned.hdf5'
    model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                    mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)

    cycle = len(train_y)/batch * 10
    cyclic = CyclicLR(mode='exp_range', base_lr = .5e-4, max_lr = 1e-3, step_size = cycle)  
    # parallel_model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=len(train_y)/batch,
    #     epochs=10,
    #     verbose = 1,
    #     callbacks = [model_checkpoint, qwk, cyclic],
    #     validation_data = val_generator,
    #     validation_steps = len(val_y)/batch,
    #     workers=1, use_multiprocessing=False)
    parallel_model.load_weights(save_model_name)
    # single_model = parallel_model.layers[-2]
    # single_model.save("/nas-homes/joonl4/blind_weights/effnet_adversarial_B3.h5")

    train_generator = My_Generator(train_x, train_y, batch, is_train=True, augment=False)
    parallel_model.load_weights(save_model_name)
    
    parallel_model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_y)/batch,
        epochs=10,
        verbose = 1,
        callbacks = [model_checkpoint, qwk, cyclic],
        validation_data = val_generator,
        validation_steps = len(val_y)/batch,
        workers=1, use_multiprocessing=False)
    parallel_model.load_weights(save_model_name)
    single_model = parallel_model.layers[-2]
    single_model.save("/nas-homes/joonl4/blind_weights/effnet_adversarial_B3_fine.h5")
