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
from efficientnet import EfficientNetB5, EfficientNetB4
from sklearn.ensemble import AdaBoostClassifier
import scipy
from imgaug import augmenters as iaa
import imgaug as ia
import gc
gc.enable()
gc.collect()


train_df = pd.read_csv("/nas-homes/joonl4/blind/train_balanced.csv")
train_df = train_df.astype(str)

train, val = train_test_split(train_df, test_size = 0.2, random_state = 69420, stratify = train_df['diagnosis'])
del train
gc.collect()
val, stack_val = train_test_split(val, test_size = 0.5, random_state = 69420, stratify = val['diagnosis'])

ada = AdaBoostClassifier(n_estimators = 500, learning_rate = 0.01)

of = lambda aug: iaa.OneOf(aug)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.35),
])

img_target = 456#256
SIZE = 456
IMG_SIZE = 456
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
#     image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


class My_Generator(Sequence):

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
#         batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        if(self.is_train):
            return self.train_generate(batch_x, batch_y)
        return self.test_generate(batch_x)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass

    def test_generate(self, batch_x):
        batch_images = []
        for (sample) in batch_x:
            img = cv2.imread('../input/aptos2019-blindness-detection/test_images/'+sample)
            img = load_ben_color(img)
            img = seq.augment_image(img)
            batch_images.append(img)
#             batch_images.append(cv2.flip(img, 0)) # horizontal flip TTA
        batch_images = np.array(batch_images, np.float32) / 255
        return batch_images

test_generator = My_Generator(val['id_code'], 1, is_train=False)
tta_steps = 5
preds_tta=[]
for model_dir in [
    "raw_effnet_pretrained_regression_fold_v20_snap1.h5"]:

    K.clear_session()
    model = load_model("../input/blind-brian-weights/" + model_dir)
    preds = []
    for i in tqdm(range(tta_steps)):
        preds.append(model.predict_generator(generator=test_generator,steps =np.ceil(test_df.shape[0])))
    preds_tta.append(np.mean(preds, axis = 0))
    del model
    gc.collect()

img_target = 380
SIZE = 380
IMG_SIZE = 380
test_generator = My_Generator(test_df['id_code'], 1, is_train=False)
for model_dir in [
    "raw_effnet_pretrained_regression_fold_v11_snap1.h5",
    "raw_effnet_pretrained_regression_fold_v11_snap2.h5"]:

    K.clear_session()
    model = load_model("../input/blind-brian-weights/" + model_dir)
    preds = []
    for i in tqdm(range(tta_steps)):
#         test_generator.reset()
        preds.append(model.predict_generator(generator=test_generator,steps =np.ceil(test_df.shape[0])))
    preds_tta.append(np.mean(preds, axis = 0))
    del model
    gc.collect()



test_df = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
test_df['diagnosis'] = predicted_class_indices
test_df.to_csv("submission.csv", index=False)