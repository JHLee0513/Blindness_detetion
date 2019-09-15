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
from efficientnet import EfficientNetB4, EfficientNetB3
import scipy
from imgaug import augmenters as iaa
import imgaug as ia
from utils.common_utils import *

"""
@author (Brian) JoonHo Lee

Pretraining first Model (Efficient B4 + MSE) to provide better initialization
for faster convergence and generalization.
first pretrained using 2019 with classes balanced from 2015 data. Check
approach1_pretrain.py for the pretraining. With pretrained weights, model is
finetuned only on 2019 data with NO augmentation. In addition, Snapshotting was
used to generated 5 ensembles.
"""

img_target = 380
SIZE = 380
IMG_SIZE = 380
batch = 8
path = '/nas-homes/joonl4'
train_df = pd.read_csv(f'{path}/blind/train_balanced.csv')
train, val = train_test_split(
    train_df,
    test_size = 0.2,
    random_state = None,
    stratify = train_df['diagnosis'])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    sometimes(iaa.size.Crop(percent = (0.05, 0.1), keep_size = True)),
    sometimes(iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-10, 10),
        cval=(0, 255),
        mode=ia.ALL))],random_order=True)

train_x = train['id_code']
train_y = train['diagnosis'].astype(int)
val_x = val['id_code']
val_y = val['diagnosis'].astype(int)
train_generator = My_Generator(
    train_x,
    train_y,
    batch,
    is_train=True,
    augment=False)
val_generator = My_Generator(val_x, val_y, batch, is_train=False)
qwk = QWKEvaluation(validation_data=(val_generator, val_y),
                    batch_size=batch, interval=1)
model = EffNetB4(freeze = False)
model.compile(loss='mse', optimizer = Adam(lr = 1e-4),
            metrics= ['accuracy'])
save_model_name = \
    f'{path}/blind_weights/raw_effnet_pretrained_regression_fold_v110.hdf5'
model_checkpoint = ModelCheckpoint(
    save_model_name,
    monitor= 'val_loss',
    mode = 'min',
    save_best_only=True,
    verbose=1,
    save_weights_only = True)
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_y)/batch,
    epochs=3,
    verbose = 1,
    callbacks = [model_checkpoint, qwk],
    validation_data = val_generator,
    validation_steps = len(val_y)/batch,
    workers=1, use_multiprocessing=False)
model.load_weights(save_model_name)
train_generator = My_Generator(
    train_x,
    train_y,
    batch,
    is_train=True,
    augment=True)
val_generator = My_Generator(val_x, val_y, batch, is_train=False)
qwk = QWKEvaluation(validation_data=(val_generator, val_y),
                    batch_size=batch, interval=1)
model = EffNetB4(freeze = False)
model.load_weights(save_model_name)
model.compile(loss='mse', optimizer = Adam(lr=1e-3),
            metrics= ['accuracy'])
cycle = len(train_y)/batch * 15
cyclic = CyclicLR(
    mode='exp_range', base_lr = 1e-4, max_lr = 1e-3, step_size = cycle)  
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_y)/batch,
    epochs=75,
    verbose = 1,
    callbacks = [model_checkpoint, qwk, cyclic],
    validation_data = val_generator,
    validation_steps = len(val_y)/batch,
    workers=1, use_multiprocessing=False)
model.load_weights(save_model_name)
model.save(
    f'{path}/blind_weights/raw_effnet_pretrained_regression_fold_v110.h5')
