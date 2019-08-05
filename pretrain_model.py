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
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from efficientnet import EfficientNetB4
import scipy
from imgaug import augmenters as iaa
import imgaug as ia
gc.enable()
gc.collect()

img_target = 256
SIZE = 256
batch = 8
train_df = pd.read_csv("/nas-homes/joonl4/blind_2015/trainLabels.csv")
#print(train_df.head())
train_df2 = pd.read_csv("/nas-homes/joonl4/blind_2015/retinopathy_solution.csv")
#print(train_df2.head())
#train_df2 = train_df2.rename(columns={"level": "label"})
train_df2 = train_df2.drop(["Usage"], axis = 1)
#train_df = train_df.astype(str)
#train_df2 = train_df2.astype(str)
train_df = pd.concat([train_df, train_df2], axis = 0, sort = False)
train_df.reset_index(drop = True)
val_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
train_df['image'] = train_df['image'].astype(str) + ".jpeg"
train_df = train_df.astype(str)
val_df['id_code'] = val_df['id_code'].astype(str) + ".png"
val_df = val_df.astype(str)
train_x = train_df['image']
#train_y = to_categorical(train_df['level'], num_classes=5)
#train_y= train_df['level'].astype(int)
#val_x = val_df['id_code']
#val_y = val_df['diagnosis'].astype(int)
#val_y = to_categorical(val_df['diagnosis'], num_classes=5)

data_gen_args = dict(#featurewise_center=True,
                     #featurewise_std_normalization=True,
                     rotation_range=180,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     vertical_flip = True,
		             horizontal_flip = True,
		             zoom_range=0.2,
                     rescale = 1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(rescale = 1./255)
#image_datagen.fit(images, augment=True, seed=seed)

train_generator = image_datagen.flow_from_dataframe(
    train_df,
    directory="/nas-homes/joonl4/blind_2015/train/",
    x_col = 'image',
    y_col = 'level',
    target_size = (img_target,img_target),
    color_mode = 'rgb',
    class_mode = 'other',
    batch_size = batch)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory="/nas-homes/joonl4/blind/train_images/",
    x_col = 'id_code',
    y_col = 'diagnosis',
    target_size = (img_target,img_target),
    color_mode = 'rgb',
    class_mode = 'other',
    batch_size = batch)

model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights='imagenet', include_top = False, pooling = 'avg')

for layers in model.layers:
    layers.trainable=True

inputs = model.input
x = model.output
x = Dropout(rate = 0.5) (x)
x = Dense(512, activation = 'elu', name = "fc") (x)
x = Dropout(rate = 0.25) (x)
x = Dense(1, activation = None, name = 'regressor') (x)

model = Model(inputs, x)
save_model_name = '/nas-homes/joonl4/blind_weights/raw_pretrain_effnet_fulldata_regresion.hdf5'
model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                   mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)
# train_generator = My_Generator(train_x, train_y, 8, is_train=False)
# train_mixup = My_Generator(train_x, train_y, 8, is_train=False, mix=True, augment=True)
# val_generator = My_Generator(val_x, val_y, 8, is_train=False)
# model.load_weights("/nas-homes/joonl4/blind_weights/raw_pretrain_effnet_B4.hdf5")
# warmup
model.compile(loss='mse', optimizer = SGD(lr=1e-3, momentum = 0.95, nesterov = True),
             metrics= ['accuracy', 'mae'])
model.fit_generator(
    train_generator,
    steps_per_epoch=88702/batch,
    epochs=3,
    verbose = 1,
    callbacks = [model_checkpoint],
    validation_data = val_generator,
    validation_steps = 3662/batch)

cycle = 88702/batch * 10
cyclic = CyclicLR(mode='exp_range', base_lr = 0.00001, max_lr = 0.001, step_size = cycle)
model.load_weights(save_model_name)
model.compile(loss='mse', optimizer = SGD(lr=0.001, momentum = 0.95, nesterov = True),
             metrics= ['accuracy', 'mae'])
model.fit_generator(
    train_mixup,
    steps_per_epoch=88702/batch,
    epochs=10,
    verbose = 1,
    callbacks = [model_checkpoint],
    validation_data = val_generator,
    validation_steps = 3662/batch)
