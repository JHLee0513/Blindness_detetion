import numpy as np
import pandas as pd
import gc
from utils.clr_callback import *
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras import backend as K
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import gc
gc.enable()
gc.collect()

img_target = 256
batch = 8
train_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
test_df = pd.read_csv("/nas-homes/joonl4/blind/test.csv")
train_df['id_code'] += ".png"
test_df['id_code'] += ".png"
train_df = train_df.astype(str)
from sklearn.model_selection import train_test_split, StratifiedKFold
from efficientnet import EfficientNetB4

data_gen_args = dict(#featurewise_center=True,
                     #featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     vertical_flip = True,
                     horizontal_flip = True,
                     zoom_range=0.2,
                     rescale = 1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(rescale = 1./255)
x = train_df['id_code']
y = train_df['diagnosis']

kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=2019)
train_all = []
evaluate_all = []
for train_idx, test_idx in kf.split(x, y):
    train_all.append(train_idx)
    evaluate_all.append(test_idx)

def get_cv_data(cv_index):
    train_index = train_all[cv_index-1]
    evaluate_index = evaluate_all[cv_index-1]
    train = train_df.iloc[train_index]
    val = train_df.iloc[evaluate_index]
    train.reset_index(drop = True)
    val.reset_index(drop = True)
    return train, val

for fold in range(5):
    train, val = get_cv_data(fold + 1)
    train_generator = image_datagen.flow_from_dataframe(
        train,
        directory="/nas-homes/joonl4/blind/train_images/",
        x_col = 'id_code',
        y_col = 'diagnosis',
        target_size = (img_target,img_target),
        color_mode = 'rgb',
        class_mode = 'categorical',
        batch_size = batch)

    val_generator = val_datagen.flow_from_dataframe(
        val,
        directory="/nas-homes/joonl4/blind/train_images/",
        x_col = 'id_code',
        y_col = 'diagnosis',
        target_size = (img_target,img_target),
        color_mode = 'rgb',
        class_mode = 'categorical',
        batch_size = batch)

    model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights = 'imagenet', include_top = False, pooling = 'avg')

    for layers in model.layers:
        layers.trainable=True

    inputs = model.input
    x = model.output
    x = Dropout(rate = 0.5) (x)
    x = Dense(512, activation = 'relu') (x)
    x = Dropout(rate = 0.5) (x)
    x = Dense(5, activation = 'softmax') (x)

    model = Model(inputs, x)

    model.compile(loss='categorical_crossentropy', optimizer = Adam(1e-3),
                metrics= ['categorical_accuracy'])

    model.load_weights("/nas-homes/joonl4/blind_weights/raw_pretrained_effnet_weights.hdf5")
    save_model_name = 'blind_B4_baseline_fold' + str(fold+1) + '.hdf5'
    model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                    mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=2560/batch,
        epochs=5,
        verbose = 1,
        callbacks = [model_checkpoint],
        validation_data = val_generator,
        validation_steps = 1100/batch)

    model.load_weights(save_model_name)
    model.compile(loss='categorical_crossentropy', optimizer = SGD(lr = 0.01, momentum = 0.9, nesterov = True),
                metrics= ['categorical_accuracy'])
    cycle = 2560/batch * 30
    cyclic = CyclicLR(mode='triangular2', base_lr = 0.00005, max_lr = 0.01, step_size = cycle)
    model.fit_generator(
        train_generator,
        steps_per_epoch=2560/batch,
        epochs=90,
        verbose = 1,
        callbacks = [model_checkpoint, cyclic],
        validation_data = val_generator,
        validation_steps = 1100/batch)
    model.load_weights(save_model_name)
    model.save('/nas-homes/joonl4/blind_weights/blind_B4_baseline_fold' + str(fold+1) + '.h5')
