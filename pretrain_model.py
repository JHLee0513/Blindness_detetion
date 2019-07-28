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
batch = 16
train_df = pd.read_csv("/nas-homes/joonl4/blind_2015/trainLabels.csv")
train_df2 = pd.read_csv("/nas-homes/joonl4/blind_2015/retinopathy_solution.csv")
train_df = pd.concat([train_df, train_df2], axis = 0, sort = False)
train_df.reset_index(drop = True)
val_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
train_df['image'] = train_df['image'].astype(str) + ".jpeg"
train_df = train_df.astype(str)
val_df['id_code'] = val_df['id_code'].astype(str) + ".png"
val_df = val_df.astype(str)
train_d = train_df

data_gen_args = dict(#featurewise_center=True,
                     #featurewise_std_normalization=True,
                     rotation_range=360, # 90
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
		             horizontal_flip = True,
                     vertical_flip = True,
		             rescale = 1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(rescale = 1./255)
#image_datagen.fit(images, augment=True, seed=seed)

train_generator = image_datagen.flow_from_dataframe(
    train_d,
    directory="/nas-homes/joonl4/blind_2015/train/",
    x_col = 'image',
    y_col = 'level',
    target_size = (img_target,img_target),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = batch)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory="/nas-homes/joonl4/blind/train_images/",
    x_col = 'id_code',
    y_col = 'diagnosis',
    target_size = (img_target,img_target),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = batch)


#from keras.applications.densenet import DenseNet121, DenseNet169
from efficientnet import EfficientNetB4
#model = DenseNet121(include_top = False, weights = 'imagenet', 
#                    input_shape = (img_target,img_target,3), pooling = 'max')

#model = DenseNet169(include_top = False, weights = 'imagenet', 
#                   input_shape = (img_target,img_target,3), pooling = 'avg')
model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights=None, include_top = False, pooling = 'avg')

for layers in model.layers:
    layers.trainable=True

inputs = model.input
x = model.output
#x = Dense(1024, activation = 'relu', use_bias = True) (x)
#x = Dense(1024, activation = 'relu', use_bias = True) (x)
#x = Dense(1024, activation = 'relu', use_bias = True) (x)
x = Dropout(rate = 0.4) (x)
x = Dense(512, activation = 'elu') (x)
x = Dropout(rate = 0.4) (x)
x = Dense(5, activation = 'softmax') (x)

model = Model(inputs, x)

model.compile(loss='categorical_crossentropy', optimizer ='adam',
             metrics= ['categorical_accuracy'])


#model.summary()

save_model_name = 'raw_pretrain_effnet_B4_fulldata.hdf5'
model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                   mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)

cycle = 35126/batch * 20
cyclic = CyclicLR(mode='exp_range', base_lr = 0.0001, max_lr = 0.01, step_size = cycle)

#model.load_weights("raw_pretrain_effnet_B4.hdf5")

model.fit_generator(
    train_generator,
    steps_per_epoch=88702/batch,
    epochs=10,
    verbose = 1,
    callbacks = [model_checkpoint],
    validation_data = val_generator,
    validation_steps = 3662/batch)

''''
K.clear_session()


#model.load_weights(save_model_name)
model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights='imagenet', include_top = False, pooling = 'avg')

for layers in model.layers:
    layers.trainable=False

inputs = model.input
x = model.output
#x = Dense(1024, activation = 'relu', use_bias = True) (x)
#x = Dense(1024, activation = 'relu', use_bias = True) (x)
#x = Dense(1024, activation = 'relu', use_bias = True) (x)
x = Dropout(rate = 0.4) (x)
x = Dense(512, activation = 'elu') (x)
x = Dropout(rate = 0.25) (x)
x = Dense(5, activation = 'softmax') (x)

model = Model(inputs, x)

for layers in model.layers:
    layers.trainable=True

model.load_weights(save_model_name)

model.compile(loss='categorical_crossentropy',optimizer= 'adam',
             metrics= ['categorical_accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=35126/batch,
    epochs=10,
    verbose = 1,
    callbacks = [model_checkpoint],
    validation_data = val_generator,
    validation_steps = 3662/batch)'''

# model.save("blind_effnetB4.h5")
'''



test_datagen= ImageDataGenerator(rescale=1./255,validation_split=0.2,horizontal_flip=True)

test_generator=test_datagen.flow_from_dataframe(dataframe=test_df,
                                                directory = "/nas-homes/joonl4/blind/test_images/",
                                                x_col="id_code",
                                                target_size=(img_target, img_target),
                                                batch_size=1,
                                                shuffle=False, 
                                                class_mode=None,
                                                color_mode = 'rgb')
                                                #seed=)
#https://www.kaggle.com/bharatsingh213/keras-resnet-test-time-augmentation
tta_steps = 10
preds_tta=[]
for i in tqdm(range(tta_steps)):
    test_generator.reset()
    preds = model.predict_generator(generator=test_generator,steps =np.ceil(test_df.shape[0]))
#     print('Before ', preds.shape)
    preds_tta.append(preds)
#     print(i,  len(preds_tta))

final_preds = np.mean(preds_tta, axis=0)

predicted_class_indices = np.argmax(final_preds, axis=1)
results=pd.DataFrame({"id_code":test_generator.filenames, "diagnosis":predicted_class_indices})  
results.id_code=results.id_code.apply(lambda x: x[:-4])# results.head()
results.to_csv("/home/joonl4/baseline_submission.csv", index=False)
'''

