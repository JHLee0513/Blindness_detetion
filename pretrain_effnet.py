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

img_target = 380#256
batch = 4
train_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
test_df = pd.read_csv("/nas-homes/joonl4/blind/test.csv")

train_df['id_code'] += ".png"

test_df['id_code'] += ".png"

train_df = train_df.astype(str)
test_df = test_df.astype(str)
from sklearn.model_selection import train_test_split

train, val = train_test_split(train_df, test_size = 0.25, stratify = train_df['diagnosis'])
train = train.reset_index(drop = True)
val = val.reset_index(drop = True)


data_gen_args = dict(#featurewise_center=True,
                     #featurewise_std_normalization=True,
                     rotation_range=360,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     vertical_flip = True,
		     horizontal_flip = True,
		     zoom_range=0.2,
                     rescale = 1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(rescale = 1./255)
#image_datagen.fit(images, augment=True, seed=seed)

train_generator = image_datagen.flow_from_dataframe(
    train,
    #directory="/nas-homes/joonl4/blind/train_images/",
    directory="/nas-homes/joonl4/blind/train_images_preprecessed/",
    x_col = 'id_code',
    y_col = 'diagnosis',
    target_size = (img_target,img_target),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = batch)

val_generator = val_datagen.flow_from_dataframe(
    val,
    #directory="/nas-homes/joonl4/blind/train_images/",
    directory="/nas-homes/joonl4/blind/train_images_preprocessed/",
    x_col = 'id_code',
    y_col = 'diagnosis',
    target_size = (img_target,img_target),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = batch)


from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
#model = ResNet50(include_top = False, weights = 'imagenet', 
#                    input_shape = (img_target,img_target,3), pooling = 'avg') #pooling = 'avg'

#model = Xception(include_top = False, weights = 'imagenet', input_shape = (img_target,img_target,3), pooling = 'max')

from efficientnet import EfficientNetB4

model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights = 'imagenet', include_top = False, pooling = 'avg')


for layers in model.layers:
    layers.trainable=True

inputs = model.input
x = model.output
#x = Dense(1000, activation = 'relu') (x)
#x = Dropout(rate = 0.25) (x)
#x = Dense(1000, activation = 'relu') (x)
#x = Dropout(rate = 0.25) (x)
#x = Dense(1000, activation = 'relu') (x)
x = Dropout(rate = 0.4) (x)
x = Dense(5, activation = 'softmax') (x)

model = Model(inputs, x)

model.compile(loss='categorical_crossentropy', optimizer = 'SGD',
             metrics= ['categorical_accuracy'])


model.summary()
model.load_weights("./raw_pretrain_effnet_B4_380.hdf5")
save_model_name = 'blind_effnetB4_transfer_bens_aug_submit.hdf5'
model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_categorical_accuracy',
                                   mode = 'max', save_best_only=True, verbose=1,save_weights_only = True)

cycle = 2560/batch * 30
cyclic = CyclicLR(mode='exp_range', base_lr = 0.0001, max_lr = 0.01, step_size = cycle)
       

model.fit_generator(
    train_generator,
    steps_per_epoch=2560/batch,
    epochs=120,
    verbose = 1,
    callbacks = [model_checkpoint, cyclic],
    validation_data = val_generator,
    validation_steps = 1100/batch)

#h = clyclic.history
#print(h['lr'])
#print("#######################")
#print(h['acc'])

'''
model.load_weights("./blind_baseline.hdf5")



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

