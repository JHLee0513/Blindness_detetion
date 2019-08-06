#inspired from "Ordinal Regression with Multiple Output"

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
import gc
gc.enable()
gc.collect()

img_target = 256
SIZE = 256
batch = 8
train_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
test_df = pd.read_csv("/nas-homes/joonl4/blind/test.csv")
train_df = train_df.astype(str)
test_df = test_df.astype(str)

# log = open("/home/joonl4/Blindness_detection_binary_log.txt", "a")
# log_fold = 1
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
            img = cv2.imread('/nas-homes/joonl4/blind/train_images/'+sample+'.png')
            img = cv2.resize(img, (SIZE, SIZE))
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
            img = cv2.imread('/nas-homes/joonl4/blind/train_images/'+sample+'.png')
            img = cv2.resize(img, (SIZE, SIZE))
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        return batch_images, batch_y

x = train_df['id_code']
y = to_categorical(train_df['diagnosis'], num_classes=5)

#binarized labeling
for row in y:
    idx = np.argmax(row)
    for i in range(idx+1):
        row[i] = 0.95 
#label smoothening
    for j in range(idx+1, 5):
#print("argmax at " + str(idx) + "0.1 till " + str(idx+1))
        row[j] = 0.05 #label smoothening
    #print(row)
#train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = 0.2, stratify = train_df['diagnosis'])
qwk_ckpt_name = './trash.h5'

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
            def flatten(y):
                #print(np.argmax(y,axis = 1).astype(int))
                #return np.argmax(y, axis=1).astype(int)
                return np.rint(np.sum(y, axis=1)).astype(int) - 1
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
                self.model.save(qwk_ckpt_name)
                #log.write(str(log_fold) + ": " + str(score) + "\n")

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        # sometimes(iaa.size.Crop(percent = (0.1,0.2), keep_size = True)),
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                ]),
                iaa.Invert(0.01, per_channel=True), # invert color channels
                iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.9, 1.1), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-1, 0),
                        first=iaa.Multiply((0.9, 1.1), per_channel=True),
                        second=iaa.ContrastNormalization((0.9, 1.1))
                    )
                ]),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True)

kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=420)
#kf.get_n_splits(x)
train_all = []
evaluate_all = []
for train_idx, test_idx in kf.split(x, train_df['diagnosis']):
    train_all.append(train_idx)
    evaluate_all.append(test_idx)

def get_cv_data(cv_index):
    train_index = train_all[cv_index-1]
    evaluate_index = evaluate_all[cv_index-1]
    x_train = np.array(train_df.id_code[train_index])
    y_train = np.array(y[train_index])
    x_valid = np.array(train_df.id_code[evaluate_index])
    y_valid = np.array(y[evaluate_index])
    return x_train,y_train,x_valid,y_valid

#for cv_index in range(1,6):
for cv_index in range(1,6):
    fold = cv_index
    log_fold = cv_index
    # qwk_ckpt_name = '/nas-homes/joonl4/blind_weights/raw_effnet_pretrained_binary_smoothen_kappa_fold'+str(fold)+'.h5'
    train_x, train_y, val_x, val_y = get_cv_data(cv_index)
    train_generator = My_Generator(train_x, train_y, batch, is_train=True)
    train_mixup = My_Generator(train_x, train_y, batch, is_train=True, mix=True, augment=True)
    val_generator = My_Generator(val_x, val_y, batch, is_train=False)
    qwk = QWKEvaluation(validation_data=(val_generator, val_y),
                        batch_size=batch, interval=1)
#model = ResNet50(include_top = False, weights = 'imagenet', 
#                    input_shape = (img_target,img_target,3), pooling = 'avg') #pooling = 'avg'
#model = Xception(include_top = False, weights = 'imagenet', input_shape = (img_target,img_target,3), pooling = 'max')
    model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights = 'imagenet', include_top = False, pooling = None)
    for layers in model.layers:
        layers.trainable=True
    inputs = model.input
    x = model.output
    # https://www.kaggle.com/ratan123/aptos-keras-efficientnet-with-attention-baseline/comments#Attention-model
    # here we do an attention mechanism to turn pixels in the GAP on an off
    pt_depth = model.get_output_shape_at(0)[-1]
    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(x))
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(1, 
                    kernel_size = (1,1), 
                    padding = 'valid', 
                    activation = 'sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
    out_layer = Dense(5, activation = 'sigmoid')(dr_steps)
    model = Model(inputs, out_layer)
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr = 1e-3),
                metrics= ['accuracy', 'mse'])
    model.summary()
    model.load_weights("/nas-homes/joonl4/blind_weights/raw_pretrain_effnet_B4.hdf5", by_name = True)
    # model.load_weights('/nas-homes/joonl4/blind_weights/raw_effnet_pretrained_binary_smoothen_fold_v2'+str(fold)+'.hdf5')
    save_model_name = '/nas-homes/joonl4/blind_weights/raw_effnet_pretrained_binary_smoothen_fold_v7'+str(fold)+'.hdf5'
    model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                    mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)
    #csv = CSVLogger('./raw_effnet_pretrained_binary_fold'+str(fold)+'.csv', separator=',', append=False)
    cycle = 2560/batch * 30
    cyclic = CyclicLR(mode='exp_range', base_lr = 0.0001, max_lr = 0.003, step_size = cycle)  
    #model.load_weights(save_model_name)
    model.fit_generator(
        train_generator,
        steps_per_epoch=2560/batch,
        epochs=10,
        verbose = 1,
        #initial_epoch = 14,
        callbacks = [model_checkpoint, qwk],
        validation_data = val_generator,
        validation_steps = 1100/batch,
        workers=1, use_multiprocessing=False)
    model.load_weights(save_model_name)
    model.save("/nas-homes/joonl4/blind_weights/raw_effnet_pretrained_binary_smoothen_fold_v7"+str(fold)+ ".h5")
    '''
    model.load_weights(save_model_name)
    model.compile(loss='binary_crossentropy', optimizer = SGD(lr = 0.003, momentum = 0.9, nesterov = True),
                metrics= ['accuracy', 'mse'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=2560/batch,
        epochs=30,
        verbose = 1,
        callbacks = [cyclic, model_checkpoint, qwk],
        validation_data = val_generator,
        validation_steps = 1100/batch,
        workers=1, use_multiprocessing=False)
    fold += 1
    #model.load_weights(save_model_name)

    #model.save('raw_effnet_pretrained_v2.h5')
    '''
