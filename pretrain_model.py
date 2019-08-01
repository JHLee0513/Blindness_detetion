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
from keras.utils import Sequence, to_categorical
from keras.callbacks import Callback
from efficientnet import EfficientNetB4
from keras.callbacks import LearningRateScheduler
import scipy
import gc
gc.enable()
gc.collect()

img_target = 256
batch = 8
train_df = pd.read_csv("/nas-homes/joonl4/blind_2015/trainLabels.csv")
train_df2 = pd.read_csv("/nas-homes/joonl4/blind_2015/retinopathy_solution.csv")
train_df2.rename(columns={"level": "label"})
print(train_df2.head())
train_df = pd.concat([train_df, train_df2], axis = 0, sort = False)
train_df.reset_index(drop = True)
val_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
train_df['image'] = train_df['image'].astype(str) + ".jpeg"
train_df = train_df.astype(str)
val_df['id_code'] = val_df['id_code'].astype(str) + ".png"
val_df = val_df.astype(str)

x = train_df['image']
y = to_categorical(train_df['label'], num_classes=5)
val_x = val_df['id_code']
val_y = val_df['diagnosis']

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

qwk_ckpt_name = '.h5'

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
                return np.argmax(y, axis=1).reshape(-1)
                # return np.sum(y.astype(int), axis=1) - 1
            
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
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
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


model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights=None, include_top = False, pooling = 'avg')

for layers in model.layers:
    layers.trainable=True

inputs = model.input
x = model.output
x = Dropout(rate = 0.4) (x)
x = Dense(512, activation = 'elu', name = "fc") (x)
x = Dropout(rate = 0.25) (x)
x = Dense(5, activation = 'softmax', name = 'classifier') (x)

model = Model(inputs, x)
save_model_name = '/nas-homes/joonl4/blind_weights/raw_pretrain_effnet_fulldata.hdf5'
model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                   mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)

qwk_ckpt_name = './raw_effnet_pretrained_fulldata.h5'
train_generator = My_Generator(x, y, 8, is_train=True)
train_mixup = My_Generator(x, y, 8, is_train=True, mix=True, augment=True)
val_generator = My_Generator(val_x, val_y, 8, is_train=False)
qwk = QWKEvaluation(validation_data=(val_generator, val_y),
                        batch_size=16, interval=1)

# warmup
model.compile(loss='binary_crossentropy', optimizer =Adam(lr=1e-3),
             metrics= [])
model.fit_generator(
    train_generator,
    steps_per_epoch=88702/batch,
    epochs=3,
    verbose = 1,
    callbacks = [model_checkpoint, qwk],
    validation_data = val_generator,
    validation_steps = 3662/batch)

cycle = 88702/batch * 10
cyclic = CyclicLR(mode='exp_range', base_lr = 0.00001, max_lr = 0.001, step_size = cycle)

model.compile(loss='binary_crossentropy', optimizer = SGD(lr=0.001, momentum = 0.95, nesterov = True),
             metrics= [])

model.fit_generator(
    train_mixup,
    steps_per_epoch=88702/batch,
    epochs=10,
    verbose = 1,
    callbacks = [model_checkpoint, qwk],
    validation_data = val_generator,
    validation_steps = 3662/batch)
