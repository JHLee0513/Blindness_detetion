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
import gc
gc.enable()
gc.collect()

img_target = 380
SIZE = 380
IMG_SIZE = 380
batch = 8
train_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
train_df['id_code'] += '.png'
# train_df = train_df.astype(str)
# df_2019 = train_df[train_df['id_code'].str.contains(".png")]

# train_2019, val_2019 = train_test_split(df_2019, test_size = 0.2, random_state = 420, stratify = df_2019['diagnosis'])
# train_2019 = train_2019.reset_index(drop = True)
# val = val_2019.reset_index(drop = True)

# train_df = train_df[~train_df.id_code.isin(val_2019.id_code)]
# train = train_df.reset_index(drop = True)

train, val = train_test_split(train_df, test_size = 0.2, random_state = None, stratify = train_df['diagnosis'])

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
    # image = crop_image_from_gray(image)
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

val_seq = iaa.Sequential([
    sometimes(iaa.size.Crop(percent = (0.1, 0.2), keep_size = True))
])

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        sometimes(iaa.size.Crop(percent = (0.05, 0.1), keep_size = True)),
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-80, 80), # rotate by -45 to +45 degrees
            # shear=(-5, 5), # shear by -16 to +16 degrees
            # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        ,iaa.SomeOf((0, 5),
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
                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                # ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                # iaa.OneOf([
                #     iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                #     iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                # ]),
                # iaa.Invert(0.01, per_channel=True), # invert color channels
                # iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                # iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                # iaa.OneOf([
                #     iaa.Multiply((0.9, 1.1), per_channel=0.5),
                #     iaa.FrequencyNoiseAlpha(
                #         exponent=(-1, 0),
                #         first=iaa.Multiply((0.9, 1.1), per_channel=True),
                #         second=iaa.ContrastNormalization((0.9, 1.1))
                #     )
                # ]),
                # # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                # # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True)


def build_model(freeze = False):
    model = EfficientNetB4(input_shape = (img_target, img_target, 3), weights = 'imagenet', include_top = False, pooling = None)
    for layers in model.layers:
        layers.trainable= not freeze
    inputs = model.input
    x = model.output
    x = GlobalAveragePooling2D()(x)
    # bn_features = BatchNormalization()(x)
    # pt_depth = model.get_output_shape_at(0)[-1]
    # attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu', name = 'ATTN1')(Dropout(0.5)(bn_features))
    # attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu', name = 'ATTN2')(attn_layer)
    # attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu', name = 'ATTN3')(attn_layer)
    # attn_layer = Conv2D(1, 
    #                 kernel_size = (1,1), 
    #                 padding = 'valid', 
    #                 activation = 'sigmoid',
    #                 name = 'ATTN4')(attn_layer)
    # # fan it out to all of the channels
    # up_c2_w = np.ones((1, 1, 1, pt_depth))
    # up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
    #             activation = 'linear', use_bias = False, weights = [up_c2_w], name = 'ATTN5')
    # up_c2.trainable = False
    # attn_layer = up_c2(attn_layer)
    # mask_features = multiply([attn_layer, bn_features])
    # gap_features = GlobalAveragePooling2D(name='GAP')(mask_features)
    # gap_mask = GlobalAveragePooling2D(name='GAP2')(attn_layer)
    # # to account for missing values from the attention model
    # gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    # gap_dr = Dropout(0.25)(gap)
    # dr_steps = Dropout(0.25)(Dense(128, activation = 'relu', name = 'ATTN6')(gap_dr))
    # out_layer = Dense(1, activation = None, name = 'ATTN_regressor') (dr_steps)
    out_layer = Dense(1, activation = None, name = 'normal_regressor') (Dropout(0.4)(x))
    model = Model(inputs, out_layer)
    return model

from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces

from keras.optimizers import Optimizer

class AdamW(Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay (L2 penalty) (default: 0.025).
        batch_size: integer >= 1. Batch size used during training.
        samples_per_epoch: integer >= 1. Number of samples (training points) per epoch.
        epochs: integer >= 1. Total number of epochs for training. 
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.025, 
                 batch_size=1, samples_per_epoch=1, 
                 epochs=1, **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.batch_size = K.variable(batch_size, name='batch_size')
            self.samples_per_epoch = K.variable(samples_per_epoch, name='samples_per_epoch')
            self.epochs = K.variable(epochs, name='epochs')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        '''Bias corrections according to the Adam paper
        '''
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            
            '''Schedule multiplier eta_t = 1 for simple AdamW
            According to the AdamW paper, eta_t can be fixed, decay, or 
            also be used for warm restarts (AdamWR to come). 
            '''
            eta_t = 1.
            p_t = p - eta_t*(lr_t * m_t / (K.sqrt(v_t) + self.epsilon))
            if self.weight_decay != 0:
                '''Normalized weight decay according to the AdamW paper
                '''
                w_d = self.weight_decay*K.sqrt(self.batch_size/(self.samples_per_epoch*self.epochs))
                p_t = p_t - eta_t*(w_d*p) 

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'batch_size': int(K.get_value(self.batch_size)),
                  'samples_per_epoch': int(K.get_value(self.samples_per_epoch)),
                  'epochs': int(K.get_value(self.epochs)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# for cv_index in range(1,6):
for cv_index in range(1):
    fold = cv_index
    train_x = train['id_code']
    train_y = train['diagnosis'].astype(int)
    val_x = val['id_code']
    val_y = val['diagnosis'].astype(int)
    train_generator = My_Generator(train_x, train_y, batch, is_train=True, augment=False)
    val_generator = My_Generator(val_x, val_y, batch, is_train=False)
    qwk = QWKEvaluation(validation_data=(val_generator, val_y),
                        batch_size=batch, interval=1)
    model = build_model(freeze = False)
    model.load_weights("/nas-homes/joonl4/blind_weights/raw_effnet_pretrained_regression_fold_v110.hdf5")
    save_model_name = '/nas-homes/joonl4/blind_weights/raw_effnet_pretrained_regression_fold_v111.hdf5'
    model_checkpoint = ModelCheckpoint(save_model_name,monitor= 'val_loss',
                                    mode = 'min', save_best_only=True, verbose=1,save_weights_only = True)

    model.compile(loss='mse', optimizer = Adam(lr=1e-3),
                metrics= ['accuracy'])
    cycle = len(train_y)/batch * 15
    cyclic = CyclicLR(mode='exp_range', base_lr = 1e-4, max_lr = 1e-3, step_size = cycle)  
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_y)/batch,
        epochs=45,
        verbose = 1,
        callbacks = [model_checkpoint, qwk, cyclic],
        validation_data = val_generator,
        validation_steps = len(val_y)/batch,
        workers=1, use_multiprocessing=False)
    model.load_weights(save_model_name)
    model.save("/nas-homes/joonl4/blind_weights/raw_effnet_pretrained_regression_fold_v111.h5")