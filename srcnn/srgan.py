import json
import os
import time

import h5py
import keras
import numpy as np
from keras import backend
from keras import backend as K
from keras.layers import Input, add, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from scipy.misc import imresize, imsave
from scipy.ndimage.filters import gaussian_filter

from pyduyp.utils.dl.ops.layers import Normalize, Denormalize
from pyduyp.utils.keras_ops import fit as bypass_fit
from pyduyp.utils.keras_ops import smooth_gan_labels
from pyduyp.utils.loss import ContentVGGRegularizer, AdversarialLossRegularizer, dummy_loss, psnr

print("================= {} ======================".format(keras.__version__))
# THEANO_WEIGHTS_PATH_NO_TOP = r'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = r"https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

if not os.path.exists("weights/"):
    os.makedirs("weights/")

if not os.path.exists("val_images/"):
    os.makedirs("val_images/")

backend.set_image_dim_ordering('tf')

if K.image_dim_ordering() == "th":
    channel_axis = 1
else:
    channel_axis = -1


class VGGNetwork:
    '''
    Helper class to load VGG and its weights to the FastNet model
    '''

    def __init__(self, img_width=384, img_height=384, vgg_weight=1.0):
        self.img_height = img_height
        self.img_width = img_width
        self.vgg_weight = vgg_weight

        self.vgg_layers = None

    def append_vgg_network(self, x_in, true_X_input, pre_train=False):
        print("================= 56 {}, {} ==================".format(x_in, true_X_input))
        # Append the initial inputs to the outputs of the SRResNet
        x = add([x_in, true_X_input])
        print("==================== 59 x : {} =====================".format(x))
        # Normalize the inputs via custom VGG Normalization layer
        x = Normalize(name="normalize_vgg")(x)
        print("==================== 62 x : {} =====================".format(x))

        # Begin adding the VGG layers
        x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv1_1', padding='same')(x)
        print("==================== 66 x : {} =====================".format(x))

        x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv1_2', padding='same')(x)
        print("==================== 69 x : {} =====================".format(x))
        x = MaxPooling2D(name='vgg_maxpool1')(x)
        print("==================== 71 x : {} =====================".format(x))

        x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv2_1', padding='same')(x)
        print("==================== 74 x : {} =====================".format(x))

        if pre_train:
            vgg_regularizer2 = ContentVGGRegularizer(weight=self.vgg_weight)
            x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv2_2',
                       padding='same', activity_regularizer=vgg_regularizer2)(x)
        else:
            x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv2_2',
                       padding='same')(x)
        print("====================83 x : {} =====================".format(x))

        x = MaxPooling2D(name='vgg_maxpool2')(x)
        print("==================== 86 x : {} =====================".format(x))

        x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv3_1', padding='same')(x)
        print("==================== 89 x : {} =====================".format(x))
        x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv3_2', padding='same')(x)
        print("==================== 91x : {} =====================".format(x))

        x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv3_3', padding='same')(x)
        print("==================== 94 x : {} =====================".format(x))
        x = MaxPooling2D(pool_size=(2, 2), name='vgg_maxpool3')(x)
        print("==================== 96 x : {} =====================".format(x))

        x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv4_1', padding='same')(x)
        print("==================== 99 x : {} =====================".format(x))
        x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv4_2', padding='same')(x)
        print("==================== 101 x : {} =====================".format(x))

        x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv4_3', padding='same')(x)
        print("==================== 104  x : {} =====================".format(x))
        x = MaxPooling2D(pool_size=(2, 2), name='vgg_maxpool4')(x)
        print("==================== x : {} =====================".format(x))

        x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv5_1', padding='same')(x)
        print("==================== 109 x : {} =====================".format(x))
        x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv5_2', padding='same')(x)
        print("==================== 111 x : {} =====================".format(x))

        if not pre_train:
            vgg_regularizer5 = ContentVGGRegularizer(weight=self.vgg_weight)
            x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv5_3',
                       padding='same',
                       activity_regularizer=vgg_regularizer5)(x)
        else:
            x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu', name='vgg_conv5_3',
                       padding='same')(x)
        print("==================== 121 x : {} =====================".format(x))

        x = MaxPooling2D(pool_size=(2, 2), name='vgg_maxpool5')(x)
        print("==================== 124 x : {} =====================".format(x))
        return x

    def load_vgg_weight(self, model):
        print("============ Start load_vgg_weight ====================")
        weights = get_file('D:\\alvin_py\\srcnn\\checkpoints\\srgan\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                           TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
        f = h5py.File(weights)

        layer_names = [name for name in f.attrs['layer_names']]
        print("================== layer names : {} ====================".format(layer_names))
        if self.vgg_layers is None:
            self.vgg_layers = [layer for layer in model.layers if 'vgg_' in layer.name]

        for i, layer in enumerate(self.vgg_layers):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        # Freeze all VGG layers
        for layer in self.vgg_layers:
            layer.trainable = False

        return model


class DiscriminatorNetwork:
    print("=============================Start DiscriminatorNetworks =============================")

    def __init__(self, img_width=384, img_height=384, adversarial_loss_weight=1, small_model=False):
        self.img_width = img_width
        self.img_height = img_height
        self.adversarial_loss_weight = adversarial_loss_weight
        self.small_model = small_model

        self.k = 3
        # self.mode = 2
        self.weights_path = "weights/Discriminator_weights.h5"

        self.gan_layers = None

    def append_gan_network(self, true_X_input):
        print("================== Start append_gan_network ====================")
        # Normalize the inputs via custom VGG Normalization layer
        x = Normalize(type="gan", value=127.5, name="gan_normalize")(true_X_input)
        print("==================== 170 x  : {} =====================".format(x))

        x = Conv2D(filters=64, kernel_size=self.k, strides=(self.k, self.k), padding='same', name='gan_conv1_1')(x)
        print("==================== 173 x : {} =====================".format(x))
        x = LeakyReLU(0.3, name="gan_lrelu1_1")(x)
        print("==================== 175 x : {} =====================".format(x))

        x = Conv2D(filters=64, kernel_size=self.k, strides=(self.k, self.k), padding='same', name='gan_conv1_2')(x)
        print("==================== 178 x : {} =====================".format(x))
        x = LeakyReLU(0.3, name='gan_lrelu1_2')(x)
        print("==================== 180 x : {} =====================".format(x))
        x = BatchNormalization(axis=channel_axis, name='gan_batchnorm1_1')(x)
        print("==================== 182 x : {} =====================".format(x))

        filters = [128, 256] if self.small_model else [128, 256, 512]

        for i, nb_filters in enumerate(filters):
            for j in range(2):
                x = Conv2D(filters=nb_filters, kernel_size=self.k, strides=(self.k, self.k), padding='same',
                           name='gan_conv%d_%d' % (i + 2, j + 1))(x)
                # subsample
                x = LeakyReLU(0.3, name='gan_lrelu_%d_%d' % (i + 2, j + 1))(x)
                x = BatchNormalization(axis=channel_axis, name='gan_batchnorm%d_%d' % (i + 2, j + 1))(x)
        print("====================192 x : {} =====================".format(x))

        x = Flatten(name='gan_flatten')(x)
        print("==================== 195 x : {} =====================".format(x))

        output_dim = 128 if self.small_model else 1024

        x = Dense(output_dim, name='gan_dense1')(x)
        print("==================== 200 x : {} =====================".format(x))
        x = LeakyReLU(0.3, name='gan_lrelu5')(x)
        print("==================== 202 x : {} =====================".format(x))

        gan_regulrizer = AdversarialLossRegularizer(weight=self.adversarial_loss_weight)
        x = Dense(2, activation="softmax", activity_regularizer=gan_regulrizer, name='gan_output')(x)
        print("==================== 206 x : {} =====================".format(x))

        return x

    def set_trainable(self, model, value=True):
        if self.gan_layers is None:
            disc_model = [layer for layer in model.layers if 'model' in layer.name][0]

            self.gan_layers = [layer for layer in disc_model.layers if 'gan_' in layer.name]

        for layer in self.gan_layers:
            layer.trainable = value

    def load_gan_weights(self, model):
        print("================ Start load_gan_weights ====================")
        f = h5py.File(self.weights_path)

        layer_names = [name for name in f.attrs['layer_names']]
        layer_names = layer_names[1:]

        if self.gan_layers is None:
            self.gan_layers = [layer for layer in model.layers
                               if 'gan_' in layer.name]

        for i, layer in enumerate(self.gan_layers):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        print("GAN Model weights loaded.")
        return model

    def save_gan_weights(self, model):
        print('GAN Weights are being saved.')
        model.save_weights(self.weights_path, overwrite=True)
        print('GAN Weights saved.')


class GenerativeNetwork:
    def __init__(self, img_width=96, img_height=96, batch_size=16, nb_upscales=2, small_model=False,
                 content_weight=1, tv_weight=2e5, gen_channels=64):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.small_model = small_model
        self.nb_scales = nb_upscales

        self.content_weight = content_weight
        self.tv_weight = tv_weight

        self.filters = gen_channels
        self.mode = 2
        self.init = 'glorot_uniform'

        self.sr_res_layers = None
        self.sr_weights_path = r"D:\srgan\srgan_keras\weights\SRGAN_du.h5"

        self.output_func = None

    def create_sr_model(self, ip):
        print("=========================== 266 ip : {}　===============".format(ip))
        x1 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_1', kernel_initializer=self.init)(ip)
        x1 = BatchNormalization(axis=channel_axis, name='G_bn_1')(x1)
        x1 = LeakyReLU(alpha=0.25, name='G_lr_1')(x1)
        print("======================== 268 x: {} =============================".format(x1))

        x2 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_2', kernel_initializer=self.init)(x1)
        x2 = BatchNormalization(axis=channel_axis, name='G_bn_2')(x2)
        x2 = LeakyReLU(alpha=0.25, name='G_lr_2')(x2)
        print("======================== 277 x: {} =============================".format(x2))
        x3 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_3', kernel_initializer=self.init)(x2)
        x3 = BatchNormalization(axis=channel_axis, name='G_bn_3')(x3)
        x3 = LeakyReLU(alpha=0.25, name='G_lr_3')(x3)
        print("======================== 282  x: {} =============================".format(x3))

        x4 = add([x1, x3], name='merge_1')
        print("================= line 286 x4 : {} =================".format(x4))
        x5 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_4', kernel_initializer=self.init)(x4)
        x5 = BatchNormalization(axis=channel_axis, name='G_bn_4')(x5)
        x5 = LeakyReLU(alpha=0.25, name='G_lr_4')(x5)
        print("======================== 291 x: {} =============================".format(x5))
        x6 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_5', kernel_initializer=self.init)(x5)
        x6 = BatchNormalization(axis=channel_axis, name='G_bn_5')(x6)
        x6 = LeakyReLU(alpha=0.25, name='G_lr_5')(x6)
        print("======================== 296  x: {} =============================".format(x6))

        x7 = add([x4, x6], name="merge_2")
        print(x7)

        x8 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_6', kernel_initializer=self.init)(x7)
        x8 = BatchNormalization(axis=channel_axis, name='G_bn_6')(x8)
        x8 = LeakyReLU(alpha=0.25, name='G_lr_6')(x8)
        print("======================== 277 x: {} =============================".format(x8))
        x9 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_7', kernel_initializer=self.init)(x8)
        x9 = BatchNormalization(axis=channel_axis, name='G_bn_7')(x9)
        x9 = LeakyReLU(alpha=0.25, name='G_lr_7')(x9)
        print("======================== 282  x: {} =============================".format(x9))

        x10 = add([x7, x9], name="merge_3")
        print(x10)

        x11 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_8', kernel_initializer=self.init)(x10)
        x11 = BatchNormalization(axis=channel_axis, name='G_bn_8')(x11)
        x11 = LeakyReLU(alpha=0.25, name='G_lr_8')(x11)
        print("======================== 277 x: {} =============================".format(x8))
        x12 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_9', kernel_initializer=self.init)(x11)
        x12 = BatchNormalization(axis=channel_axis, name='G_bn_9')(x12)
        x12 = LeakyReLU(alpha=0.25, name='G_lr_9')(x12)
        print("======================== 282  x: {} =============================".format(x12))

        x13 = add([x10, x12], name='merge_4')

        x14 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_10', kernel_initializer=self.init)(x13)
        x14 = BatchNormalization(axis=channel_axis, name='G_bn_10')(x14)
        x14 = LeakyReLU(alpha=0.25, name='G_lr_10')(x14)
        print("======================== 277 x: {} =============================".format(x8))
        x15 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_11', kernel_initializer=self.init)(x14)
        x15 = BatchNormalization(axis=channel_axis, name='G_bn_11')(x15)
        x15 = LeakyReLU(alpha=0.25, name='G_lr_11')(x15)
        print("======================== 282  x: {} =============================".format(x15))

        x16 = add([x13, x15], name="merge_5")
        print(x16)
        x1 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_12', kernel_initializer=self.init)(ip)
        x1 = BatchNormalization(axis=channel_axis, name='G_bn_12')(x1)
        x1 = LeakyReLU(alpha=0.25, name='G_lr_12')(x1)
        print("======================== 268 x: {} =============================".format(x1))

        x2 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_13', kernel_initializer=self.init)(x1)
        x2 = BatchNormalization(axis=channel_axis, name='G_bn_13')(x2)
        x2 = LeakyReLU(alpha=0.25, name='G_lr_13')(x2)
        print("======================== 277 x: {} =============================".format(x2))
        x3 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_14', kernel_initializer=self.init)(x2)
        x3 = BatchNormalization(axis=channel_axis, name='G_bn_14')(x3)
        x3 = LeakyReLU(alpha=0.25, name='G_lr_14')(x3)
        print("======================== 282  x: {} =============================".format(x3))

        x4 = add([x1, x3], name='merge_6')
        print("================= line 286 x4 : {} =================".format(x4))
        x5 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_15', kernel_initializer=self.init)(x4)
        x5 = BatchNormalization(axis=channel_axis, name='G_bn_15')(x5)
        x5 = LeakyReLU(alpha=0.25, name='G_lr_15')(x5)
        print("======================== 291 x: {} =============================".format(x5))
        x6 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_16', kernel_initializer=self.init)(x5)
        x6 = BatchNormalization(axis=channel_axis, name='G_bn_16')(x6)
        x6 = LeakyReLU(alpha=0.25, name='G_lr_16')(x6)
        print("======================== 296  x: {} =============================".format(x6))

        x7 = add([x4, x6], name="merge_7")
        print(x7)

        x8 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_17', kernel_initializer=self.init)(x7)
        x8 = BatchNormalization(axis=channel_axis, name='G_bn_17')(x8)
        x8 = LeakyReLU(alpha=0.25, name='G_lr_17')(x8)
        print("======================== 277 x: {} =============================".format(x8))
        x9 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                    name='G_conv_18', kernel_initializer=self.init)(x8)
        x9 = BatchNormalization(axis=channel_axis, name='G_bn_18')(x9)
        x9 = LeakyReLU(alpha=0.25, name='G_lr_18')(x9)
        print("======================== 282  x: {} =============================".format(x9))

        x10 = add([x7, x9], name="merge_8")
        print(x10)

        x11 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_19', kernel_initializer=self.init)(x10)
        x11 = BatchNormalization(axis=channel_axis, name='G_bn_19')(x11)
        x11 = LeakyReLU(alpha=0.25, name='G_lr_19')(x11)
        print("======================== 277 x: {} =============================".format(x8))
        x12 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_20', kernel_initializer=self.init)(x11)
        x12 = BatchNormalization(axis=channel_axis, name='G_bn_20')(x12)
        x12 = LeakyReLU(alpha=0.25, name='G_lr_20')(x12)
        print("======================== 282  x: {} =============================".format(x12))

        x13 = add([x10, x12], name='merge_9')

        x14 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_21', kernel_initializer=self.init)(x13)
        x14 = BatchNormalization(axis=channel_axis, name='G_bn_21')(x14)
        x14 = LeakyReLU(alpha=0.25, name='G_lr_21')(x14)
        print("======================== 277 x: {} =============================".format(x8))
        x15 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_22', kernel_initializer=self.init)(x14)
        x15 = BatchNormalization(axis=channel_axis, name='G_bn_22')(x15)
        x15 = LeakyReLU(alpha=0.25, name='G_lr_22')(x15)
        print("======================== 282  x: {} =============================".format(x15))

        x16 = add([x13, x15], name="merge_10")
        print(x16)

        x14 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_23', kernel_initializer=self.init)(x13)
        x14 = BatchNormalization(axis=channel_axis, name='G_bn_23')(x14)
        x14 = LeakyReLU(alpha=0.25, name='G_lr_23')(x14)
        print("======================== 277 x: {} =============================".format(x8))
        x15 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_24', kernel_initializer=self.init)(x14)
        x15 = BatchNormalization(axis=channel_axis, name='G_bn_24')(x15)
        x15 = LeakyReLU(alpha=0.25, name='G_lr_24')(x15)
        print("======================== 282  x: {} =============================".format(x15))

        x16 = add([x13, x15], name="merge_11")
        print(x16)

        x14 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_25', kernel_initializer=self.init)(x13)
        x14 = BatchNormalization(axis=channel_axis, name='G_bn_25')(x14)
        x14 = LeakyReLU(alpha=0.25, name='G_lr_25')(x14)
        print("======================== 277 x: {} =============================".format(x8))
        x15 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_26', kernel_initializer=self.init)(x14)
        x15 = BatchNormalization(axis=channel_axis, name='G_bn_26')(x15)
        x15 = LeakyReLU(alpha=0.25, name='G_lr_26')(x15)
        print("======================== 282  x: {} =============================".format(x15))

        x16 = add([x13, x15], name="merge_12")
        print(x16)
        x17 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_27', kernel_initializer=self.init)(x16)
        x17 = BatchNormalization(axis=channel_axis, name='G_bn_27')(x17)
        x17 = LeakyReLU(alpha=0.25, name='G_lr_27')(x17)
        print("======================== 277 x: {} =============================".format(x17))
        x18 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_28', kernel_initializer=self.init)(x17)
        x18 = BatchNormalization(axis=channel_axis, name='G_bn_28')(x18)
        x18 = LeakyReLU(alpha=0.25, name='G_lr_28')(x18)
        print("======================== 282  x: {} =============================".format(x18))

        x19 = add([x16, x18], name="merge_13")
        print(x19)
        x20 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_29', kernel_initializer=self.init)(x19)
        x20 = BatchNormalization(axis=channel_axis, name='G_bn_29')(x20)
        x20 = LeakyReLU(alpha=0.25, name='G_lr_29')(x20)
        print("======================== 277 x: {} =============================".format(x20))
        x21 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_30', kernel_initializer=self.init)(x20)
        x21 = BatchNormalization(axis=channel_axis, name='G_bn_30')(x21)
        x21 = LeakyReLU(alpha=0.25, name='G_lr_30')(x21)
        print("======================== 282  x: {} =============================".format(x15))

        x22 = add([x19, x21], name="merge_14")
        print(x22)
        x23 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_31', kernel_initializer=self.init)(x22)
        x23 = BatchNormalization(axis=channel_axis, name='G_bn_31')(x23)
        x23 = LeakyReLU(alpha=0.25, name='G_lr_31')(x23)
        print("======================== 277 x: {} =============================".format(x23))
        x24 = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_32', kernel_initializer=self.init)(x23)
        x24 = BatchNormalization(axis=channel_axis, name='G_bn_32')(x24)
        x24 = LeakyReLU(alpha=0.25, name='G_lr_32')(x24)
        print("======================== 282  x: {} =============================".format(x24))

        x25 = add([x22, x24], name="merge_15")
        print("=================== 480 x25 : {} ====================".format(x25))

        x26 = Conv2D(filters=self.filters * 2, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_33', kernel_initializer=self.init)(x25)
        print("==================== 484 x26 : {} ================".format(x26))
        x27 = LeakyReLU(alpha=0.25, name='G_lr_33')(x26)
        x28 = UpSampling2D(size=(2, 2))(x27)
        print("================ 487 x28 : {} =================".format(x28))
        x29 = Conv2D(filters=self.filters * 2, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_34', kernel_initializer=self.init)(x28)
        print("==================== 490 x29 : {} ================".format(x29))
        x30 = LeakyReLU(alpha=0.25, name='G_lr_34')(x29)
        x31 = Conv2D(filters=self.filters * 2, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_35', kernel_initializer=self.init)(x30)
        print("==================== 494 x31 : {} ================".format(x31))
        x32 = LeakyReLU(alpha=0.25, name='G_lr_35')(x31)
        x33 = UpSampling2D(size=(2, 2))(x32)
        print("=============== 497 x33: {} =============".format(x33))
        x34 = Conv2D(filters=self.filters * 2, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_36', kernel_initializer=self.init)(x33)
        print("==================== 500 x31 : {} ================".format(x34))
        x35 = LeakyReLU(alpha=0.25, name='G_lr_36')(x34)
        x36 = Conv2D(filters=3, kernel_size=3, strides=(1, 1), activation='linear', padding='same',
                     name='G_conv_37', kernel_initializer=self.init)(x35)
        print("==================== 504 x36 : {} ================".format(x36))
        x37 = Denormalize(name='sr_res_conv_denorm')(x36)
        print("================== 506 x37 : {} ===================".format(x37))
        print("\n" * 5)
        return x37

    def set_trainable(self, model, value=True):
        if self.sr_res_layers is None:
            self.sr_res_layers = [layer for layer in model.layers if 'sr_res_' in layer.name]

        for layer in self.sr_res_layers:
            layer.trainable = value

    def get_generator_output(self, input_img, srgan_model):
        if self.output_func is None:
            gen_output_layer = [layer for layer in srgan_model.layers if layer.name == "sr_res_conv_denorm"][0]
            print("=================== 520 gen output layer : {} ==================".format(gen_output_layer))
            self.output_func = K.function([srgan_model.layers[0].input], [gen_output_layer.output])

        return self.output_func([input_img])


class SRGANNetwork:
    def __init__(self, img_width=96, img_height=96, batch_size=16, nb_scales=2):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.nb_scales = nb_scales

        self.discriminative_network = None  # type: DiscriminatorNetwork
        self.generative_network = None  # type: GenerativeNetwork
        self.vgg_network = None  # type: VGGNetwork

        self.srgan_model_ = None  # type: Model
        self.generative_model_ = None  # type: Model
        self.discriminative_model_ = None  # type: Model

    def build_srgan_pretrain_model(self, use_small_srgan=False):
        print("================== Start build_srgan_pretrain_model ========================")
        large_width = self.img_width * 4
        large_height = self.img_height * 4

        self.generative_network = GenerativeNetwork(self.img_width, self.img_height, self.batch_size, self.nb_scales,
                                                    use_small_srgan)
        print()
        print("================ 548 generative_network : {} =====================".format(self.generative_network))
        self.vgg_network = VGGNetwork(large_width, large_height)

        ip = Input(shape=(self.img_width, self.img_height, 3), name='x_generator')
        ip_vgg = Input(shape=(large_width, large_height, 3), name='x_vgg')  # Actual X images

        sr_output = self.generative_network.create_sr_model(ip)
        self.generative_model_ = Model(ip, sr_output)

        vgg_output = self.vgg_network.append_vgg_network(sr_output, ip_vgg, pre_train=True)

        self.srgan_model_ = Model(input=[ip, ip_vgg],
                                  output=vgg_output)

        self.vgg_network.load_vgg_weight(self.srgan_model_)

        srgan_optimizer = Adam(lr=1e-4)
        generator_optimizer = Adam(lr=1e-4)

        self.generative_model_.compile(generator_optimizer, dummy_loss)
        self.srgan_model_.compile(srgan_optimizer, dummy_loss)

        return self.srgan_model_

    def build_discriminator_pretrain_model(self, use_small_srgan=False, use_small_discriminator=False):
        large_width = self.img_width * 4
        large_height = self.img_height * 4

        self.generative_network = GenerativeNetwork(self.img_width, self.img_height, self.batch_size, self.nb_scales,
                                                    use_small_srgan)
        self.discriminative_network = DiscriminatorNetwork(large_width, large_height,
                                                           small_model=use_small_discriminator)

        ip = Input(shape=(3, self.img_width, self.img_height), name='x_generator')
        ip_gan = Input(shape=(3, large_width, large_height), name='x_discriminator')  # Actual X images

        sr_output = self.generative_network.create_sr_model(ip)
        self.generative_model_ = Model(ip, sr_output)
        # self.generative_network.set_trainable(self.generative_model_, value=False)

        gan_output = self.discriminative_network.append_gan_network(ip_gan)
        self.discriminative_model_ = Model(ip_gan, gan_output)

        generator_out = self.generative_model_(ip)
        gan_output = self.discriminative_model_(generator_out)

        self.srgan_model_ = Model(input=ip, output=gan_output)

        srgan_optimizer = Adam(lr=1e-4)
        generator_optimizer = Adam(lr=1e-4)
        discriminator_optimizer = Adam(lr=1e-4)

        self.generative_model_.compile(generator_optimizer, loss='mse')
        self.discriminative_model_.compile(discriminator_optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.srgan_model_.compile(srgan_optimizer, loss='categorical_crossentropy', metrics=['acc'])

        return self.discriminative_model_

    def build_srgan_model(self, use_small_srgan=False, use_small_discriminator=False):
        large_width = self.img_width * 4
        large_height = self.img_height * 4

        self.generative_network = GenerativeNetwork(self.img_width, self.img_height, self.batch_size,
                                                    nb_upscales=self.nb_scales, small_model=use_small_srgan)
        print("===================== line 394: {} =============".format(self.generative_network))
        self.discriminative_network = DiscriminatorNetwork(large_width, large_height,
                                                           small_model=use_small_discriminator)
        print("===================== line 396: {} =============".format(self.discriminative_network))

        self.vgg_network = VGGNetwork(large_width, large_height)

        ip = Input(shape=(self.img_width, self.img_height, 3), name='x_generator')
        print("==================== ip :{} =========================".format(ip))
        ip_gan = Input(shape=(large_width, large_height, 3), name='x_discriminator')  # Actual X images
        ip_vgg = Input(shape=(large_width, large_height, 3), name='x_vgg')  # Actual X images
        print("================== line 459 {} ======================".format(ip_vgg))
        sr_output = self.generative_network.create_sr_model(ip)
        print("==================== line 405: {} ==================".format(sr_output))
        self.generative_model_ = Model(ip, sr_output)
        gan_output = self.discriminative_network.append_gan_network(ip_gan)
        self.discriminative_model_ = Model(ip_gan, gan_output)

        gan_output = self.discriminative_model_(self.generative_model_.output)

        vgg_output = self.vgg_network.append_vgg_network(self.generative_model_.output, ip_vgg)

        self.srgan_model_ = Model(input=[ip, ip_gan, ip_vgg], output=[gan_output, vgg_output])

        self.vgg_network.load_vgg_weight(self.srgan_model_)

        srgan_optimizer = Adam(lr=1e-4)
        generator_optimizer = Adam(lr=1e-4)
        discriminator_optimizer = Adam(lr=1e-4)

        self.generative_model_.compile(generator_optimizer, dummy_loss)
        self.discriminative_model_.compile(discriminator_optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.srgan_model_.compile(srgan_optimizer, dummy_loss)

        return self.srgan_model_

    def pre_train_srgan(self, image_dir, nb_images=5000, epochs=1, use_small_srgan=False):
        self.build_srgan_pretrain_model(use_small_srgan=use_small_srgan)

        self._train_model(image_dir, nb_images=nb_images, epochs=epochs, pre_train_srgan=True,
                          load_generative_weights=True)

    def pre_train_discriminator(self, image_dir, nb_images=50000, epochs=1, batch_size=128,
                                use_small_discriminator=False):

        self.batch_size = batch_size
        self.build_discriminator_pretrain_model(use_small_discriminator)

        self._train_model(image_dir, nb_images, epochs, pre_train_discriminator=True,
                          load_generative_weights=True)

    def train_full_model(self, image_dir, nb_images=50000, epochs=10, use_small_srgan=False,
                         use_small_discriminator=False):
        print("=============== Start train full model ========================")
        self.build_srgan_model(use_small_srgan, use_small_discriminator)

        self._train_model(image_dir, nb_images, epochs, load_generative_weights=True, load_discriminator_weights=True)

    def _train_model(self, image_dir, nb_images, epochs=10, pre_train_srgan=False,
                     pre_train_discriminator=False, load_generative_weights=False, load_discriminator_weights=False,
                     save_loss=True, disc_train_flip=0.1):
        print("========== 672  Start Train_model =================")
        assert self.img_width >= 16, "Minimum image width must be at least 16"
        assert self.img_height >= 16, "Minimum image height must be at least 16"

        if load_generative_weights:
            try:
                self.generative_model_.load_weights(self.generative_network.sr_weights_path)
                print("Generator weights loaded.")
            except:
                print("Could not load generator weights.")

        if load_discriminator_weights:
            try:
                self.discriminative_network.load_gan_weights(self.srgan_model_)
                print("Discriminator weights loaded.")
            except:
                print("Could not load discriminator weights.")

        datagen = ImageDataGenerator(rescale=1. / 255)
        img_width = self.img_width * 4
        img_height = self.img_height * 4

        early_stop = False
        iteration = 0
        prev_improvement = -1

        if save_loss:
            if pre_train_srgan:
                loss_history = {'generator_loss': [], 'val_psnr': [], }
            elif pre_train_discriminator:
                loss_history = {'discriminator_loss': [], 'discriminator_acc': [], }
            else:
                loss_history = {'discriminator_loss': [], 'discriminator_acc': [], 'generator_loss': [],
                                'val_psnr': [], }

        y_vgg_dummy = np.zeros((self.batch_size * 2, img_width // 32, img_height // 32, 3))  # 5 Max Pools = 2 ** 5 = 32

        print("\n" * 5)
        print("Training SRGAN network")
        for i in range(epochs):
            print()
            print("Epoch : %d" % (i + 1))
            print("image dir : {} ".format(image_dir))
            for x in datagen.flow_from_directory(image_dir, class_mode=None, batch_size=self.batch_size,
                                                 target_size=(img_width, img_height),
                                                 classes=False):
                print("xxxxxxxxxxxxxxxxxxxxx :{}, {}".format(type(x), x.shape))
                try:
                    t1 = time.time()
                    if not pre_train_srgan and not pre_train_discriminator:
                        x_vgg = x.copy() * 255
                    x_temp = x.copy()
                    x_generator = np.empty((self.batch_size, self.img_width, self.img_height, 3))
                    for j in range(self.batch_size):
                        img = gaussian_filter(x_temp[j], sigma=0.1)
                        img = imresize(img, (self.img_width, self.img_height), interp='bicubic')
                        x_generator[j, :, :, :] = img

                    if iteration % 50 == 0 and iteration != 0 and not pre_train_discriminator:
                        print("Validation image..")
                        output_image_batch = self.generative_network.get_generator_output(x_generator, self.srgan_model_)
                        if type(output_image_batch) == list:
                            output_image_batch = output_image_batch[0]
                        mean_axis = (0, 1, 2) if K.image_dim_ordering() == 'tf' else (0, 2, 1)
                        average_psnr = 0.0
                        print('gen img mean :', np.mean(output_image_batch / 255., axis=mean_axis))
                        print('val img mean :', np.mean(x, axis=mean_axis))
                        for x_i in range(self.batch_size):
                            average_psnr += psnr(x[x_i], np.clip(output_image_batch[x_i], 0, 255) / 255.)
                        average_psnr /= self.batch_size
                        if save_loss:
                            loss_history['val_psnr'].append(average_psnr)
                        iteration += self.batch_size
                        t2 = time.time()
                        print("Time required : %0.2f. Average validation PSNR over %d samples = %0.2f" %
                              (t2 - t1, self.batch_size, average_psnr))

                        for x_i in range(self.batch_size):
                            root = r"D:\srgan\srgan_keras\val_images"
                            real_path = os.path.join(root, "val_images/epoch_%d_iteration_%d_num_%d_real_.png" % (i + 1, iteration, x_i + 1))
                            generated_path = os.path.join(root, "val_images/epoch_%d_iteration_%d_num_%d_generated.png" % (i + 1, iteration, x_i + 1))

                            val_x = x[x_i].copy() * 255.
                            # val_x = val_x.transpose((1, 2, 0))
                            val_x = np.clip(val_x, 0, 255).astype('uint8')

                            output_image = output_image_batch[x_i]
                            output_image = output_image.transpose((1, 2, 0))
                            output_image = np.clip(output_image, 0, 255).astype('uint8')

                            imsave(real_path, val_x)
                            imsave(generated_path, output_image)

                        continue

                    if pre_train_srgan:
                        # Train only generator + vgg network

                        # Use custom bypass_fit to bypass the check for same input and output batch size
                        hist = bypass_fit(self.srgan_model_, [x_generator, x * 255], y_vgg_dummy,
                                          batch_size=self.batch_size, nb_epoch=1, verbose=0)
                        sr_loss = hist.history['loss'][0]

                        if save_loss:
                            loss_history['generator_loss'].extend(hist.history['loss'])

                        if prev_improvement == -1:
                            prev_improvement = sr_loss

                        improvement = (prev_improvement - sr_loss) / prev_improvement * 100
                        prev_improvement = sr_loss

                        iteration += self.batch_size
                        t2 = time.time()

                        print("Iter : %d / %d | Improvement : %0.2f percent | Time required : %0.2f seconds | "
                              "Generative Loss : %0.2f" % (iteration, nb_images, improvement, t2 - t1, sr_loss))
                    elif pre_train_discriminator:
                        # Train only discriminator
                        X_pred = self.generative_model_.predict(x_generator, self.batch_size)

                        X = np.concatenate((X_pred, x * 255))

                        # Using soft and noisy labels
                        if np.random.uniform() > disc_train_flip:
                            # give correct classifications
                            y_gan = [0] * self.batch_size + [1] * self.batch_size
                        else:
                            # give wrong classifications (noisy labels)
                            y_gan = [1] * self.batch_size + [0] * self.batch_size

                        y_gan = np.asarray(y_gan, dtype=np.int).reshape(-1, 1)
                        y_gan = to_categorical(y_gan, num_classes=2)
                        y_gan = smooth_gan_labels(y_gan)

                        hist = self.discriminative_model_.fit(X, y_gan, batch_size=self.batch_size,
                                                              nb_epoch=1, verbose=0)

                        discriminator_loss = hist.history['loss'][-1]
                        discriminator_acc = hist.history['acc'][-1]

                        if save_loss:
                            loss_history['discriminator_loss'].extend(hist.history['loss'])
                            loss_history['discriminator_acc'].extend(hist.history['acc'])

                        if prev_improvement == -1:
                            prev_improvement = discriminator_loss

                        improvement = (prev_improvement - discriminator_loss) / prev_improvement * 100
                        prev_improvement = discriminator_loss

                        iteration += self.batch_size
                        t2 = time.time()

                        print("Iter : %d / %d | Improvement : %0.2f percent | Time required : %0.2f seconds | "
                              "Discriminator Loss / Acc : %0.4f / %0.2f" % (iteration, nb_images,
                                                                            improvement, t2 - t1,
                                                                            discriminator_loss, discriminator_acc))

                    else:
                        # Train only discriminator, disable training of srgan
                        self.discriminative_network.set_trainable(self.srgan_model_, value=True)
                        self.generative_network.set_trainable(self.srgan_model_, value=False)

                        # Use custom bypass_fit to bypass the check for same input and output batch size
                        # hist = bypass_fit(self.srgan_model_, [x_generator, x * 255, x_vgg],
                        #                   [y_gan, y_vgg_dummy],
                        #                   batch_size=self.batch_size, nb_epoch=1, verbose=0)

                        X_pred = self.generative_model_.predict(x_generator, self.batch_size)

                        X = np.concatenate((X_pred, x * 255))

                        # Using soft and noisy labels
                        if np.random.uniform() > disc_train_flip:
                            # give correct classifications
                            y_gan = [0] * self.batch_size + [1] * self.batch_size
                        else:
                            # give wrong classifications (noisy labels)
                            y_gan = [1] * self.batch_size + [0] * self.batch_size

                        y_gan = np.asarray(y_gan, dtype=np.int).reshape(-1, 1)
                        y_gan = to_categorical(y_gan, num_classes=2)
                        y_gan = smooth_gan_labels(y_gan)

                        hist1 = self.discriminative_model_.fit(X, y_gan, verbose=0, batch_size=self.batch_size,
                                                               nb_epoch=1)

                        discriminator_loss = hist1.history['loss'][-1]

                        # Train only generator, disable training of discriminator
                        self.discriminative_network.set_trainable(self.srgan_model_, value=False)
                        self.generative_network.set_trainable(self.srgan_model_, value=True)

                        # Using soft labels
                        y_model = [1] * self.batch_size
                        y_model = np.asarray(y_model, dtype=np.int).reshape(-1, 1)
                        y_model = to_categorical(y_model, num_classes=2)
                        y_model = smooth_gan_labels(y_model)

                        # Use custom bypass_fit to bypass the check for same input and output batch size
                        hist2 = bypass_fit(self.srgan_model_, [x_generator, x, x_vgg], [y_model, y_vgg_dummy],
                                           batch_size=self.batch_size, nb_epoch=1, verbose=0)

                        generative_loss = hist2.history['loss'][0]

                        if save_loss:
                            loss_history['discriminator_loss'].extend(hist1.history['loss'])
                            loss_history['discriminator_acc'].extend(hist1.history['acc'])
                            loss_history['generator_loss'].extend(hist2.history['loss'])

                        if prev_improvement == -1:
                            prev_improvement = discriminator_loss

                        improvement = (prev_improvement - discriminator_loss) / prev_improvement * 100
                        prev_improvement = discriminator_loss

                        iteration += self.batch_size
                        t2 = time.time()
                        print("Iter : %d / %d | Improvement : %0.2f percent | Time required : %0.2f seconds | "
                              "Discriminator Loss : %0.3f | Generative Loss : %0.3f" %
                              (iteration, nb_images, improvement, t2 - t1, discriminator_loss, generative_loss))

                    if iteration % 1000 == 0 and iteration != 0:
                        print("Saving model weights.")
                        # Save predictive (SR network) weights
                        self._save_model_weights(pre_train_srgan, pre_train_discriminator)
                        self._save_loss_history(loss_history, pre_train_srgan, pre_train_discriminator, save_loss)

                    if iteration >= nb_images:
                        break

                except KeyboardInterrupt:
                    print("Keyboard interrupt detected. Stopping early.")
                    early_stop = True
                    break

            iteration = 0

            if early_stop:
                break

        print("Finished training SRGAN network. Saving model weights.")
        # Save predictive (SR network) weights
        self._save_model_weights(pre_train_srgan, pre_train_discriminator)
        self._save_loss_history(loss_history, pre_train_srgan, pre_train_discriminator, save_loss)

    def _save_model_weights(self, pre_train_srgan, pre_train_discriminator):
        if not pre_train_discriminator:
            self.generative_model_.save_weights(self.generative_network.sr_weights_path, overwrite=True)

        if not pre_train_srgan:
            # Save GAN (discriminative network) weights
            self.discriminative_network.save_gan_weights(self.discriminative_model_)

    def _save_loss_history(self, loss_history, pre_train_srgan, pre_train_discriminator, save_loss):
        if save_loss:
            print("Saving loss history")

            if pre_train_srgan:
                with open('pretrain losses - srgan.json', 'w') as f:
                    json.dump(loss_history, f)
            elif pre_train_discriminator:
                with open('pretrain losses - discriminator.json', 'w') as f:
                    json.dump(loss_history, f)
            else:
                with open('fulltrain losses.json', 'w') as f:
                    json.dump(loss_history, f)

            print("Saved loss history")


if __name__ == "__main__":
    # Path to MS COCO dataset
    coco_path = "./Train/bsds300train"
    srgan_network = SRGANNetwork(img_width=32, img_height=32, batch_size=2)
    print("================ 1: {} ================".format(srgan_network))
    srgan_network.build_srgan_model()
    # plot(srgan_network.srgan_model_, 'SRGAN.png', show_shapes=True)

    # # Pretrain the SRGAN network
    # srgan_network.pre_train_srgan(coco_path, nb_images=1000, epochs=1)
    # # # #
    # # # # # Pretrain the discriminator network
    # srgan_network.pre_train_discriminator(coco_path, nb_images=1000, epochs=1, batch_size=16)

    # Fully train the SRGAN with VGG loss and Discriminator loss
    srgan_network.train_full_model(coco_path, nb_images=200, epochs=20)


