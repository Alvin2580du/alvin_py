from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, ZeroPadding2D, Concatenate
from keras.layers import LeakyReLU, PReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.models import Model
from keras import backend as K

import datetime
import os
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt


def build_data():
    # 分割原始数据，分为2个文件夹
    # data_path = 'F:\\QQFiles\\562078401\\FileRecv\\ID-CGAN-20190804T114213Z-001\\ID-CGAN\\dataset\\rain\\training'
    test_path = 'F:\\QQFiles\\562078401\\FileRecv\\ID-CGAN-20190804T114213Z-001\\ID-CGAN\\dataset\\rain\\test_syn'
    # 训练集和测试集的目录

    path = glob("{}//*".format(test_path))   # 遍历所有图像
    img_res = (256, 256)  # resize图像大小到 256 * 256
    state = 'testing'  # training
    save_path = 'datasets\\rain\\{}'.format(state)

    for img_path in path:
        file_name = img_path.split("\\")[-1]
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        _w = int(w / 2)  # 高不变，长一分为二
        img_A = cv2.resize(img[:, :_w, :], img_res)
        img_B = cv2.resize(img[:, _w:, :], img_res)
        cv2.imwrite("{}\\ground truth\\{}".format(save_path, file_name), img_A)  # 保存到对应的文件夹
        cv2.imwrite("{}\\rainy image\\{}".format(save_path, file_name), img_B)


class DataLoader():
    # 数据读取函数
    def __init__(self, dataset_name="rain", img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        # 读取ground truth和对应的rainy image
        data_A_path = 'mytrainData/ground truth'
        path_B_path = 'mytrainData/rainy image'
        path_A = os.listdir(data_A_path)
        batch_images = np.random.choice(path_A, size=batch_size)

        imgs_A = []
        imgs_B = []
        for ba in batch_images:
            # 读取ground truth
            img_A = cv2.imread("{}/{}".format(data_A_path, ba))
            img_A = cv2.resize(img_A, self.img_res)
            if not is_testing and np.random.random() > 0.5:
                img_A = np.fliplr(img_A)
            imgs_A.append(img_A)

        for bb in batch_images:
            # 读取rainy image
            imgB = "{}/{}".format(path_B_path, bb)
            img_B = cv2.imread(imgB)

            img_B = cv2.resize(img_B, self.img_res)
            if not is_testing and np.random.random() > 0.5:
                img_B = np.fliplr(img_B)
            imgs_B.append(img_B)

        imgs_A_out = np.array(imgs_A) / 127.5 - 1.
        imgs_B_out = np.array(imgs_B) / 127.5 - 1.

        return imgs_A_out, imgs_B_out

    # imgs_A 是 groudtruth ; imgs_B 是有下雨效果的图
    def load_batch(self, batch_size=1, is_testing=False):
        # 每次读取一个batch的数据，方法和上面一样，只是这里会返回一个迭代器
        data_A_path = 'mytrainData/ground truth/'
        path_B_path = 'mytrainData/rainy image/'
        path_A = os.listdir(data_A_path)

        self.n_batches = int(len(path_A) / batch_size)

        for i in range(self.n_batches - 1):
            batchA = path_A[i * batch_size:(i + 1) * batch_size]
            image_B_names = []
            imgs_A = []
            for ba in batchA:
                image_B_names.append(ba)
                img_A = cv2.imread("{}{}".format(data_A_path, ba))
                img_A = cv2.resize(img_A, self.img_res)
                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                imgs_A.append(img_A)

            imgs_B = []

            for bb in image_B_names:
                imgB = "{}{}".format(path_B_path, bb)
                img_B = cv2.imread(imgB)
                img_B = cv2.resize(img_B, self.img_res)
                if not is_testing and np.random.random() > 0.5:
                    img_B = np.fliplr(img_B)
                imgs_B.append(img_B)

            imgs_A_out = np.array(imgs_A) / 127.5 - 1.
            imgs_B_out = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A_out, imgs_B_out


class IDCGAN():
    # IDCGAN模型搭建
    def __init__(self):

        self.img_rows = 256  # resize 之后的图像大小
        self.img_cols = 256
        self.channels = 3  # 图像通道数

        self.img_shape = (self.img_rows, self.img_cols, self.channels)  # 图像大小

        self.dataset_name = 'rain'  # 数据集的名字
        # 实例化一个DataLoader对象，进行数据读取
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
        # 判别器输出的图像大小
        self.disc_out = (14, 14, 72)

        self.discriminator = self.build_discriminator()  # 建立判别器模型
        self.generator = self.build_generator()  # 建立生成器模型
        self.CGAN_model = self.build_CGAN()  # 建立CGAN模型

        self.optimizer_cgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # 用Adam优化器进行cgan模型训练损失函数，和论文一致
        self.optimizer_discriminator = SGD(lr=1E-3, momentum=0.9, decay=1e-6, nesterov=False)  # 判别器的优化器是SGD

    def build_CGAN(self):
        # 生成器训练的时候， 判别器停止训练
        self.discriminator.trainable = False
        img_B = Input(shape=self.img_shape)
        fake_A = self.generator(img_B)  # 生成器生成一个假的图像
        discriminator_output = self.discriminator([fake_A, img_B])
        CGAN_model = Model(inputs=[img_B], outputs=[fake_A, fake_A, discriminator_output], name='CGAN')
        # outputs 包含3部分，对应3部分损失
        return CGAN_model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            # 判别器的网络层，有4个这样的网络层组合起来
            x = Conv2D(filters, kernel_size=f_size, strides=1)(layer_input)  # Conv2D卷积
            x = LeakyReLU()(x)  # 激活函数
            if bn:
                x = BatchNormalization(momentum=0.8)(x)  # 正则化
            x = MaxPooling2D()(x)  # 池化层
            return x

        def Deconv2d(layer_input, filters, kernel=4, dropout_rate=0):
            # 反卷积层
            x = UpSampling2D(size=2)(layer_input)
            x = Conv2D(filters, kernel_size=kernel, strides=1, padding='same', activation='relu')(x)
            if dropout_rate:
                x = Dropout(dropout_rate)(x)
            x = BatchNormalization(momentum=0.8)(x)
            return x

        # 空间金字塔池化
        def Pyramid_Pool(layer_input):
            x_list = [layer_input]

            def Pool(size):
                x = MaxPooling2D(pool_size=(size * 2, size * 2))(layer_input)  # 二维最大值池化
                for i in range(size):
                    x = Deconv2d(x, 2)  # 反卷积
                return x

            # 金子塔的第一层
            x_list.append(Pool(1))
            x2 = MaxPooling2D(pool_size=(4, 4))(layer_input)  # 金字塔的第二层
            x2 = Deconv2d(x2, 2)
            x2 = Deconv2d(x2, 2)
            x2 = ZeroPadding2D(padding=(1, 1))(x2)
            x_list.append(x2)
            x3 = MaxPooling2D(pool_size=(8, 8))(layer_input)  # 金字塔的最后一层
            x3 = Deconv2d(x3, 4)
            x3 = Deconv2d(x3, 4)
            x3 = Deconv2d(x3, 4)
            x3 = ZeroPadding2D(padding=(3, 3))(x3)
            x_list.append(x3)
            x = Concatenate(axis=-1)(x_list)
            return x

        img_A = Input(shape=self.img_shape)  # 输入的ground truth
        img_B = Input(shape=self.img_shape)  # 输入有雨的图像
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])  # 组合起来
        x0 = d_layer(combined_imgs, 64, 3)  # 第一层
        x1 = d_layer(x0, 256, 3)
        x2 = d_layer(x1, 512, 3)
        x3 = d_layer(x2, 64, 3)
        x4 = Pyramid_Pool(x3)
        out = Activation('sigmoid')(x4)  # 判别器的输出是72通道
        return Model([img_A, img_B], out)

    def build_generator(self):
        # 建立生成器的卷积层
        def Conv2d(layer_input, no_filters, kernel, stride, bn=False, padding='valid'):
            x = Conv2D(filters=no_filters, kernel_size=kernel, strides=stride, padding=padding)(layer_input)
            x = BatchNormalization(momentum=0.8)(x)
            x = Activation('relu')(x)
            return x

        # Dense 块使用Dense net 模型的残差链接方法
        def dense_block(layer_input, num_layers):
            x_list = [layer_input]
            for i in range(num_layers):
                x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(layer_input)
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                x_list.append(x)
                x = Concatenate(axis=-1)(x_list)  # 把所有的残差链接层组合起来
            return x

        # 上采样模块，包含上采样，卷积，BN，droupout
        def Deconv2d(layer_input, filters, kernel=4, dropout_rate=0):
            x = UpSampling2D(size=2)(layer_input)
            x = Conv2D(filters, kernel_size=kernel, strides=1, padding='same', activation='relu')(x)
            if dropout_rate:
                x = Dropout(dropout_rate)(x)
            x = BatchNormalization(momentum=0.8)(x)
            return x

        inp = Input(shape=self.img_shape)

        # DownSampling 下采样
        x0 = Conv2d(inp, 64, (3, 3), (1, 1), bn=True)
        x0 = MaxPooling2D()(x0)
        x1 = dense_block(x0, 4)
        x1 = Conv2d(x1, 128, (3, 3), (2, 2), bn=True)
        x2 = dense_block(x1, 6)
        x2 = Conv2d(x2, 256, (3, 3), (2, 2), bn=True)
        x3 = dense_block(x2, 8)
        x3 = Conv2d(x3, 512, (3, 3), (1, 1), bn=True, padding='same')
        x4 = dense_block(x3, 8)
        x4 = Conv2d(x4, 128, (3, 3), (1, 1), bn=True, padding='same')

        # UpSampling 上采样
        x5 = dense_block(x4, 6)
        x5 = Deconv2d(x5, 120)
        x6 = dense_block(x5, 4)
        x6 = Deconv2d(x6, 64)
        x7 = dense_block(x6, 4)
        x7 = Deconv2d(x7, 64)
        x8 = dense_block(x7, 4)
        x8 = Conv2d(x8, 16, (3, 3), (1, 1), bn=True, padding='same')
        x9 = ZeroPadding2D(padding=(5, 5))(x8)
        x10 = Conv2D(filters=3, kernel_size=(3, 3))(x9)
        out = Activation('tanh')(x10)

        return Model(inp, out)  # 生成器模型的输出

    def train(self, epochs, batch_size=5, sample_interval=25):
        # 论文里面提到的Perceptual Loss 使用预训练的VGG16模型
        def perceptual_loss(img_true, img_generated):
            # 定义perceptual_loss
            image_shape = self.img_shape
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
            loss_block3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
            loss_block3.trainable = False
            loss_block2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
            loss_block2.trainable = False
            loss_block1 = Model(input=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
            loss_block1.trainable = False
            return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2 * K.mean(
                K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5 * K.mean(
                K.square(loss_block3(img_true) - loss_block3(img_generated)))

        self.discriminator.trainable = False  # 生成器训练的时候，判别器停止训练
        self.generator.compile(loss=perceptual_loss, optimizer=self.optimizer_cgan)  # 编译生成器
        # print(self.generator.summary())

        CGAN_loss = ['mse', perceptual_loss, 'mse']  # CGAN模型的三种损失
        CGAN_loss_weights = [6.6e-3, 1, 6.6e-3]
        # 编译CGAN 模型
        self.CGAN_model.compile(loss=CGAN_loss, loss_weights=CGAN_loss_weights, optimizer=self.optimizer_cgan)
        # print(self.CGAN_model.summary())
        # 开始训练判别器模型
        self.discriminator.trainable = True
        self.discriminator.compile(loss="mse", optimizer=self.optimizer_discriminator)  # 判别器的损失函数是mse
        # print(self.discriminator.summary())

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_out)  # 这个是真实的有雨的图像
        fake = np.zeros((batch_size,) + self.disc_out)  # 这个是生成的有雨的图像

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                fake_A = self.generator.predict(imgs_B)  # 生成器生成的假的图像
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # 判别器的损失
                self.CGAN_model.trainable = True  # 训练CGAN模型
                self.discriminator.trainable = False
                g_loss = self.CGAN_model.train_on_batch(imgs_B, [imgs_A, imgs_A, valid])
                elapsed_time = datetime.datetime.now() - start_time

                if batch_i % sample_interval == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                              batch_i,
                                                                                              self.data_loader.n_batches,
                                                                                              d_loss,
                                                                                              g_loss[0],
                                                                                              elapsed_time))
                    # 每隔sample_interval保存一次生成的图像
                    self.sample_images(epoch, batch_i)

            if epoch % 50 == 1 or epoch == epochs-1:
                self.CGAN_model.save_weights("./results/saved_models/com_model_{}.h5".format(epoch))
                self.generator.save_weights("./results/saved_models/gen_model_{}.h5".format(epoch))
                self.discriminator.save_weights("./results/saved_models/dis_model_{}.h5".format(epoch))
                print("Model 保存成功")

    def sample_images(self, epoch, batch_i):
        # 保存采样图像
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)  # 创建图像保存文件夹
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        # 归一化图像到0-1之间
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['WithRain', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))  # 保存图像
        plt.close()


if __name__ == '__main__':

    method = 'train'

    if method == 'build_data':
        # 第一步先创建训练集和测试集，只需要修改原始数据目录就行
        build_data()

    if method == 'train':
        # 存量CGAN模型，batch为1
        gan = IDCGAN()
        gan.train(epochs=200, batch_size=1, sample_interval=50)

    if method == 'testone':
        # 测试任意一张输入图像
        epoch = 149
        img_rows = 256
        img_cols = 256
        dataset_name = 'rain'
        gan = IDCGAN()
        # 使用训练保存的模型进行预测
        generator = gan.build_generator()
        generator.load_weights("results/saved_models/gen_model_{}.h5".format(epoch))
        data_loader = DataLoader(dataset_name=dataset_name, img_res=(img_rows, img_cols))
        imgs_A, imgs_B = data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = generator.predict(imgs_B)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        gen_imgs = 0.5 * gen_imgs + 0.5
        r, c = 3, 3

        titles = ['WithRain', 'Generated', 'Original']

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):

                plot_image = gen_imgs[cnt]

                axs[i, j].imshow(plot_image)
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("results/test_images/p_{}_{}.png".format(dataset_name, epoch))
        plt.close()
        print('save success')
        # 保存成功
