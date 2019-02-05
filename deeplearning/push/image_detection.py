# encoding:utf-8
import tensorflow as tf
from matplotlib import pyplot as plt
from nets import inception
from preprocessing import inception_preprocessing
import numpy as np
from datasets import imagenet

# 获取inception_resnet_v2默认图片尺寸，这里为299
image_size = inception.inception_resnet_v2.default_image_size
# 获取imagenet所有分类的名字，这里有1000个分类
names = imagenet.create_readable_names_for_imagenet_labels()

slim = tf.contrib.slim

# 待测试图片路径
sample_image = 'bus.png'

# 打开原图
image = tf.image.decode_jpeg(tf.read_file(sample_image), channels=3)
# 对原图进行裁剪、缩放、归一化等处理，将图片大小缩放至299×299
processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
# 增加一个维度
processed_images = tf.expand_dims(processed_image, 0)

# 创建模型
arg_scope = inception.inception_resnet_v2_arg_scope()
with slim.arg_scope(arg_scope):
    logits, end_points = inception.inception_resnet_v2(processed_images, is_training=False)

with tf.Session() as sess:
    # 这里是我们下载下来的模型的路径
    checkpoint_file = 'checkpoint/inception_resnet_v2_2016_08_30.ckpt'
    # 加载已训练好的模型
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    # 通过softmax获取分类
    probabilities = tf.nn.softmax(logits)

    srcimage, predict_values, logit_values = sess.run([image, processed_images, probabilities])

    print(np.max(logit_values))
    print(np.argmax(logit_values), names[np.argmax(logit_values)])

    plt.figure()
    p1 = plt.subplot(121)
    p2 = plt.subplot(122)

    # 显示原始图片
    p1.imshow(srcimage)
    p1.axis('off')
    p1.set_title('source image')

    # 显示预处理后的图片
    p2.imshow(predict_values[0, :, :, :])
    p2.axis('off')
    p2.set_title('image')

    plt.show()
