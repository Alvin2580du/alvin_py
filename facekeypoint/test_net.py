import sys
sys.path.append( '../tools' )
import FFDio
from FFDIter import FFDIter
import mxnet as mx
import numpy as np
import os
import cv2

if __name__ == '__main__':
    batch_size = 1
    
    # make validation data

    # image_list_indoor = FFDio.collect_data_set('/home/dms/mxnet_face/original_images/300W/01_Indoor')
    # image_list_outdoor = FFDio.collect_data_set('/home/dms/mxnet_face/original_images/300W/02_Outdoor')
    # image_list_val = image_list_indoor + image_list_outdoor
    image_list_val = FFDio.collect_data_set("/home/dms/alvin_data/face data/original_images/helen/trainset")
    # image_list_val = FFDio.collect_data_set("/home/dms/alvin_data/face_test")
    # make iterator
    iter_val = FFDIter(image_list_val, batch_size, False, 224, 224)
    prefix = './vgg16_result/vgg_16_reduced'
    # prefix = "./vgg16_result/inception_bn"
    model = mx.model.FeedForward.load(prefix, 50)

    data_batch = iter_val.next()
    data = data_batch.data
    label = data_batch.label
    print data[0].shape

    pred_loc = model.predict(data[0])

    for i in range(0, batch_size):
        image = data[0][i].asnumpy()
        pts = pred_loc[i]

        image = image.transpose((1, 2, 0))
        image += [128, 128, 128]
        image = image.astype('uint8')
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for j in range(0, 68):
            cv2.circle(image, (int(pts[j] * image.shape[1]), int(pts[j + 68] * image.shape[0])), 1, (80, 235, 210), 2)

        if i % 2 ==0:
            cv2.imwrite("./result_image/inception_bn_epoch_100_%d.jpg" %i, image)

