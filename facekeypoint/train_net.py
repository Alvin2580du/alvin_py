import argparse
import FFDio
from FFDio import collect_data_set
from FFDIter import FFDIter
import numpy as np
import config
import mxnet as mx
# from inception_bn import get_symbol
from vgg_16_reduced import getsymbol
from vgg_16_tensorflow import axelnet


def train_w_predetermined_data(args):
    dataiter_train = mx.io.ImageRecordIter(
        path_imgrec="../processed_data/training_data.rec",
        data_shape=(3, 224, 224),
        path_imglist="../processed_data/training_data.lst",
        label_width=136,
        mean_r=128,
        mean_g=128,
        mean_b=128,
        batch_size=args.batch_size,
        label_name='label'
    )

    dataiter_test = mx.io.ImageRecordIter(
        path_imgrec="../processed_data/test_data.rec",
        data_shape=(3, 224, 224),
        path_imglist="../processed_data/test_data.lst",
        label_width=136,
        mean_r=128,
        mean_g=128,
        mean_b=128,
        batch_size=args.batch_size,
        label_name='label'
    )

    network = getsymbol()
    model = mx.model.FeedForward(
        ctx=config._get_devs(**kwargs),
        symbol=network,
        num_epoch=config._get_num_epoch(**kwargs),
        optimizer=mx.optimizer.Adam(learning_rate=1e-4),
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=0.05))

    model.fit(
        X=dataiter_train,
        eval_data=dataiter_test,
        eval_metric=[nrmse, 'rmse'],
        batch_end_callback=mx.callback.Speedometer(args.batch_size, 1),
        epoch_end_callback=mx.callback.do_checkpoint("../inceptionbn_result" + args.save_prefix),)

    return model


def train_w_realtime_data(model):
    # make train data
    image_list_AFW = collect_data_set('/home/dms/alvin_data/face data/original_images/afw')
    image_list_IBUG = collect_data_set('/home/dms/alvin_data/face data/original_images/ibug')
    image_list_HELEN = collect_data_set('/home/dms/alvin_data/face data/original_images/helen/trainset')
    image_list_LFPW = collect_data_set('/home/dms/alvin_data/face data/original_images/lfpw/trainset')
    image_list_train = image_list_AFW + image_list_HELEN + image_list_LFPW + image_list_IBUG

    # make validation data
    image_list_indoor = FFDio.collect_data_set('/home/dms/alvin_data/face data/original_images/300W/01_Indoor')
    image_list_outdoor = FFDio.collect_data_set('/home/dms/alvin_data/face data/original_images/300W/02_Outdoor')
    image_list_val = image_list_indoor + image_list_outdoor

    iter_val = mx.io.ImageRecordIter(
        path_imgrec="../processed_data/test_data.rec",
        data_shape=(3, 224, 224),
        path_imglist="../processed_data/test_data.lst",
        label_width=136,
        mean_r=128,
        mean_g=128,
        mean_b=128,
        batch_size=args.batch_size,
        label_name='label'
    )

    # make iterator
    iter_train = FFDIter(image_list_train, args.batch_size, True, 224, 224)
    # iter_val = FFDIter(image_list_val, args.batch_size, False, 224, 224)

    model.fit(
        X=iter_train,  # training data
        eval_data=iter_val,  # validation data
        eval_metric=nrmse,  # 'rmse',
        # output progress for each 200 data batches
        batch_end_callback=mx.callback.Speedometer(args.batch_size, 1),
        epoch_end_callback=mx.callback.do_checkpoint(args.save_prefix + '_finetuned'),
    )


def nrmse(label, pred):
    batch_size = label.shape[0]

    total_sum = 0
    for i in range(0, batch_size):
        gt = label[i, :]
        gt = np.reshape(gt, (2, 68))

        pts = pred[i, :]
        pts = np.reshape(pts, (2, 68))
        iod = np.linalg.norm(gt[:, 36] - gt[:, 45])

        sum = 0
        for i in range(0, 68):
            sum += np.linalg.norm(gt[:, i] - pts[:, i])
        rmse_68 = sum / (68 * iod)
        total_sum += rmse_68

    return total_sum / batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MXNet Trainer')
    # parser.add_argument('--save-prefix', type=str, required=False, default="vgg_16_reduced")

    parser.add_argument('--save-prefix', type=str, required=False, default="inception_bn1")

    parser.add_argument('--batch-size', type=int, required=False, default=16)
    parser.add_argument('--gpus', type=str, required=False, default="0")
    parser.add_argument('--num-epoch', type=int, required=False, default=50)

    args = parser.parse_args()
    print("args is ", args)
    kwargs = args.__dict__
    print("kwargs is ", kwargs)
    config._set_logger(**kwargs)

    model = train_w_predetermined_data(args)
    # finetuned
    # model = train_w_realtime_data(model)
