import mxnet as mx
import mxnet.symbol as mxy


def getsymbol(num_classes=136):
    # define alexnet
    data = mxy.Variable(name="data")
    label = mxy.Variable(name="label")

    # group 1
    conv1_1 = mxy.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mxy.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mxy.Pooling(data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")

    # group 2
    conv2_1 = mxy.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mxy.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mxy.Pooling(data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")

    # group 3
    conv3_1 = mxy.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mxy.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mxy.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mxy.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mxy.Pooling(data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")

    # group 4
    conv4_1 = mxy.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mxy.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mxy.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mxy.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mxy.Pooling(data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")

    # group 5
    conv5_1 = mxy.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mxy.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mxy.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mxy.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mxy.Pooling(data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")

    # group 6
    flatten = mxy.Flatten(data=pool5, name="flatten")
    fc6 = mxy.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mxy.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mxy.Dropout(data=relu6, p=0.5, name="drop6")

    # group 7
    fc7 = mxy.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mxy.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mxy.Dropout(data=relu7, p=0.5, name="drop7")

    # output
    fc8 = mxy.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    loc_loss = mxy.LinearRegressionOutput(data=fc8, label=label, name="loc_loss")

    #loc_loss_ = mxy.smooth_l1(name="loc_loss_", data=(fc8 - label), scalar=1.0)
    #loc_loss_ = mxy.smooth_l1(name="loc_loss_", data=fc8, scalar=1.0)
    #loc_loss = mx.sym.MakeLoss(name='loc_loss', data=loc_loss_)

    return loc_loss

