from easydict import EasyDict as edict
import json

train = edict()

# Adam
train.batch_size = 2
train.lr_init = 1e-4
train.beta1 = 0.9

# initialize G
train.n_epoch_init = 100
# train.lr_decay_init = 0.1
# train.decay_every_init = int(train.n_epoch_init / 2)

# adversarial learning (SRGAN)
train.n_epoch = 100
train.lr_decay = 0.1
train.decay_every = int(train.n_epoch / 2)

# train set location
train.hr_img_path = 'D:\\SRGAN_master\\asset\\data\\yaogan\\train\\'
train.lr_img_path = 'D:\\SRGAN_master\\asset\\data\\yaogan\\train_x4\\'

valid = edict()
# test set location
valid.hr_img_path = 'D:\\SRGAN_master\\asset\\data\\yaogan\\test\\'
valid.lr_img_path = 'D:\\SRGAN_master\\dasset\\data\\yaogan\\test_x4\\'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
