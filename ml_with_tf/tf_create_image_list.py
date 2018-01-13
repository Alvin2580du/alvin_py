import hashlib
import os.path
import re

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage, save_dir=None):

    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    # The root directory comes first, so skip it.
    is_root_dir = True
    dir_names = []
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        dir_names.append(dir_name)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(hash_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(hash_name)
            else:
                training_images.append(hash_name)

        result[label_name] = {
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }

        data_sets = ['training', 'testing', 'validation']

        for i in dir_names:
            if save_dir == None:
                class_data_path = os.path.join(os.getcwd(), "{}".format(i))
            else:
                class_data_path = os.path.join(save_dir, "{}".format(i))
            if not os.path.exists(class_data_path):
                os.makedirs(class_data_path)

            for dataset in data_sets:
                for x in result['{}'.format(i)]['{}'.format(dataset)]:
                    file_saved_name = os.path.join(class_data_path, "{}".format(dataset))
                    file = open(file_saved_name, 'a+')
    return result


def create_unsupervised_image_listimage_lists(image_dir, testing_percentage, validation_percentage, save_dir=None):
    result = {"training": [], "testing": [], "validation": []}
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(gfile.Glob(file_glob))

    training_images = []
    testing_images = []
    validation_images = []

    for file_name in file_list:
        hash_name = re.sub(r'_nohash_.*$', '', file_name)
        hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                           (100.0 / MAX_NUM_IMAGES_PER_CLASS))
        if percentage_hash < validation_percentage:
            validation_images.append(hash_name)
        elif percentage_hash < (testing_percentage + validation_percentage):
            testing_images.append(hash_name)
        else:
            training_images.append(hash_name)

    result['training'] = training_images
    result['testing'] = testing_images
    result['validation'] = validation_images

    datasets = ['training', 'testing', 'validation']
    for dataset in datasets:
        if save_dir == None:
            class_data_path = os.path.join(os.getcwd())
        else:
            class_data_path = save_dir

        for x in result['{}'.format(dataset)]:
            file_saved_name = os.path.join(class_data_path, "{}".format(dataset))
            file = open(file_saved_name, 'a+')
    return result



