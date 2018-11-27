import numpy as np
import progressbar
from multiprocessing import Pool
from PIL import Image
from functools import partial
import os


def enum(**enums):
    return type('Enum', (), enums)


FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3),
                   FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
                FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


def ensemble_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0


def ensemble_vote_all(int_imgs, classifiers):
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: List of assigned labels, 1 if image was classified positively, else
    0
    :rtype: list[int]
    """
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


def reconstruct(classifiers, img_size):
    """
    Creates an image by putting all given classifiers on top of each other
    producing an archetype of the learned class of object.
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :param img_size: Tuple of width and height
    :type img_size: (int, int)
    :return: Reconstructed image
    :rtype: PIL.Image
    """
    image = np.zeros(img_size)
    for c in classifiers:
        polarity = pow(1 + c.polarity, 2) / 4
        if c.type == FeatureType.TWO_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if y >= c.height / 2:
                        sign = (sign + 1) % 2
                    image[c.top_left[1] + y, c.top_left[0] + x] += 1 * sign * c.weight
        elif c.type == FeatureType.TWO_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x >= c.width / 2:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x % c.width / 3 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if x % c.height / 3 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.FOUR:
            sign = polarity
            for x in range(c.width):
                if x % c.width / 2 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    if x % c.height / 2 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
    image -= image.min()
    image /= image.max()
    image *= 255
    result = Image.fromarray(image.astype(np.uint8))
    return result


def load_images(path):
    images = []
    for _file in os.listdir(path):
        if '.pgm'in _file:
            img_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            img_arr /= img_arr.max()
            images.append(img_arr)
        if '.jpg' in _file:
            img_arr = np.array(Image.open((os.path.join(path, _file))).convert('L').resize((64, 64)), dtype=np.float64)
            img_arr /= img_arr.max()
            images.append(img_arr)
    return images


def to_integral_image(img_arr):
    """
    Calculates the integral image based on this instance's original image data.
    :param img_arr: Image source data
    :type img_arr: numpy.ndarray
    :return Integral image for given image
    :rtype: numpy.ndarray
    """
    row_sum = np.zeros(img_arr.shape)
    integral_image_arr = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
    for x in range(img_arr.shape[1]):
        for y in range(img_arr.shape[0]):
            row_sum[y, x] = row_sum[y - 1, x] + img_arr[y, x]
            integral_image_arr[y + 1, x + 1] = integral_image_arr[y + 1, x - 1 + 1] + row_sum[y, x]
    return integral_image_arr


def sum_region(integral_img_arr, top_left, bottom_right):
    """
    Calculates the sum in the rectangle specified by the given tuples.
    :param integral_img_arr:
    :type integral_img_arr: numpy.ndarray
    :param top_left: (x, y) of the rectangle's top left corner
    :type top_left: (int, int)
    :param bottom_right: (x, y) of the rectangle's bottom right corner
    :type bottom_right: (int, int)
    :return The sum of all pixels in the given rectangle
    :rtype int
    """
    # swap tuples
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    if top_left == bottom_right:
        return integral_img_arr[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return integral_img_arr[bottom_right] - integral_img_arr[top_right] - integral_img_arr[bottom_left] + integral_img_arr[top_left]


class HaarLikeFeature(object):
    """
    Class representing a haar-like feature.
    """
    def __init__(self, feature_type, position, width, height, threshold, polarity):
        """
        Creates a new haar-like feature.
        :param feature_type: Type of new feature, see FeatureType enum
        :type feature_type: violajonse.HaarLikeFeature.FeatureTypes
        :param position: Top left corner where the feature begins (x, y)
        :type position: (int, int)
        :param width: Width of the feature
        :type width: int
        :param height: Height of the feature
        :type height: int
        :param threshold: Feature threshold
        :type threshold: float
        :param polarity: polarity of the feature -1 or 1
        :type polarity: int
        """
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = 1

    def get_score(self, int_img):
        """
        Get score for given integral image array.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: Score for given feature
        :rtype: float
        """
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = sum_region(int_img, self.top_left,
                               (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                                self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = sum_region(int_img, self.top_left,
                               (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = sum_region(int_img, self.top_left,
                               (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = sum_region(int_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]),
                                (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = sum_region(int_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]),
                               self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = sum_region(int_img, self.top_left,
                               (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            third = sum_region(int_img, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)),
                               self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = sum_region(int_img, self.top_left,
                               (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            # top right area
            second = sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            # bottom left area
            third = sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                               (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            # bottom right area
            fourth = sum_region(int_img,
                                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)),
                                self.bottom_right)
            score = first - second - third + fourth
        return score

    def get_vote(self, int_img):
        """
        Get vote of this feature for given integral image.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: 1 iff this feature votes positively, otherwise -1
        :rtype: int
        """
        score = self.get_score(int_img)
        return self.weight * (1 if score < self.polarity * self.threshold else -1)


def learn(positive_iis, negative_iis, num_classifiers=-1, min_feature_width=1, max_feature_width=-1,
          min_feature_height=1, max_feature_height=-1):
    """
    Selects a set of classifiers. Iteratively takes the best classifiers based
    on a weighted error.
    :param min_feature_height: 
    :param positive_iis: List of positive integral image examples
    :type positive_iis: list[numpy.ndarray]
    :param negative_iis: List of negative integral image examples
    :type negative_iis: list[numpy.ndarray]
    :param num_classifiers: Number of classifiers to select, -1 will use all
    classifiers
    :type num_classifiers: int

    :return: List of selected features
    :rtype: list[violajones.HaarLikeFeature.HaarLikeFeature]
    """
    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg
    img_height, img_width = positive_iis[0].shape

    # Maximum feature width and height default to image width and height
    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if max_feature_width == -1 else max_feature_width

    # Create initial weights and labels
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))

    images = positive_iis + negative_iis

    # Create features for all sizes and locations
    features = _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                                max_feature_height)
    num_features = len(features)
    feature_indexes = list(range(num_features))

    num_classifiers = num_features if num_classifiers == -1 else num_classifiers

    print('Calculating scores for images..')

    votes = np.zeros((num_imgs, num_features))
    bar = progressbar.ProgressBar()
    # Use as many workers as there are CPUs
    pool = Pool(processes=None)
    for i in bar(range(num_imgs)):
        votes[i, :] = np.array(list(pool.map(partial(_get_feature_vote, image=images[i]), features)))
    # select classifiers
    classifiers = []

    print('Selecting classifiers..')
    bar = progressbar.ProgressBar()
    for _ in bar(range(num_classifiers)):
        classification_errors = np.zeros(len(feature_indexes))
        # normalize weights
        weights *= 1. / np.sum(weights)

        # select best classifier based on the weighted error
        for f in range(len(feature_indexes)):
            f_idx = feature_indexes[f]
            # classifier error is the sum of image weights where the classifier
            # is right
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            classification_errors[f] = error

        # get best feature, i.e. with smallest error
        min_error_idx = np.argmin(classification_errors)
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_indexes[min_error_idx]

        # set feature weight
        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        best_feature.weight = feature_weight

        classifiers.append(best_feature)

        # update image weights
        weights = np.array(list(map(
            lambda img_idx: weights[img_idx] * np.sqrt((1 - best_error) / best_error) if labels[img_idx] != votes[
                img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error / (1 - best_error)), range(num_imgs))))

        # remove feature (a feature can't be selected twice)
        feature_indexes.remove(best_feature_idx)

    return classifiers


def _get_feature_vote(feature, image):
    return feature.get_vote(image)


def _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
    print('Creating haar-like features..')
    features = []
    for feature in FeatureTypes:
        # FeatureTypes are just tuples
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
            feature_start_height = max(min_feature_height, feature[1])
            for feature_height in range(feature_start_height, max_feature_height, feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, 1))
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, -1))
    print('..done. ' + str(len(features)) + ' features created.\n')
    return features


if __name__ == "__main__":
    method = 'test_haarlikefeature'
    if method == 'test_haarlikefeature':
        test_image_name = './lfw1000/Aaron_Eckhart_0001.pgm'
        img_arr = np.array(Image.open(test_image_name), dtype=np.float64)
        int_img = to_integral_image(img_arr)
        feature = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 24, 24, 100000, 1)
        left_area = sum_region(int_img, (0, 0), (24, 12))
        right_area = sum_region(int_img, (0, 12), (24, 24))
        expected = 1 if feature.threshold * feature.polarity > left_area - right_area else 0
        assert feature.get_vote(int_img) == expected

        feature = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 24, 24, 100000, 1)
        left_area = sum_region(int_img, (0, 0), (24, 12))
        right_area = sum_region(int_img, (0, 12), (24, 24))
        expected = 1 if feature.threshold * -1 > left_area - right_area else 0
        assert feature.get_vote(int_img) != expected

        feature = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 24, 24, 100000, 1)
        left_area = sum_region(int_img, (0, 0), (24, 12))
        right_area = sum_region(int_img, (0, 12), (24, 24))
        expected = 1 if feature.threshold * feature.polarity > left_area - right_area else 0
        assert feature.get_vote(int_img) == expected

        feature = HaarLikeFeature(FeatureType.THREE_HORIZONTAL, (0, 0), 24, 24, 100000, 1)
        left_area = sum_region(int_img, (0, 0), (8, 24))
        middle_area = sum_region(int_img, (8, 0), (16, 24))
        right_area = sum_region(int_img, (16, 0), (24, 24))
        expected = 1 if feature.threshold * feature.polarity > left_area - middle_area + right_area else 0
        assert feature.get_vote(int_img) == expected

        feature = HaarLikeFeature(FeatureType.THREE_VERTICAL, (0, 0), 24, 24, 100000, 1)
        left_area = sum_region(int_img, (0, 0), (24, 8))
        middle_area = sum_region(int_img, (0, 8), (24, 16))
        right_area = sum_region(int_img, (0, 16), (24, 24))
        expected = 1 if feature.threshold * feature.polarity > left_area - middle_area + right_area else 0
        assert feature.get_vote(int_img) == expected

        feature = HaarLikeFeature(FeatureType.THREE_HORIZONTAL, (0, 0), 24, 24, 100000, 1)
        top_left_area = sum_region(int_img, (0, 0), (12, 12))
        top_right_area = sum_region(int_img, (12, 0), (24, 12))
        bottom_left_area = sum_region(int_img, (0, 12), (12, 24))
        bottom_right_area = sum_region(int_img, (12, 12), (24, 24))
        expected = 1 if feature.threshold * feature.polarity > top_left_area - top_right_area - bottom_left_area + bottom_right_area else 0
        assert feature.get_vote(int_img) == expected

    if method == 'adaboost':
        pos_training_path = './lfw1000'
        neg_training_path = './nonface'
        pos_testing_path = './lfw1000_test'
        neg_testing_path = './nonface_test'

        num_classifiers = 2
        # For performance reasons restricting feature size
        min_feature_height = 8
        max_feature_height = 10
        min_feature_width = 8
        max_feature_width = 10

        print('Loading faces..')
        faces_training = load_images(pos_training_path)
        faces_ii_training = list(map(to_integral_image, faces_training))
        print('..done. ' + str(len(faces_training)) + ' faces loaded.\n\nLoading non faces..')
        non_faces_training = load_images(neg_training_path)
        print("non_faces_training:{}".format(len(non_faces_training)), )
        non_faces_ii_training = list(map(to_integral_image, non_faces_training))
        print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')

        # classifiers are haar like features
        classifiers = learn(faces_ii_training, non_faces_ii_training, num_classifiers, min_feature_height,
                            max_feature_height, min_feature_width, max_feature_width)

        print('Loading test faces..')
        faces_testing = load_images(pos_testing_path)
        faces_ii_testing = list(map(to_integral_image, faces_testing))
        print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
        non_faces_testing = load_images(neg_testing_path)
        non_faces_ii_testing = list(map(to_integral_image, non_faces_testing))
        print("non_faces_ii_testing:{}".format(non_faces_ii_testing))
        print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

        print('Testing selected classifiers..')
        correct_faces = 0
        correct_non_faces = 0
        correct_faces = sum(ensemble_vote_all(faces_ii_testing, classifiers))
        correct_non_faces = len(non_faces_testing) - sum(ensemble_vote_all(non_faces_ii_testing, classifiers))

        print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
              + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
              + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
              + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')

        recon = reconstruct(classifiers, faces_testing[0].shape)
        recon.save('reconstruction.png')