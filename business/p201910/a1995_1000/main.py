# coding=utf-8
import SimpleITK as sitk

from radiomics.firstorder import RadiomicsFirstOrder
from radiomics.shape import RadiomicsShape
from radiomics.shape2D import RadiomicsShape2D
from radiomics.glrlm import RadiomicsGLRLM
from radiomics.ngtdm import RadiomicsNGTDM
from radiomics.gldm import RadiomicsGLDM
from radiomics.glcm import RadiomicsGLCM
from radiomics.glszm import RadiomicsGLSZM

import six

import pandas as pd
import os


def getFirstOrder(imageName, maskName):
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glcmFeatures = RadiomicsFirstOrder(image, mask)
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val
    return rows


def getShape(imageName, maskName):
    # 三维图像shape参数
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glcmFeatures = RadiomicsShape(image, mask)
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val
        rows['Compactness1FeatureValue'] = glcmFeatures.getCompactness1FeatureValue()
        rows['Compactness2FeatureValue'] = glcmFeatures.getCompactness2FeatureValue()

    return rows


def getShape2D(imageName, maskName):
    # 二维图像shape 参数
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glcmFeatures = RadiomicsShape2D(image, mask, force2D=True)
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val

    return rows


def getNGTDM(imageName, maskName):
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glrlmFeatures = RadiomicsNGTDM(image, mask, )
    glrlmFeatures.enableAllFeatures()
    results = glrlmFeatures.execute()

    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val

    return rows


def getGLRLM(imageName, maskName):
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glrlmFeatures = RadiomicsGLRLM(image, mask, )
    glrlmFeatures.enableAllFeatures()
    results = glrlmFeatures.execute()
    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val
    return rows


def getGLDM(imageName, maskName):
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glcmFeatures = RadiomicsGLDM(image, mask, )
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val

    return rows


def getGLCM(imageName, maskName):
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glcmFeatures = RadiomicsGLCM(image, mask, )
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val

    return rows


def getGLSZM(imageName, maskName):
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
    glcmFeatures = RadiomicsGLSZM(image, mask, )
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    rows = {}
    for (key, val) in six.iteritems(results):
        rows[key] = val

    return rows


def combine_dict(one, two, three, four, five, six, seven, imageName):
    one['file'] = imageName

    for k, v in two.items():
        if k not in one.keys():
            one[k] = v
        else:
            one["{}_2".format(k)] = v

    for k, v in three.items():
        if k not in one.keys():
            one[k] = v
        else:
            one["{}_3".format(k)] = v

    for k, v in four.items():
        if k not in one.keys():
            one[k] = v
        else:
            one["{}_4".format(k)] = v

    for k, v in five.items():
        if k not in one.keys():
            one[k] = v
        else:
            print(k, '4')
            one["{}_5".format(k)] = v

    for k, v in six.items():
        if k not in one.keys():
            one[k] = v
        else:
            one["{}_6".format(k)] = v

    for k, v in seven.items():
        if k not in one.keys():
            one[k] = v
        else:
            print(k, '6')
            one["{}_7".format(k)] = v

    return one


if __name__ == "__main__":

    step = 'good'
    image_path = 't2'
    mask_path = 't2l'
    imageNames = os.listdir('{}\\{}'.format(image_path, step))
    maskNames = os.listdir('{}\\{}'.format(mask_path, step))
    getFirstOrders = []
    for im, mk in zip(imageNames, maskNames):  # 循环
        one = getFirstOrder('{}\\{}\\{}'.format(image_path, step, im),
                            '{}\\{}\\{}'.format(mask_path, step, mk))

        two = getShape('{}\\{}\\{}'.format(image_path, step, im),
                       '{}\\{}\\{}'.format(mask_path, step, mk))

        three = getNGTDM('{}\\{}\\{}'.format(image_path, step, im),
                         '{}\\{}\\{}'.format(mask_path, step, mk))

        four = getGLRLM('{}\\{}\\{}'.format(image_path, step, im),
                        '{}\\{}\\{}'.format(mask_path, step, mk))

        five = getGLDM('{}\\{}\\{}'.format(image_path, step, im),
                       '{}\\{}\\{}'.format(mask_path, step, mk))

        six_value = getGLCM('{}\\{}\\{}'.format(image_path, step, im),
                            '{}\\{}\\{}'.format(mask_path, step, mk))

        seven = getGLSZM('{}\\{}\\{}'.format(image_path, step, im),
                         '{}\\{}\\{}'.format(mask_path, step, mk))

        res = combine_dict(one, two, three, four, five, six_value, seven, im)

        print("病人 {} 共计算{}个指标".format(im, len(res)-1))

        getFirstOrders.append(res)

    df = pd.DataFrame(getFirstOrders)
    df.to_excel("getFirstOrders mask T2.xlsx")
    print(df.shape)
