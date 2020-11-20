import numpy as np
import matplotlib.pyplot as plt
from hashlib import md5

import cv2
import os
import time
import scipy

IMAGE_DIR = '../../Finding-Similar-Items/Data/Object/test/'

os.chdir(IMAGE_DIR)

image_files = os.listdir()


def filter_images(images):
    image_list = []
    for image in images:
        try:
            assert (cv2.imread(image)).shape == 3
            image_list.append(image)
        except AssertionError as e:
            print(e)
    return image_list

# First turn the image into a gray scale image


def img_gray(image):
    image = cv2.imread(image)
    return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)

# resize image and flatten


def resize(image, height=30, width=30):
    row_res = resize(image, (height, width),
                     interpolation=cv2.INTER_AREA).flatten()
    col_res = resize(image, (height, width),
                     interpolation=cv2.INTER_AREA).flatten('F')
    return row_res, col_res

# gradient direction based on intensity


def intensity_diff(row_res, col_res):
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()


def difference_score(image, height=30, width=30):
    gray = img_gray(image)
    row_res, col_res = resize(gray, height, width)
    difference = intensity_diff(row_res, col_res)

    return difference


def difference_score_dict_hash(image_list):
    ds_dict = {}
    duplicates = []
    hash_ds = []
    for image in image_list:
        ds = difference_score(image)
        hash_ds.append(ds)
        filehash = md5(ds).hexdigest()  # min-hash
        if filehash not in ds_dict:
            ds_dict[filehash] = image
        else:
            duplicates.append((image, ds_dict[filehash]))

    return duplicates, ds_dict, hash_ds


image_files = filter_images(image_files)
duplicates, ds_dict, hash_ds = difference_score_dict_hash(image_files)

for file_names in duplicates[:30]:
    try:

        plt.subplot(121), plt.imshow(cv2.imread(file_names[0]))
        plt.title('Duplicate'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(cv2.imread(file_names[1]))
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.show()

    except OSError as e:
        continue
