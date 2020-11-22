import itertools
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from hashlib import md5
from sklearn.metrics import jaccard_score

IMAGE_DIR = 'data/'
SIZE = 128
SIMILARITY = 0.6

os.chdir(IMAGE_DIR)
image_files = os.listdir()

# --- Images Processing ---


def filter_images(images):
    image_list = []
    for image in images:
        try:
            assert (cv2.imread(image)).shape[2] == 3
            image_list.append(image)
        except AssertionError as e:
            print(e)
    return image_list


def img_gray(image):
    image = cv2.imread(image)
    return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)


def resize(image, height=SIZE, width=SIZE):  # higher number = higher accuracy
    row_res = cv2.resize(image, (height, width),
                         interpolation=cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image, (height, width),
                         interpolation=cv2.INTER_AREA).flatten('F')
    return row_res, col_res


def intensity_diff(row_res, col_res):
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()


def difference_score(image, height=SIZE, width=SIZE):
    gray = img_gray(image)
    row_res, col_res = resize(gray, height, width)
    difference = intensity_diff(row_res, col_res)
    return difference

# --- Images Processing ---


def difference_score_dict(image_list):
    ds_dict = {}
    duplicates = []
    for image in image_list:
        ds = difference_score(image)
        if image not in ds_dict:
            ds_dict[image] = ds
        else:
            duplicates.append((image, ds_dict[image]))

    return duplicates, ds_dict


image_files = filter_images(image_files)
duplicates, ds_dict = difference_score_dict(image_files)


def jaccard(arr1: np.ndarray, arr2: np.ndarray) -> float:
    count = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            count += 1
    return count/len(arr1)


for k1, k2 in itertools.combinations(ds_dict, 2):
    print(jaccard(ds_dict[k1], ds_dict[k2]))
    if jaccard(ds_dict[k1], ds_dict[k2]) >= SIMILARITY:
        duplicates.append((k1, k2))

for file_names in duplicates:
    try:

        plt.subplot(121), plt.imshow(cv2.imread(file_names[0]))
        plt.title('Similar'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(cv2.imread(file_names[1]))
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.show()

    except OSError as e:
        continue
