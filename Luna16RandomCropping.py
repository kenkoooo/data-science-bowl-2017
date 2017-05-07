import random

import numpy as np

random.seed(71)


def load_numpy_array(patient_id: str, directory: str) -> (np.ndarray, np.ndarray):
    lung_img = np.load("{}/{}_lung_img.npz".format(directory, patient_id))
    nodule_mask = np.load("{}/{}_nodule_mask.npz".format(directory, patient_id))
    return lung_img, nodule_mask


def transpose_y(img: np.ndarray) -> np.ndarray:
    return img.transpose((1, 0, 2))


def transpose_z(img: np.ndarray) -> np.ndarray:
    return img.transpose((2, 1, 0))


def crop(img: np.ndarray, *, idx: int, h_edge: int, w_edge: int, size: int) -> np.ndarray:
    return img[idx, h_edge:h_edge + size, w_edge:w_edge + size]


def crop_tuple(img: np.ndarray, size: int) -> (int, int, int):
    layers, height, width = img.shape
    l = random.randint(0, layers - 1)
    h = random.randint(0, height - size - 1)
    w = random.randint(0, width - size - 1)
    return l, h, w


def cropped_list(img: np.ndarray, mask: np.ndarray, *, size: int, num: int, axis="x") -> (np.ndarray, np.ndarray):
    if axis == "y":
        img = transpose_y(img)
        mask = transpose_y(mask)
    elif axis == "z":
        img = transpose_z(img)
        mask = transpose_z(mask)

    img_list = []
    mask_list = []
    for i in range(num):
        idx, h, w = crop_tuple(img, size)
        img_list.append(crop(img, idx=idx, h_edge=h, w_edge=w, size=size))
        mask_list.append(crop(mask, idx=idx, h_edge=h, w_edge=w, size=size))

    return np.array(img_list), np.array(mask_list)
