import random

import numpy as np


def load_numpy_array(patient_id: str, directory: str) -> (np.ndarray, np.ndarray):
    lung_img = np.load("{}/{}_lung_img.npz".format(directory, patient_id))
    nodule_mask = np.load("{}/{}_nodule_mask.npz".format(directory, patient_id))
    return lung_img, nodule_mask


def crop_x(lung_img: np.ndarray, nodule_mask: np.ndarray, size: int):
    return crop(lung_img, nodule_mask, size)


def crop_y(lung_img: np.ndarray, nodule_mask: np.ndarray, size: int):
    t = (1, 0, 2)
    return crop(lung_img.transpose(t), nodule_mask.transpose(t), size)


def crop_z(lung_img: np.ndarray, nodule_mask: np.ndarray, size: int):
    t = (2, 1, 0)
    return crop(lung_img.transpose(t), nodule_mask.transpose(t), size)


def crop(lung_img: np.ndarray, nodule_mask: np.ndarray, size: int):
    layers, height, width = lung_img.shape
    l = random.randint(0, layers - 1)
    h = random.randint(0, height - size - 1)
    w = random.randint(0, width - size - 1)
    return lung_img[l, h:h + size, w:w + size], nodule_mask[l, h:h + size, w:w + size]
