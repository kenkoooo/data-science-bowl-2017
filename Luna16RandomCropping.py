import argparse
import multiprocessing as mp
import random
import traceback

import numpy as np
import os
import pandas as pd
from tqdm import tqdm

random.seed(71)
np.random.seed(71)


def load_numpy_array(patient_id: str, directory: str) -> (np.ndarray, np.ndarray):
    lung_filepath = "{}/{}_lung_img.npz".format(directory, patient_id)
    nodule_filepath = "{}/{}_nodule_mask.npz".format(directory, patient_id)
    if not os.path.exists(lung_filepath) or not os.path.exists(nodule_filepath):
        return None

    lung_img = np.load(lung_filepath)
    nodule_mask = np.load(nodule_filepath)
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


def single_task(args) -> None:
    patient_id, directory, num = args

    size = 128
    # noinspection PyBroadException
    try:
        loaded_array = load_numpy_array(patient_id, directory)
        if not loaded_array:
            return

        lung_img, nodule_mask = loaded_array
        lung_x, nodule_x = cropped_list(lung_img, nodule_mask, size=size, num=num, axis="x")
        lung_y, nodule_y = cropped_list(lung_img, nodule_mask, size=size, num=num, axis="y")
        lung_z, nodule_z = cropped_list(lung_img, nodule_mask, size=size, num=num, axis="z")

        lung_resample = np.concatenate((lung_x, lung_y, lung_z))
        np.random.shuffle(lung_resample)
        lung_resample.dump("{}/{}_lung_cropped_{}.npz".format(directory, patient_id, size))

        nodule_resample = np.concatenate((nodule_x, nodule_y, nodule_z))
        np.random.shuffle(nodule_resample)
        nodule_resample.dump("{}/{}_nodule_cropped_{}.npz".format(directory, patient_id, size))
    except Exception:
        traceback.print_exc()
        print(patient_id)


def main(args) -> None:
    pool = mp.Pool(4)

    annotations = pd.read_csv(args.a)
    directory = args.d
    num = args.n

    patient_ids = list(set(annotations["seriesuid"]))
    args_list = [(patient_id, directory, num) for patient_id in patient_ids]
    progress_bar = tqdm(total=len(args_list))

    for _ in pool.imap_unordered(single_task, args_list):
        progress_bar.update(1)
    progress_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", type=str, help="annotation csv file")
    parser.add_argument("-d", type=str, help="image directory")
    parser.add_argument("-n", type=str, help="number of crops for each image", default=500)
    main(parser.parse_args())
