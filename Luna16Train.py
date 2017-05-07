import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(71)


def sample(img_array: np.ndarray, sample_list: list) -> np.ndarray:
    return np.array([img_array[i] for i in range(len(img_array)) if i in sample_list])


def main(args):
    annotations = pd.read_csv(args.a)
    directory = args.d
    output = args.o
    patient_ids = list(set(annotations["seriesuid"]))
    size = 128
    k = 150

    lungs = []
    masks = []
    for patient_id in tqdm(patient_ids):
        lung_path = "{}/{}_lung_cropped_{}.npz".format(directory, patient_id, size)
        mask_path = "{}/{}_nodule_cropped_{}.npz".format(directory, patient_id, size)

        lung_array = np.load(lung_path)
        mask_array = np.load(mask_path)
        sample_list = random.sample(range(lung_array), k=k)

        lungs.append(sample(lung_array, sample_list))
        masks.append(sample(mask_array, sample_list))
    lungs = np.array(lungs)
    masks = np.array(masks)
    print(lungs.shape)
    print(masks.shape)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="image directory")
    parser.add_argument("-a", type=str, help="annotation csv file")
    parser.add_argument("-o", type=str, help="path to output the checkpoint file")
    main(parser.parse_args())
