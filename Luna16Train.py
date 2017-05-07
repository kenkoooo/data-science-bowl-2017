import argparse
import random

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

import UNet

random.seed(71)


def sample(img_array: np.ndarray, sample_list: list) -> np.ndarray:
    return np.array([img_array[i] for i in range(len(img_array)) if i in sample_list])


def load_images(patient_ids: list, directory: str, *, size, k) -> (np.ndarray, np.ndarray):
    lungs = []
    masks = []
    for patient_id in tqdm(patient_ids):
        lung_path = "{}/{}_lung_cropped_{}.npz".format(directory, patient_id, size)
        mask_path = "{}/{}_nodule_cropped_{}.npz".format(directory, patient_id, size)

        lung_array = np.load(lung_path)
        mask_array = np.load(mask_path)
        sample_list = random.sample(range(len(lung_array)), k=k)

        lungs.extend(sample(lung_array, sample_list))
        masks.extend(sample(mask_array, sample_list))
    lungs = np.array(lungs)
    masks = np.array(masks)
    return lungs, masks


def main(args):
    annotations = pd.read_csv(args.a)
    directory = args.d
    output = args.o
    k = args.k
    patient_ids = list(set(annotations["seriesuid"]))
    size = 128

    lungs, masks = load_images(patient_ids, directory, size=128, k=k)

    print(lungs.shape)
    print(masks.shape)
    print(output)

    model = UNet.get_unet(size, size)
    model_checkpoint = ModelCheckpoint('{}/weights.h5'.format(output), monitor='val_loss', save_best_only=True)
    model.fit(lungs, masks, batch_size=32, nb_epoch=20, verbose=1, shuffle=True, validation_split=0.2,
              callbacks=[model_checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="image directory")
    parser.add_argument("-a", type=str, help="annotation csv file")
    parser.add_argument("-o", type=str, help="path to output the checkpoint file", default="./")
    parser.add_argument("-k", type=str, help="sampling from each patient", default=150)
    main(parser.parse_args())
