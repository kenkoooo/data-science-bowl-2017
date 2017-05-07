import argparse

import numpy as np
import pandas as pd


def main(args):
    annotations = pd.read_csv(args.a)
    directory = args.d
    output = args.o
    patient_ids = list(set(annotations["seriesuid"]))
    size = 128

    lungs = []
    masks = []
    for patient_id in patient_ids:
        lung_path = "{}/{}_lung_cropped_{}.npz".format(directory, patient_id, size)
        mask_path = "{}/{}_nodule_cropped_{}.npz".format(directory, patient_id, size)
        lungs.append(np.load(lung_path))
        masks.append(np.load(mask_path))
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
