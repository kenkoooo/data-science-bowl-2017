import argparse
import traceback

import pandas as pd
from tqdm import tqdm

from Luna16PreprocessUtils import images_and_nodules


def main(args) -> None:
    """
    load & resize the images and nodule masks, and dump them
    :param args: 
    :return: 
    """
    annotations = pd.read_csv(args.a)
    directory = args.d

    patient_ids = list(set(annotations["seriesuid"]))
    for patient_id in tqdm(patient_ids):
        # noinspection PyBroadException
        try:
            images_nodules = images_and_nodules(patient_id, annotations, directory)
            if not images_nodules:
                continue

            lung_img, nodule_mask = images_nodules
            lung_img.dump("{}/{}_lung_img.npz".format(directory, patient_id))
            nodule_mask.dump("{}/{}_nodule_mask.npz".format(directory, patient_id))
        except Exception:
            traceback.print_exc()
            print(patient_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", type=str, help="annotation csv file")
    parser.add_argument("-d", type=str, help="image directory")
    main(parser.parse_args())
