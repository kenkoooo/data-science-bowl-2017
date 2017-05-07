import argparse
import multiprocessing as mp
import time
import traceback

import pandas as pd
from tqdm import tqdm

from Luna16PreprocessUtils import images_and_nodules


def load_and_dump(patient_id: str, directory: str, annotations: pd.DataFrame) -> bool:
    # noinspection PyBroadException
    try:
        images_nodules = images_and_nodules(patient_id, annotations, directory)
        if not images_nodules:
            return False

        lung_img, nodule_mask = images_nodules
        lung_img.dump("{}/{}_lung_img.npz".format(directory, patient_id))
        nodule_mask.dump("{}/{}_nodule_mask.npz".format(directory, patient_id))
        return True
    except Exception:
        traceback.print_exc()
        print(patient_id)
        return False


def single_task(args) -> str:
    patient_id = args[0]

    if not load_and_dump(patient_id, args[1], args[2]):
        time.sleep(0.1)
    return patient_id


def main(args) -> None:
    """
    load & resize the images and nodule masks, and dump them
    :param args: 
    :return: 
    """
    pool = mp.Pool(4)

    annotations = pd.read_csv(args.a)
    directory = args.d

    patient_ids = list(set(annotations["seriesuid"]))
    args_list = [(patient_id, directory, annotations) for patient_id in patient_ids]
    progress_bar = tqdm(total=len(args_list))

    for _ in pool.imap_unordered(single_task, args_list):
        progress_bar.update(1)
    progress_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", type=str, help="annotation csv file")
    parser.add_argument("-d", type=str, help="image directory")
    main(parser.parse_args())
