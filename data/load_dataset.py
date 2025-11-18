import os
import tarfile
import numpy as np

from .misc_data_util import transforms as trans
from .misc_data_util.url_save import save
from zipfile import ZipFile


def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val), each of which is a PyTorch dataset.
    """
    data_path = data_config["data_path"]  # path to data directory
    # if data_path is not None:
    if data_path is None:
        assert os.path.exists(data_path), "Data path {} not found.".format(data_path)

    # the name of the dataset to load
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case

    mode = data_config['mode']
    train = val = None
    if dataset_name == 'satellite':
        from .dataloader import Satellite, Satellite_Whole

        if mode == 'train':
            train_data_dir = '/workspace3/daikuai/process_FY-4A_data/nature_256_satellite_images/train'
            val_data_dir = '/workspace3/daikuai/process_FY-4A_data/nature_256_satellite_images/test'
            train = Satellite(train_data_dir, 24, train=True)
            val = Satellite(val_data_dir, 24, train=False)
        else:
            test_data_dir = './samples'
            val = Satellite_Whole(test_data_dir, 24, train=False)
            train = val
    else:
        raise Exception("Dataset name not found.")

    return train, val
