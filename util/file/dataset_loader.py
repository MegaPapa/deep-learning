import logging
import os
import tarfile

import wget
import numpy as np

DATASETS_SOURCES_ROOT = "./datasets/sources/"
DATASETS_UNPACKED_ROOT = "./datasets/unpacked/"
SAVED_NUMPY_ARRAYS_PATH = "./datasets/arrays/"


def get_path_to_unpacked_dir(dir_name):
    return DATASETS_UNPACKED_ROOT + dir_name


def get_path_to_sources_dir(dir_name):
    return DATASETS_SOURCES_ROOT + dir_name


def load_numpy_array_into_file(np_array, dataset_uniq_name):
    """
    Loads numpy array into file to get it later
    :param np_array: array that will be loaded in the file
    :param dataset_uniq_name: special uniq name for that dataset
    """
    if not os.path.exists(SAVED_NUMPY_ARRAYS_PATH):
        logging.info("Package with prepared numpy arrays doesn't exist, creating it...")
        os.mkdir(SAVED_NUMPY_ARRAYS_PATH)
    full_path = SAVED_NUMPY_ARRAYS_PATH + dataset_uniq_name + ".npy"
    logging.info("Saving numpy array into %s", full_path)
    np.save(full_path, np_array)
    logging.info("Numpy array saved successfully!")


def load_numpy_array_from_file(dataset_uniq_name):
    """
    Loads numpy array from file
    :param dataset_uniq_name: name that will be used to load numpy array
    :return: numpy array or None if file with numpy array doesn't exist
    """
    full_path = SAVED_NUMPY_ARRAYS_PATH + dataset_uniq_name + ".npy"
    if not os.path.exists(full_path):
        logging.info("Doesn't exist numpy array %s , skipping loading!", full_path)
        return None
    logging.info("Loading prepared numpy array from %s", full_path)
    arr = np.load(full_path)
    logging.info("Numpy array was loaded successfully!")
    return arr


def prepare_dataset(url, save_as_name):
    """
    Prepares dataset to work with it
    :param url: url where resources is storing
    :param save_as_name: name of directory where it will be saved
    """
    if not os.path.exists(DATASETS_SOURCES_ROOT):
        logging.info("Dataset's source root doesn't exist, creating it...")
        os.mkdir(DATASETS_SOURCES_ROOT)

    if not os.path.exists(DATASETS_UNPACKED_ROOT):
        logging.info("Dataset's unpacking directory doesn't exist, creating it...")
        os.mkdir(DATASETS_UNPACKED_ROOT)

    logging.info("Start the dataset downloading...")
    sources_path = get_path_to_sources_dir(save_as_name)
    if os.path.exists(sources_path):
        logging.info("Dataset exists, downloading was ended...")
        return
    else:
        os.mkdir(sources_path)
        filename = fetch_dataset(url, sources_path)
        unpacked_path = get_path_to_unpacked_dir(save_as_name)
        unpack_dataset(filename, unpacked_path)


def fetch_dataset(url, path):
    """
    Fetch dataset from the URL
    :param url: url where resources is storing
    :param path:
    :return: downloaded files
    """
    logging.info("Start fetching the dataset from %s ...", url)
    downloaded_filename = wget.download(url, path)
    logging.info("Dataset downloading has been successfully!")
    return downloaded_filename


# unpacks .tar and .tar.gz archives
def unpack_dataset(path_extract_from, path_extract_to):
    """
    Unpacks dataset archive to the path
    :param path_extract_from: source of archive
    :param path_extract_to: target where to unpack
    """
    logging.info("Start unpacking from %s to %s ...", path_extract_from, path_extract_to)
    if path_extract_from.endswith("tar.gz"):
        tar = tarfile.open(path_extract_from, "r:gz")
        tar.extractall(path=path_extract_to)
        tar.close()
    elif path_extract_from.endswith("tar"):
        tar = tarfile.open(path_extract_from, "r:")
        tar.extractall(path=path_extract_to)
        tar.close()
    logging.info("File was extracted successfully!")
