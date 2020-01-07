import logging
import os
import tarfile

import wget

DATASETS_SOURCES_ROOT = "./datasets/sources/"
DATASETS_UNPACKED_ROOT = "./datasets/unpacked/"


# URL - where dataset will be loaded
# save_as_name - name of directory where downloaded file will be loaded
def prepare_dataset(url, save_as_name):
    logging.info("Start the dataset downloading...")
    sources_path = DATASETS_SOURCES_ROOT + save_as_name
    if os.path.exists(sources_path):
        logging.info("Dataset exists, downloading was ended...")
        return
    else:
        os.mkdir(sources_path)
        filename = fetch_dataset(url, sources_path)
        unpacked_path = DATASETS_UNPACKED_ROOT + save_as_name
        unpack_dataset(filename, unpacked_path)


# fetch dataset from the URL
def fetch_dataset(url, path):
    logging.info("Start fetching the dataset from %s ...", url)
    downloaded_filename = wget.download(url, path)
    logging.info("Dataset downloading has been successfully!")
    return downloaded_filename


# unpacks .tar and .tar.gz archives
def unpack_dataset(path_extract_from, path_extract_to):
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