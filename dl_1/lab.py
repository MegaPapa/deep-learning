from util.file.dataset_loader import prepare_dataset
from util.runner import Runner

FIRST_DATASET_URL = "https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz"
FIRST_UNIQ_DATASET_PATH_NAME = "notMnist_large"

SECOND_DATASET_URL = "https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz"
SECOND_UNIQ_DATASET_PATH_NAME = "notMnist_small"


class Lab1(Runner):

    def __init__(self):
        prepare_dataset(FIRST_DATASET_URL, FIRST_UNIQ_DATASET_PATH_NAME)
        prepare_dataset(SECOND_DATASET_URL, SECOND_UNIQ_DATASET_PATH_NAME)

    def run(self):
        pass
