from util.file.dataset_loader import prepare_dataset
from util.runner import Runner

DATASET_URL = ""
UNIQ_DATASET_PATH_NAME = ""


class Lab1(Runner):

    def __init__(self):
        prepare_dataset(DATASET_URL, UNIQ_DATASET_PATH_NAME)

    def run(self):
        pass
