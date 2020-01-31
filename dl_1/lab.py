import os

from util.file.dataset_loader import prepare_dataset, get_path_to_unpacked_dir
from util.file.image_handler import show_image
from util.runner import Runner

FIRST_DATASET_URL = "https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz"
FIRST_UNIQ_DATASET_PATH_NAME = "notMnist_large"

SECOND_DATASET_URL = "https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz"
SECOND_UNIQ_DATASET_PATH_NAME = "notMnist_small"


class Lab1(Runner):

    def __init__(self):
        prepare_dataset(FIRST_DATASET_URL, FIRST_UNIQ_DATASET_PATH_NAME)
        prepare_dataset(SECOND_DATASET_URL, SECOND_UNIQ_DATASET_PATH_NAME)
        self.learning_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def run(self):
        for letter in self.learning_letters:
            path_to_img_dir = get_path_to_unpacked_dir(SECOND_UNIQ_DATASET_PATH_NAME)\
                          + "/"\
                          + SECOND_UNIQ_DATASET_PATH_NAME\
                          + "/"\
                          + letter
            files_in_dir = os.listdir(path_to_img_dir)
            path_to_img = path_to_img_dir + "/" + files_in_dir[0]
            # (1)
            show_image(path_to_img)

