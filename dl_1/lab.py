import logging
import os

import dl_1.const as const
from util.file.dataset_loader import prepare_dataset, get_path_to_unpacked_dir
from util.file.image_handler import show_image, calc_file_hash
from util.runner import Runner


class Lab1(Runner):

    def __init__(self):
        prepare_dataset(const.FIRST_DATASET_URL, const.FIRST_UNIQ_DATASET_PATH_NAME)
        prepare_dataset(const.SECOND_DATASET_URL, const.SECOND_UNIQ_DATASET_PATH_NAME)

    def run(self):
        # images_count = {}
        # for letter in const.LEARNING_LETTERS:
        #     path_to_img_dir = get_path_to_unpacked_dir(const.FIRST_UNIQ_DATASET_PATH_NAME)\
        #                   + "/"\
        #                   + const.FIRST_UNIQ_DATASET_PATH_NAME\
        #                   + "/"\
        #                   + letter
        #     files_in_dir = os.listdir(path_to_img_dir)
        #     path_to_img = path_to_img_dir + "/" + files_in_dir[0]
        #     # (1)
        #     show_image(path_to_img)
        #     # save count of images in letter in map
        #     images_count[letter] = len(files_in_dir)
        #
        # #(2)
        # print(images_count)
        # base = images_count[const.LEARNING_LETTERS[0]]
        # for letter in const.LEARNING_LETTERS:
        #     if images_count[letter] - base > const.CLASSES_DIFFERENCE_ERROR:
        #         logging.error("Images have too much error!")
        #     logging.info("Classes have normal error. [letter %s , elements %s]", letter, images_count[letter])

        uniq_images = {}
        duplicate_images = {}
        # (3)
        logging.info("Start deleting duplicates...")
        for letter in const.LEARNING_LETTERS:
            path_to_img_dir = get_path_to_unpacked_dir(const.SECOND_UNIQ_DATASET_PATH_NAME)\
                          + "/"\
                          + const.SECOND_UNIQ_DATASET_PATH_NAME\
                          + "/"\
                          + letter
            files_in_dir = os.listdir(path_to_img_dir)
            for file in files_in_dir:
                hash = calc_file_hash(path_to_img_dir + "/" + file)
                if hash not in uniq_images:
                    uniq_images[hash] = file
                else:
                    duplicate_images[hash] = file

        logging.info("Was found %d duplicate images, total count of unique images = %d", len(duplicate_images), len(uniq_images))

