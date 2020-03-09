import logging
import os

import dl_1.const as const
from util.file.dataset_loader import prepare_dataset, get_path_to_unpacked_dir
from util.file.image_handler import show_image, calc_file_hash, load_image_into_numpy_array
from util.runner import Runner
import numpy as np
from sklearn.linear_model import LogisticRegression


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

        # (3)
        training_set = []
        validation_set = []
        test_set = []

        logging.info("Start sorting by sets")
        for letter in const.LEARNING_LETTERS:
            path_to_img_dir = get_path_to_unpacked_dir(const.SECOND_UNIQ_DATASET_PATH_NAME) \
                              + "/" \
                              + const.SECOND_UNIQ_DATASET_PATH_NAME \
                              + "/" \
                              + letter
            files_in_dir = list(map(lambda path: path_to_img_dir + "/" + path, os.listdir(path_to_img_dir)))
            files_count = len(files_in_dir)
            # calculate how many pics are going to each set
            to_training = int(files_count * const.TRAIN_SET_PERCENTS)
            to_validation = int(files_count * const.VALIDATION_SET_PERCENTS)
            to_test = files_count - to_training - to_validation
            # using slice put file paths to sets
            training_set = training_set + files_in_dir[0: to_training - 1]
            validation_set = validation_set + files_in_dir[to_training: to_training + to_validation - 1]
            test_set = test_set + files_in_dir[to_training + to_validation: to_training + to_validation + to_test - 1]


        logging.info(
            "Total set was separted to 3 sets: training (%d - %d %%), validation (%d - %d %%) and test (%d - %d %%)",
            len(training_set), const.TRAIN_SET_PERCENTS * 100,
            len(validation_set), const.VALIDATION_SET_PERCENTS * 100,
            len(test_set), const.TEST_SET_PERCENTS * 100
        )

        # (4)
        uniq_images = {}
        duplicate_images = {}
        logging.info("Start deleting duplicates...")
        for letter in const.LEARNING_LETTERS:
            path_to_img_dir = get_path_to_unpacked_dir(const.SECOND_UNIQ_DATASET_PATH_NAME) \
                              + "/" \
                              + const.SECOND_UNIQ_DATASET_PATH_NAME \
                              + "/" \
                              + letter
            files_in_dir = os.listdir(path_to_img_dir)
            for file in files_in_dir:
                full_path = path_to_img_dir + "/" + file
                hash = calc_file_hash(full_path)
                if hash not in uniq_images:
                    uniq_images[hash] = full_path
                else:
                    duplicate_images[hash] = full_path

        logging.info("Was found %d duplicate images, total count of unique images = %d", len(duplicate_images),
                     len(uniq_images))

        logging.info("Delete duplicated images from the sets")
        for non_uniq_hash in duplicate_images:
            non_uniq_path = duplicate_images[non_uniq_hash]
            if non_uniq_path in training_set:
                training_set.remove(non_uniq_path)
            if non_uniq_path in validation_set:
                validation_set.remove(non_uniq_path)
            if non_uniq_path in test_set:
                test_set.remove(non_uniq_path)

        logging.info(
            "Total count of sets after duplicate deleting: training: %d ; validation: %d ; test: %d",
            len(training_set),
            len(validation_set),
            len(test_set)
        )

        # (5)
        # Prepare y's
        y = np.zeros((len(training_set), ))
        for letter in const.LEARNING_LETTERS:
            path_to_img_dir = get_path_to_unpacked_dir(const.SECOND_UNIQ_DATASET_PATH_NAME) \
                              + "/" \
                              + const.SECOND_UNIQ_DATASET_PATH_NAME \
                              + "/" \
                              + letter
            # On every letter set their own number 0 - len(const.LEARNING_LETTERS),
            # where 0 is A, 1 is B, C is 2 and so on
            for index, path in enumerate(training_set):
                if path.startswith(path_to_img_dir):
                    y[index] = const.LEARNING_LETTERS.index(letter)
        # Creates input values by concatenating of image values
        x = np.asarray([])
        for path in training_set[:]:
            image_np_array = load_image_into_numpy_array(path)
            # if file was deleted --- remove it from the set and from the y's
            if image_np_array is None:
                y = np.delete(y, [training_set.index(path)])
                training_set.remove(path)
                continue
            if x.shape[0] == 0:
                x = image_np_array.reshape((-1, 1))
            else:
                x = np.concatenate((x, image_np_array.reshape(-1, 1)), axis=1)

        # Learning using scikit
        logging.info("Start training model (with logistic regression)...")
        logistic_regression = LogisticRegression()
        logistic_regression.fit(x.T, y)
        # logistic_regression.predict(load_image_into_numpy_array(test_set[0]).reshape((-1, 1)).T)
        # print(is_a)






