import logging
import os

import dl_1.const as const
from util.file.dataset_loader import prepare_dataset, get_path_to_unpacked_dir, load_numpy_array_from_file, \
    load_numpy_array_into_file
from util.file.image_handler import show_image, calc_file_hash, load_image_into_numpy_array
from util.runner import Runner
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import matplotlib.pyplot as plt


class Lab1(Runner):

    def __init__(self):
        prepare_dataset(const.FIRST_DATASET_URL, const.FIRST_UNIQ_DATASET_PATH_NAME)
        prepare_dataset(const.SECOND_DATASET_URL, const.SECOND_UNIQ_DATASET_PATH_NAME)

    def run(self):
        usable_dataset_name = const.SECOND_UNIQ_DATASET_PATH_NAME
        usable_dataset_url = const.SECOND_DATASET_URL
        # images_count = {}
        # for letter in const.LEARNING_LETTERS:
        #     path_to_img_dir = get_path_to_unpacked_dir(usable_dataset_name)\
        #                   + "/"\
        #                   + usable_dataset_name\
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

        logging.info("Start sorting into sets")
        for letter in const.LEARNING_LETTERS:
            path_to_img_dir = get_path_to_unpacked_dir(usable_dataset_name) \
                              + "/" \
                              + usable_dataset_name \
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
            path_to_img_dir = get_path_to_unpacked_dir(usable_dataset_name) \
                              + "/" \
                              + usable_dataset_name \
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
        y_output_name = usable_dataset_name + "_y_training_set"
        y = self.generate_outputs(usable_dataset_name, training_set, y_output_name)

        y_test_output_name = usable_dataset_name + "_y_test_set"
        y_test = self.generate_outputs(usable_dataset_name, test_set, y_test_output_name)

        # Creates input values by concatenating of image values
        x_trainset_name = usable_dataset_name + "_x_training_set"
        x_train = self.load_features(training_set, x_trainset_name, y)

        x_testset_name = usable_dataset_name + "_x_test_set"
        x_test = self.load_features(test_set, x_testset_name, y_test)

        # Learning using scikit
        logging.info("Start training model (with logistic regression)...")
        # self.train_on_n_examples(x_train, x_test, y, y_test, 50, 50)
        # self.train_on_n_examples(x_train, x_test, y, y_test, 100, 100)
        # self.train_on_n_examples(x_train, x_test, y, y_test, 1000, 1000)
        set_50 = self.get_small_set(usable_dataset_name, training_set, 50)
        y_output_name_50 = usable_dataset_name + "_y_training_set_50"
        x_output_name_50 = usable_dataset_name + "_x_training_set_50"
        y_50 = self.generate_outputs(usable_dataset_name, set_50, y_output_name_50)
        x_50 = self.load_features(set_50, x_output_name_50, y_50)

        set_100 = self.get_small_set(usable_dataset_name, training_set, 100)
        y_output_name_100 = usable_dataset_name + "_y_training_set_100"
        x_output_name_100 = usable_dataset_name + "_x_training_set_100"
        y_100 = self.generate_outputs(usable_dataset_name, set_100, y_output_name_100)
        x_100 = self.load_features(set_100, x_output_name_100, y_100)

        set_1000 = self.get_small_set(usable_dataset_name, training_set, 1000)
        y_output_name_1000 = usable_dataset_name + "_y_training_set_1000"
        x_output_name_1000 = usable_dataset_name + "_x_training_set_1000"
        y_1000 = self.generate_outputs(usable_dataset_name, set_1000, y_output_name_1000)
        x_1000 = self.load_features(set_1000, x_output_name_1000, y_1000)

        result_50 = self.train_on_n_examples(x_50, x_test, y_50, y_test)
        result_100 = self.train_on_n_examples(x_100, x_test, y_100, y_test)
        result_1000 = self.train_on_n_examples(x_1000, x_test, y_1000, y_test)
        result_all = self.train_on_n_examples(x_train, x_test, y, y_test)
        plt.ylabel('Accuracy')
        plt.xlabel('Examples count')
        plt.plot([result_50[0], result_100[0], result_1000[0], result_all[0]], [result_50[1], result_100[1], result_1000[1], result_all[1]])
        plt.show()
        # working configurations:
        #   max_iter=1000000, solver='liblinear' - 31 min
        #   max_iter=1000000, tol=1e-2 - 1m 40 sec

        # logistic_regression.predict(load_image_into_numpy_array(test_set[0]).reshape((-1, 1)).T)
        # print(is_a)

    def get_small_set(self, usable_dataset_name, set, size):
        count_of_elements = 0
        # index of element in new filtered set
        element_num = 0
        new_set = []
        while size > count_of_elements:
            for letter in const.LEARNING_LETTERS:
                path_to_img_dir = get_path_to_unpacked_dir(usable_dataset_name) \
                                  + "/" \
                                  + usable_dataset_name \
                                  + "/" \
                                  + letter
                filtered_set = list(filter(lambda el: el.startswith(path_to_img_dir), set))
                new_set.append(filtered_set[element_num])
                count_of_elements += 1
            element_num += 1
        return new_set


    def train_on_n_examples(self, x_train, x_test, y_train, y_test):
        train_examples_count = x_train.shape[1]
        test_examples_count = x_test.shape[1]
        logistic_regression = LogisticRegression(max_iter=1000, tol=1e-2, C=0.5, solver='liblinear',
                                                 penalty='l1')  # 1 mln
        logistic_regression.fit(x_train.T, y_train)
        logging.info("Model has been trained successfully!")

        score_result = logistic_regression.score(x_test.T, y_test)
        score_result_2 = logistic_regression.score(x_train.T, y_train)
        logging.info("Train examples count: %d", train_examples_count)
        logging.info("Test examples count: %d", test_examples_count)
        logging.info("Score on test data: %f", score_result)
        logging.info("Score on train data: %f", score_result_2)
        return (train_examples_count, score_result)

    def generate_outputs(self, usable_dataset_name, source_set, outputset_name):
        """
        Generate vector of output values - every letter their own index (a - 0, b - 1 ...)
        :param usable_dataset_name: name of dataset which will be used (unpacked)
        :param source_set: for which dataset will be generate output
        :param outputset_name: name of output that will be used to save y's to the file
        :return: y's vector
        """
        y = load_numpy_array_from_file(outputset_name)
        # if preloaded y's don't exist - create it
        if y is None:
            y = np.zeros((len(source_set),))
            for letter in const.LEARNING_LETTERS:
                path_to_img_dir = get_path_to_unpacked_dir(usable_dataset_name) \
                                  + "/" \
                                  + usable_dataset_name \
                                  + "/" \
                                  + letter
                # On every letter set their own number 0 - len(const.LEARNING_LETTERS),
                # where 0 is A, 1 is B, C is 2 and so on
                for index, path in enumerate(source_set):
                    if path.startswith(path_to_img_dir):
                        y[index] = const.LEARNING_LETTERS.index(letter)
            load_numpy_array_into_file(y, outputset_name)
        return y

    def load_features(self, dataset, dataset_name, y_vector):
        """
        Translate set of files into numpy array
        :param dataset: dataset with paths to the files
        :param dataset_name: name of the file where numpy array will be saved
        :return: numpy array of features
        """
        x = load_numpy_array_from_file(dataset_name)
        # if preloaded x_train's don't exist - create it
        if x is None:
            x = np.asarray([])
            for path in dataset[:]:
                image_np_array = load_image_into_numpy_array(path)
                # if file was deleted --- remove it from the set and from the y's
                if image_np_array is None:
                    y_vector = np.delete(y_vector, [dataset.index(path)])
                    dataset.remove(path)
                    continue
                if x.shape[0] == 0:
                    x = image_np_array.reshape((-1, 1))
                else:
                    x = np.concatenate((x, image_np_array.reshape(-1, 1)), axis=1)
            load_numpy_array_into_file(x, dataset_name)
        return x



