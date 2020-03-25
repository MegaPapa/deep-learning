from dl_2 import const
from util.file.dataset_loader import prepare_dataset, load_numpy_array_from_file, load_numpy_array_into_file, \
    get_path_to_unpacked_dir
from util.file.image_handler import load_image_into_numpy_array
from util.runner import Runner

import tensorflow as tf
from tensorflow import keras
import numpy as np


class Lab2(Runner):

    def __init__(self):
        prepare_dataset(const.FIRST_DATASET_URL, const.FIRST_UNIQ_DATASET_PATH_NAME)
        prepare_dataset(const.SECOND_DATASET_URL, const.SECOND_UNIQ_DATASET_PATH_NAME)

    def run(self):
        usable_dataset_name = const.SECOND_UNIQ_DATASET_PATH_NAME
        usable_dataset_url = const.SECOND_DATASET_URL

        y_output_name = usable_dataset_name + "_y_training_set"
        # Here we use None for set parameter because before I loaded it into file
        # so result will be ok
        y = self.generate_outputs(usable_dataset_name, None, y_output_name)
        x_trainset_name = usable_dataset_name + "_x_training_set"
        # The same as with Y's
        x_train = self.load_features(None, x_trainset_name, y)

        # (1)
        model = keras.Sequential([
            keras.layers.Input((784, 11170)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        model.fit(x=x_train, y=y)



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

