from PIL import Image
import numpy as np


def load_image_into_numpy_array(img_path, mode="int32"):
    """
    Translates image into numpy array
    :param img_path: path to the translatable image
    :param mode: mode of reading, by default has int32 value
    :return: numpy array with image values
    """
    img = Image.open(img_path)
    img.load()
    data = np.asarray(img, dtype=mode)
    return data


def save_numpy_array_as_image(narray, path, mode="uint8", image_mode="L"):
    """
    Saves narray as image
    :param narray: numpy array which has values to the saving
    :param path: path, where image will be saved
    :param mode: mode to save, f.e. - int32, by default uint8
    :param image_mode: mode to save file, by default - L, to save in rgb use "RGB"
    :return:
    """
    img = Image.fromarray(np.asarray(np.clip(narray, 0, 255), dtype=mode), image_mode)
    img.save(path)
