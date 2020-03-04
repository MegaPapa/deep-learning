from _md5 import md5

from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def show_image(path):
    """
    Shows image which is placed on defined path
    :param path: path to the image
    """
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()
    plt.close()


def calc_file_hash(filepath):
    """
    Calculates hash of file
    :param filepath: path to the file
    :return: hash as string
    """
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()


def compare_images(first_img_path, second_img_path):
    """
    Compares two pictures and return result of comparing
    :param first_img_path:
    :param second_img_path:
    :return: boolean
    """
    img1 = Image.open(first_img_path)
    img2 = Image.open(second_img_path)

    diff = ImageChops.difference(img1, img2)
    print(diff.getbbox())