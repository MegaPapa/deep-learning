import numpy as np

SUPPORTED_ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def get_onehot_for_letter(letter):
    """
    Translates letter of an alphabet to the numpy-onehot array, for example:
    A -> [ 1 0 0 ... ]
    B -> [ 0 1 0 ... ]
    :param letter: letter of an alphabet
    :return: numpy array with 1 on position of letter in the alphabet
    """
    onehot = np.zeros((len(SUPPORTED_ALPHABET), ))
    onehot[SUPPORTED_ALPHABET.index(letter)] = 1
    return onehot
