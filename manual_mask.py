import numpy as np
import cv2

def create_yellow_mask(image):
    """ Creates a mask for yellow.

    :param image: ndarray
        Image of which the mask is wanted.
    :return: ndarray
        The mask for yellow color of the image.
    """
    low_yellow = np.array([20., 80., 0.])
    upper_yellow = np.array([50., 255., 255.])
    mask = cv2.inRange(image, low_yellow, upper_yellow)
    return mask

def create_red_mask(image):
    """ Creates a mask for red.

    :param image: ndarray
        Image of which the mask is wanted.
    :return: ndarray
        The mask for red color of the image.
    """
    low_red = np.array([0., 80., 0.])
    upper_red = np.array([10., 255., 255.])
    mask = cv2.inRange(image, low_red, upper_red)
    return mask

def create_green_mask(image):
    """ Creates a mask for green.

    :param image: ndarray
        Image of which the mask is wanted.
    :return: ndarray
        The mask for red color of the image.
    """
    low_green = np.array([50., 80., 0.])
    upper_green = np.array([70., 255., 255.])
    mask = cv2.inRange(image, low_green, upper_green)
    return mask

def create_blue_mask(image):
    """ Creates a mask for blue.

    :param image: ndarray
        Image of which the mask is wanted.
    :return: ndarray
        The mask for red color of the image.
    """
    low_blue = np.array([100., 80., 0.])
    upper_blue = np.array([165., 255., 255.])
    mask = cv2.inRange(image, low_blue, upper_blue)
    return mask