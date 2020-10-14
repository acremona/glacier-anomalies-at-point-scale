import numpy as np
import cv2

def automatic_mask(image):
    """Generate a mask with the most recurring color of an HSV image, based on the Hue values.

    Intervals: 0-20: red
               20-40: yellow
               40-70: green
               70-130: blue
               130-180: violett

    Parameters
    ----------
    image: ndarray
        Image for which the mask has to be created.
    Returns
    -------
        ndarray
            Mask containing the most recurrent color.
    """

    bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180] # do less color with differents range length an normalize?
    h1,bins = np.histogram(image, bins=bins)
    print("h1",h1)
    color = np.where(h1 == np.max(h1))
    print("color ",color)
    low_hue = int(bins[color])
    up_hue = int(bins[color[0]+1])

    print("low",low_hue,"up",up_hue)
    low = np.array([low_hue, 100., 0.]) #prima era 80
    up = np.array([up_hue, 255., 255.])
    mask = cv2.inRange(image, low, up)
    return mask