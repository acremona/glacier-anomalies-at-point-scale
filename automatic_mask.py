import numpy as np
import cv2

def automatic_mask(hue,image):
    """Generate a mask with the most recurring color of an HSV image, based on the Hue values.

    Intervals: 0-15: red
               15-35: yellow
               35-70: green
               70-145: blue
               145-180: red

    Parameters
    ----------
    image: ndarray
        Image for which the mask has to be created.
    Returns
    -------
        ndarray
            Mask containing the most recurrent color.
    """

    h1, bins = np.histogram(hue, bins=36)
    print("h1", h1)
    ####################################
    bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180] # do less color with differents range length an normalize?
    #bins = [0, 15, 35, 70, 145, 180]
    #range = [15,20,35,75,35]
    h1,bins = np.histogram(hue, bins=bins)
    #print("h1",h1)
    #h1 = h1/range
    print("h1",h1)
    color = np.where(h1 == np.max(h1))
    print("color ",color)
    if color == 0 or color == 9:
        low1 = np.array([0, 100., 0.])
        up1 = np.array([15, 255., 255.])
        low2 = np.array([145, 100., 0.])
        up2 = np.array([180, 255., 255.])
        mask = cv2.inRange(image, low1, up1) + cv2.inRange(image, low2, up2)
        return mask
    else:
        if color == 1:
            low = np.array([15, 100., 0.])
            up = np.array([35, 255., 255.])
            mask = cv2.inRange(image, low, up)
            return mask
        else:
            if color == 2:
                low = np.array([35, 100., 0.])
                up = np.array([70, 255., 255.])
                mask = cv2.inRange(image, low, up)
                return mask
            else:
                if color == 4 or color == 5 or color == 6 or color ==  7:
                    low = np.array([70, 100., 0.])
                    up = np.array([145, 255., 255.])
                    mask = cv2.inRange(image, low, up)
                    return mask
                else:
                    low_hue = int(bins[color])
                    up_hue = int(bins[color[0]+1])
                    print("low",low_hue,"up",up_hue)
                    low = np.array([low_hue, 100., 0.]) #prima era 80
                    up = np.array([up_hue, 255., 255.])
                    mask = cv2.inRange(image, low, up)
                    return mask