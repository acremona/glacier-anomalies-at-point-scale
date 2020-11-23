import numpy as np
import math

threshDuplicate = 25  # threshold to find duplicates within matches (in pixel)

def remove_duplicates(points):
    """
    This function is approximating points that are very close together into 1 single point

    Parameters
    ----------
    points : list of tuple of float
        list of x and y coordinates e.g. [[x1, y1], [x2, y2], ...]

    Returns
    -------
    points: list of tuple of float
        list of x and y coordinates of fewer points.
    """
    a = 0
    filtered_matches = []
    flags = []
    while a < len(points):
        if a in flags:
            a += 1
            continue
        duplicates = []
        for b in range(len(points)):
            d = math.sqrt((points[a][0] - points[b][0])**2 + (points[a][1] - points[b][1])**2)    # distance between 2 points
            if d < threshDuplicate and a <= b:
                duplicates.append(points[b])
                flags.append(b)
        duplicates = np.array(duplicates)
        if len(duplicates) < 1:
            filtered_matches.append(points[a])
        else:
            filtered_matches.append(duplicates.mean(axis=0))
    return [arr.tolist() for arr in filtered_matches]   # transform numpy array to a list
