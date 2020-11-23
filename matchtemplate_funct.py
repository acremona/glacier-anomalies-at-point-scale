import cv2
import numpy as np
import math

#from find_collinear import find_collinear
#from remove_duplicates import remove_duplicates

threshGray = 0.6  # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8  # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
wait = 2  # time between every frame in ms, 0 for manual scrolling
threshAngle = 2                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)


def draw_rectangle(img, points, w, h, color, thickness):
    for count, point in enumerate(points):
        cv2.rectangle(img, (int(round(point[0])), int(round(point[1]))), (int(round(point[0]))+w, int(round(point[1]))+h), color, thickness)


def find_collinear(points):
    """
    This function searches for points that are collinear (on 1 straight line). If there are several lines, the one with
    the most points on it is chosen.

    Parameters
    ----------
    points : list of tuple of float
        list of x and y coordinates e.g. [[x1, y1], [x2, y2], ...]

    Returns
    -------
    collinear_points : list of tuple of float
        list of coordinates of all matches that are collinear
    angle : float
        the inclination of the pole. returns 0 if no collinear matches.
    """
    angles = []
    origins = []
    collinear_points = []
    bin_angles = []
    bin_origins = []

    if len(points) > 1:
        for a in range(len(points)):
            if a < len(points)-1:
                for b in range(a+1, len(points), 1):
                    dx = points[b][0] - points[a][0]       # getting distance in x direction
                    dy = points[b][1] - points[a][1]       # getting distance in y direction

                    if dy != 0:
                        angle = np.arctan(dx/dy)            # getting the angle of the connecting line between the 2 points
                        if abs(angle) < 35*np.pi/180:
                            origins.append(points[b][0]-points[b][1]*np.tan(angle))     # save angle and origin to a list, so the most common one can be picked later
                            angles.append(angle * 180 / np.pi)
        if len(angles) > 0:
            if len(angles) > 1:
                density, bin_edges = np.histogram(angles, bins=np.arange(min(angles), max(angles) + threshAngle, threshAngle))     # generating a histogram of all found angles
                found_angle = bin_edges[np.argmax(density)]*np.pi/180         # choose the highest density of calculated angles
            else:
                found_angle = angles[0]
        else:
            found_angle = 0

        if len(origins) > 0:
            if len(origins) > 1:
                density, bin_edges = np.histogram(origins, bins=np.arange(min(origins), max(origins) + 10, 10))     # generating a histogram of all found angles
                found_origin = bin_edges[np.argmax(density)]            # analog to angles
            else:
                found_origin = origins[0]
        else:
            found_origin = 0

        ####################################################################################
        # found_origin and found_angle are the lower bounds of the most common bin.
        # To increase accuracy, the average of all matches inside the bin is formed instead.
        ####################################################################################

        for point_a in points:
            for point_b in points:                 # 2 loops comparing all points with each other
                dx = point_b[0] - point_a[0]       # getting distance in x direction
                dy = point_b[1] - point_a[1]       # getting distance in y direction
                if dy != 0 and dx != 0:
                    angle = np.arctan(dx/dy)                            # getting the angle of the connecting line between the 2 points
                    origin = point_b[0]-point_b[1]*np.tan(angle)
                    if found_angle < angle < found_angle + threshAngle*np.pi/180 and found_origin < origin < found_origin + 20:  # if the angle is close to the most common angle and the same for the origin, the match is considered to be on the pole
                        collinear_points.append(point_a)
                        bin_angles.append(angle)
                        bin_origins.append(origin)
                        break                                           # if 1 pair of collinear points is found the iteration can be finished

        if len(bin_angles) > 0 and len(collinear_points) > 0:
            tuple_transform = [tuple(l) for l in collinear_points]              # getting rid of duplicate values in the array by transforming into a tuple
            o = np.average(bin_origins)
            an = np.average(bin_angles)
            #cv2.line(img, (int(round(o)), 0), (int(round(o+500*np.tan(an))), 500), (0, 0, 255), 2)
            return [t for t in (set(tuple(i) for i in tuple_transform))], np.average(bin_angles)        # and then creating a set (can only contain unique values) before transforming back to a list
        else:
            return [], 0
    else:
        return [], 0           # if 1 or less points is given as an argument, no collinear points can be found


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



def match_template(im,temp):
    """Finds areas of an image that match (are similar) to a template image.

    Parameters
    ----------
    im: ndarray
       (Source image) The image in which we expect to find a match to the template image.
    temp: ndarray
        (Template image) The image which will be compared to the source image.
    Returns
    -------
        list
            The coordinates of the matching points found (Left uppermost corner of the ROI).
    """

    print("[info] matchtemplate running ...")

    template_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # turn template into grayscale
    template_hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)    # turn template into HSV
    template_sat = template_hsv[:, :, 1]                    # extracting only the saturation channel of the HSV template

    matches = []
    gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # turn image into grayscale
    hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # turn image into HSV
    sat_img = hsv_img[:, :, 1]  # extracting only the saturation channel of the HSV image
    h, w = template_gray.shape

    resultGray = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
    locGray = np.where(resultGray >= threshGray)  # filter out bad matches
    for pt in zip(*locGray[::-1]):  # save all matches to a list
        matches.append([pt[0], pt[1]])

    resultSat = cv2.matchTemplate(sat_img, template_sat, cv2.TM_CCOEFF_NORMED)
    locSat = np.where(resultSat >= threshSat)  # filter out bad matches
    for pt in zip(*locSat[::-1]):  # add all matches to the list
        matches.append([pt[0], pt[1]])

    matches.sort(key=lambda y: int(y[1]))  # sort the matched points by y coordinate
    filteredMatches = remove_duplicates(matches)
    collinearMatches, pole_inclination = find_collinear(filteredMatches)
    print(len(collinearMatches))
    #cv2.imshow('im',im)
    #cv2.waitKey(0)
    if len(collinearMatches) > 0:
        return collinearMatches
    else:
        print('Canot find any match')
