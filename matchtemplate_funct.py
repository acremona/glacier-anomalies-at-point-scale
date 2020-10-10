import cv2
import numpy as np
import os
import math

threshGray = 0.6  # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8  # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
threshDuplicate = 25  # threshold to find duplicates within matches (in pixel)
threshAngle = 1  # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
wait = 0  # time between every frame in ms, 0 for manual scrolling

def find_collinear(points):
    """
    This function searches for points that are collinear (on 1 straight line). If there are several lines, the one with
    the most points on it is chosen.

    :param points: list of x and y coordinates e.g. [[x1, y1], [x2, y2], ...]
    :return: list of x and y coordinates of points that are collinear.
    """
    angles = []
    origins = []
    collinear_points = []
    if len(points) > 1:
        for point_a in points:
            for point_b in points:
                dx = abs(point_a[0] - point_b[0])       # getting distance in x direction
                dy = abs(point_a[1] - point_b[1])       # getting distance in y direction

                if dy > 0 and dx > 0:
                    angle = np.arctan(dx/dy)      # getting the angle of the connecting line between the 2 points
                    angles.append(angle*180/np.pi)
                    if angle < 45*np.pi/180:
                        origins.append(point_b[0]-point_b[1]*np.tan(angle))

        density, bin_edges = np.histogram(angles, bins=100)     # generating a histogram of all found angles
        found_angle = bin_edges[np.argmax(density)]*np.pi/180         # choose the highest density of calculated angles
        density, bin_edges = np.histogram(origins, bins=100)     # generating a histogram of all found angles
        found_origin = bin_edges[np.argmax(density)]

        for point_a in points:
            for point_b in points:             # 2 loops comparing all points with each other
                dx = abs(point_a[0] - point_b[0])       # getting distance in x direction
                dy = abs(point_a[1] - point_b[1])       # getting distance in y direction
                if dy > 0 and dx > 0:
                    angle = np.arctan(dx/dy)                            # getting the angle of the connecting line between the 2 points
                    origin = point_b[0]-point_b[1]*np.tan(angle)
                    if abs(angle-found_angle) < threshAngle*np.pi/180 and abs(origin-found_origin) < 10:  # if the angle is close to the angle of the chosen line, the point lies on the line
                        collinear_points.append(point_a)
                        break                                           # if 1 pair of collinear points is found the iteration can be finished

        tuple_transform = [tuple(l) for l in collinear_points]              # getting rid of duplicate point in the array by transforming into a tuple
        return [t for t in (set(tuple(i) for i in tuple_transform))]        # and then creating a set (can only contain unique values) before transforming back to a list
    else:
        return points                                                       # if 1 or less points is given as an argument, no collinear points can be found, so output=input


def remove_duplicates(points):
    """
    This function is approximating points that are very close together into 1 single point

    :param points: list of x and y coordinates e.g. [[x1, y1], [x2, y2], ...]
    :return: list of x and y coordinates of fewer points.
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


def draw_rectangle(img, points, w, h, color, thickness):
    for count, point in enumerate(points):
        cv2.rectangle(img, (int(round(point[0])), int(round(point[1]))), (int(round(point[0]))+w, int(round(point[1]))+h), color, thickness)


def match_template(im,temp):

    print("[info] matchtemplate running ...")
    template_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # turn template into grayscale
    template_hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)    # turn template into HSV
    template_sat = template_hsv[:, :, 1]                        # extracting only the saturation channel of the HSV template

    for img in im:
        matches = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn image into grayscale
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # turn image into HSV
        sat_img = hsv_img[:, :, 1]                        # extracting only the saturation channel of the HSV image
        h, w = template_gray.shape                        # get width and height of image

        resultGray = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
        locGray = np.where(resultGray >= threshGray)  # filter out bad matches
        for pt in zip(*locGray[::-1]):                # save all matches to a list
            matches.append([pt[0], pt[1]])

        resultSat = cv2.matchTemplate(sat_img, template_sat, cv2.TM_CCOEFF_NORMED)
        locSat = np.where(resultSat >= threshSat)     # filter out bad matches
        for pt in zip(*locSat[::-1]):                 # add all matches to the list
            matches.append([pt[0], pt[1]])

        sorted(matches, key=lambda x: x[0])           # sort the matched points by x coordinate

        #print("Matches: " + str(len(matches)))
        filteredMatches = remove_duplicates(matches)
        #print("After duplicate check: " + str(len(filteredMatches)))
        verticalMatches = find_collinear(filteredMatches)
        #print("After collinearity check: " + str(len(verticalMatches)))
        verticalMatches = verticalMatches[0:2]                            #select the number of matches you want

        #draw_rectangle(img, matches, w, h, (150, 250, 255), 1)
        #draw_rectangle(img, filteredMatches, w, h, (0, 255, 0), 1)
        #draw_rectangle(img, verticalMatches, w, h, (255, 0, 0), 2)
        #cv2.imshow("img", img)
        cv2.waitKey(wait)
    print("[info] matchtemplate ended ...")
    return verticalMatches

cv2.destroyAllWindows()