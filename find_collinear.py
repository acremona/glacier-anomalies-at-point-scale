import numpy as np


threshAngle = 2  # threshold to check if matches are on straight line (a.k.a. the pole) in degrees

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
