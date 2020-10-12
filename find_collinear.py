import numpy as np


threshAngle = 1  # threshold to check if matches are on straight line (a.k.a. the pole) in degrees

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
