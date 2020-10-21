import cv2
import numpy as np
import os
import math
import pyqtgraph as pg
import matplotlib.pyplot as plt
import datetime

########################################################################################################
path = "C:\\Users\\joelb\\Downloads\\holfuy_images_2019\\1001_selection"           # change path to folder with images
template = cv2.imread("resources/roi.jpg")  # change to path of RoI
threshGray = 0.6                            # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8                             # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)
threshAngle = 2                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
threshTracking = 3                          # threshold to catch match from last frame (in pixel)
wait = 100                                  # time between every frame in ms, 0 for manual scrolling
########################################################################################################


def load_images_from_folder(folder):
    """
    function loading all image files inside a specified folder.

    :param folder: path of the folder (string). can be a relative or absolute path.
    :return: list of opencv images
    """
    print("Images loading...")
    images = []
    times = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
            filename = filename.replace('_', '-')
            time = list(map(int, filename.split('.')[0].split('-')))                # remove file ending (eg. .jpg) and split string into a list of y, m, d, h, s
            dt = datetime.datetime(time[0], time[1], time[2], time[3], time[4])     # convert into datetime format
            if len(times) == 0:                                                     # exception for first frame
                first_time = dt.timestamp()                                         # convert datetime into seconds
                times.append(0)
            else:
                times.append((dt.timestamp()-first_time)/3600)                      # get time difference from first frame in hours

    print("Images loaded")
    return images, times


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
                dx = point_b[0] - point_a[0]       # getting distance in x direction
                dy = point_b[1] - point_a[1]       # getting distance in y direction

                if dy != 0 and dx != 0:
                    angle = np.arctan(dx/dy)      # getting the angle of the connecting line between the 2 points
                    angles.append(angle*180/np.pi)
                    if abs(angle) < 35*np.pi/180:
                        origins.append(point_b[0]-point_b[1]*np.tan(angle))

        density, bin_edges = np.histogram(angles, bins=100)     # generating a histogram of all found angles
        found_angle = bin_edges[np.argmax(density)]*np.pi/180         # choose the highest density of calculated angles
        density, bin_edges = np.histogram(origins, bins=100)     # generating a histogram of all found angles
        found_origin = bin_edges[np.argmax(density)]
        for point_a in points:
            for point_b in points:             # 2 loops comparing all points with each other
                dx = point_b[0] - point_a[0]       # getting distance in x direction
                dy = point_b[1] - point_a[1]       # getting distance in y direction
                if dy != 0 and dx != 0:
                    angle = np.arctan(dx/dy)                            # getting the angle of the connecting line between the 2 points
                    origin = point_b[0]-point_b[1]*np.tan(angle)
                    if abs(angle-found_angle) < threshAngle*np.pi/180 and abs(origin-found_origin) < 20:  # if the angle is close to the angle of the chosen line, the point lies on the line
                        collinear_points.append(point_a)
                        break                                           # if 1 pair of collinear points is found the iteration can be finished

        tuple_transform = [tuple(l) for l in collinear_points]              # getting rid of duplicate point in the array by transforming into a tuple
        return [t for t in (set(tuple(i) for i in tuple_transform))]        # and then creating a set (can only contain unique values) before transforming back to a list
    else:
        return points                                                       # if 1 or less points is given as an argument, no collinear points can be found, so output=input


def get_scale(points):
    distances = []
    scaled_distances = []
    points.sort(key=lambda x: int(x[1]))                                     # sort matches by y coordinate
    if len(points) > 1:
        for a in range(len(points)-1):
            for b in range(a+1, len(points)-1, 1):
                dist = math.sqrt((points[a][0]-points[b][0])**2 + (points[a][1]-points[b][1])**2)
                distances.append(dist)
                # if b-a > 1:
                #     distances.append(dist/2)
                #     distances.append(dist/3)

        density, bin_edges = np.histogram(distances, bins=np.arange(min(distances), max(distances) + 10, 10))  # generating a histogram of all found distances
        diff = bin_edges[np.argmax(density)]

        for dist in distances:
            for c in range(int(round(dist/diff))):
                if c > 0:
                    scaled_distances.append(dist/c)
    else:
        print("Cannot calculate scale with less than 2 matches.")


def get_distance(newmatches, oldmatches):
    """Detects the most common distance between 2 sets of points. This function is beneficial because all tape stripes
    on the pole are supposed to move the same amount of distance between 2 frames. Therefore, if as input all matches
    in the old frame and all matches in the new frame are chosen, the most common distance between all combination of points
    will be the displacement of the pole.

    Parameters
    ----------
    newmatches : list
        List of point coordinates (x, y)
    oldmatches : list
        List of point coordinates (x, y)
    Returns
    -------
        float
            The most common distance between every single combination of the two given sets of coordinates.
    """
    differences = []
    if len(newmatches) > 0:
        if len(oldmatches) > 0:
            for newmatch in newmatches:
                for oldmatch in oldmatches:
                    if oldmatch[1] > newmatch[1]:
                        differences.append(math.sqrt((oldmatch[0]-newmatch[0])**2 + (oldmatch[1]-newmatch[1])**2))
            if len(differences) > 1:
                density, bin_edges = np.histogram(differences, bins=np.arange(min(differences), max(differences) + 1, 1))  # generating a histogram of all found distances
                diff = bin_edges[np.argmax(density)]                      # determining the bin with the highest amount of occurrences
                if 0 < diff < 40:
                    return diff
                else:
                    return 0
            else:
                return 0


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


def compare_matches(current_matches, old_matches, total_displacement, delta):
    for old_match in old_matches:  # plots a circle at the position where a match occurred in the previous frame
        cv2.circle(img, (int(round(old_match[0])), int(round(old_match[1]))), 2, (255, 255, 0), cv2.FILLED)

    found_difference = get_distance(current_matches, old_matches)   # get most common distance between current and previous frame
    try:                                                            # general case, add displacement of current frame to total displacements
        total_displacement.append(total_displacement[-delta] + found_difference)
    except IndexError:
        total_displacement.append(found_difference)                 # special case for the first frame
    for new in current_matches:                                     # the following loops visualize the displacement between 2 consecutive frames
        for old in old_matches:
            if old[1] > new[1]:                                     # only draw when the displacement is negative (glacier melts)
                difference = math.sqrt((old[0] - new[0]) ** 2 + (old[1] - new[1]) ** 2)  # pythagoras (to-do: change to displacement in direction of pole only)
                if abs(difference - found_difference) < threshTracking:  # if displacement is within 3 px of most common displacement, plot it
                    all_lines.append([(int(round(old[0])), int(round(old[1]))), (int(round(new[0])), int(round(new[1])))])
                    if len(all_lines) > 10:
                        del all_lines[0]
    for line in all_lines:
        cv2.line(img, line[0], line[1], (0, 0, 255), 2)
    print(str(delta) + "-frame displacement: " + str(found_difference) + "px. Total: " + str(total_displacement[-1]) + "px.")
    return total_displacement


def draw_rectangle(img, points, w, h, color, thickness):
    for count, point in enumerate(points):
        cv2.rectangle(img, (int(round(point[0])), int(round(point[1]))), (int(round(point[0]))+w, int(round(point[1]))+h), color, thickness)


images, times = load_images_from_folder(path)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # turn template into grayscale
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)    # turn template into HSV
template_sat = template_hsv[:, :, 1]                        # extracting only the saturation channel of the HSV template
all_matches = []                                            # list where all matches of all frames will be safed
one_frame_displacement = []                                 # list for plotting where all cumulative displacements will be stored
all_lines = []                                              # list for visualizing displacements between consecutive frames
x = []                                                      # x coordinate for plotting
win = pg.GraphicsWindow()                                   # initialize plotting
pw = win.addPlot()
disp = pw.plot()

for frame_nr, img in enumerate(images):
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

    if len(matches) > 0:
        matches.sort(key=lambda y: int(y[1]))           # sort the matched points by y coordinate
        filteredMatches = remove_duplicates(matches)
        collinearMatches = find_collinear(filteredMatches)
        all_matches.append(collinearMatches)
        # draw_rectangle(img, matches, w, h, (150, 250, 255), 1)
        # draw_rectangle(img, filteredMatches, w, h, (0, 255, 0), 1)
        draw_rectangle(img, collinearMatches, w, h, (255, 0, 0), 1)
        # get_scale(collinearMatches)
    else:
        all_matches.append([])

    if len(all_matches[-1]) > 0:                    # if there are matches in the current frame
        if frame_nr > 0:                            # special case for 1st frame because no displacement can be calculated there.
            if len(all_matches[-2]) > 0:            # if there are matches in the previous frame.
                one_frame_displacement = compare_matches(all_matches[-1], all_matches[-2], one_frame_displacement, 1)
                if one_frame_displacement[-1] > 0:
                    x.append(times[frame_nr])
                else:
                    if frame_nr > 1 and len(all_matches[-3]) > 0:
                        one_frame_displacement = compare_matches(all_matches[-1], all_matches[-3], one_frame_displacement, 2)
                        x.append(times[frame_nr])
            else:
                if frame_nr > 1 and len(all_matches[-3]) > 0:
                    one_frame_displacement = compare_matches(all_matches[-1], all_matches[-3], one_frame_displacement, 2)
                    x.append(times[frame_nr])
            disp.setData(x, one_frame_displacement, symbolBrush=('b'))

    else:
        print("No matches found in current frame!")

    cv2.imshow("img", img)
    cv2.waitKey(wait)

cv2.destroyAllWindows()
