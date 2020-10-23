import cv2
import numpy as np
import os
import math
import pyqtgraph as pg
import matplotlib.pyplot as plt
import datetime

########################################################################################################
path = "C:\\Users\\joelb\\Downloads\\holfuy_images_2019\\1003"           # change path to folder with images
template = cv2.imread("resources/roi3.jpg")  # change to path of RoI
threshGray = 0.6                            # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8                             # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)
threshAngle = 2                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
threshTracking = 3                          # threshold to catch match from last frame (in pixel)
wait = 10                                  # time between every frame in ms, 0 for manual scrolling
########################################################################################################


def load_images_from_folder(folder):
    """
    function loading all image files inside a specified folder. File names must contain date and time with delimiters - or _

    Parameters
    ----------
    folder : string
        path of the folder (string). can be a relative or absolute path.

    Returns
    -------
        list of opencv images
        list of time differences to the first image in hours. e.g. [0, 0.5, 1.0, ...]
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

    Parameters
    ----------
    points : list
        list of x and y coordinates e.g. [[x1, y1], [x2, y2], ...]

    Returns
    -------
        list
            [x, y]: list of coordinates of matches that are collinear
            angle: the inclination of the pole
    """
    angles = []
    origins = []
    collinear_points = []
    bin_angles = []

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
                        break                                           # if 1 pair of collinear points is found the iteration can be finished

        if len(bin_angles) > 0 and len(collinear_points) > 0:
            tuple_transform = [tuple(l) for l in collinear_points]              # getting rid of duplicate values in the array by transforming into a tuple
            return [t for t in (set(tuple(i) for i in tuple_transform))], np.average(bin_angles)        # and then creating a set (can only contain unique values) before transforming back to a list
        else:
            return [], 0
    else:
        return [], 0           # if 1 or less points is given as an argument, no collinear points can be found


def get_scale(points, scale_array):
    """
    Gets the most common distance between all matches in 1 frame.
    Which corresponds to the distance between 2 tape stripes.

    Parameters
    ----------
    points : list
        list of all matches in a frame
    scale_array : list
        list of float numbers of scales from previous frames, so an average can be calculated
    Returns
    -------
        float
            the distance between 2 tape stripes
    """
    scale = 0
    if len(points) > 1:
        averages = []
        for a in range(len(points)-1):
            for b in range(a+1, len(points)-1, 1):
                dist = math.sqrt((points[a][0]-points[b][0])**2 + (points[a][1]-points[b][1])**2)
                scale_array.append(dist)

        density, bin_edges = np.histogram(scale_array, bins=np.arange(min(scale_array), max(scale_array) + 10, 10))  # generating a histogram of all found distances
        diff = bin_edges[np.argmax(density)]
        for d in scale_array:
            if diff < d < diff + 10:
                averages.append(d)
        if len(scale_array) > 100:
            del scale_array[:-100]
        scale = np.average(averages)
    else:
        print("Cannot calculate scale with less than 2 matches.")
    return scale


def get_distance(newmatches, oldmatches, angle):
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
    angle : float
        The inclination of the pole (retrieved with the find_collinear function)
    Returns
    -------
        list
            bin_lower: lower limit of where the most common displacements were found.
            bin_upper: upper limit of where the most common displacements were found.

    """
    differences = []
    diff = 0
    if len(newmatches) > 1:
        if len(oldmatches) > 1:
            for newmatch in newmatches:
                for oldmatch in oldmatches:
                    if oldmatch[1] - newmatch[1] > -5:  # displacement should be positive (glacier melts).
                                                        # > 0 is not chosen, because it causes problems in phases with close to 0 melting
                                                        # => small negative displacements are allowed
                        differences.append((oldmatch[-1]-newmatch[-1])/np.cos(angle))   # get displacement projected on pole, append to list to find most common displacement
            if len(differences) > 1:
                density, bin_edges = np.histogram(differences, bins=np.arange(min(differences), max(differences) + threshTracking, threshTracking))  # generating a histogram of all found distances
                diff = bin_edges[np.argmax(density)]                      # determining the bin with the highest amount of occurrences
    if diff != 0:
        return diff, diff+threshTracking
    else:
        return 0, 0


def remove_duplicates(points):
    """
    This function is approximating points that are very close together into 1 single point

    Parameters
    ----------
    points : list
        list of x and y coordinates e.g. [[x1, y1], [x2, y2], ...]

    Returns
    -------
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


def compare_matches(matches, inclination, delta):
    """
    This function compares the last two sets of points in a whole time series of points and calculates the displacement between those.
    If no displacement can be calculated, it recursively tries to compare earlier points with the current one.

    Parameters
    ----------
    matches : list
        A 2d list containing several points for each time step. e.g. [[[x11, y11], [x12, y12]], [[x21, y21], [x22, y22]]]
    inclination : float
        Inclination angle of the pole in rad
    delta : int
        difference between the positions of the images that are compared. default is 1 (last image compared to second last).

    Returns
    -------
        displacement between the frames in px.
        delta to which frame the last frame was compared to.
    """
    displacement = 0
    if delta < 20 and (delta < len(matches)-1 or delta < 2):    # if no matches within the last 20 frames are found, the iteration is stopped.
        current_matches = matches[-1]
        old_matches = matches[-1-delta]
        differences = []

        diff_lower, diff_upper = get_distance(current_matches, old_matches, inclination)   # get most common distance between current and previous frame
        if diff_upper != 0 and (times[frame_nr-delta] in x or frame_nr <=1):
            for new in current_matches:                                     # the following loops visualize the displacement between 2 consecutive frames
                for old in old_matches:
                    if old[1] - new[1] > -5:                                     # only draw when the displacement is negative (glacier melts)
                        difference = (old[1] - new[1])/np.cos(inclination)      # displacement projected to pole axis
                        if diff_lower < difference < diff_upper:  # if displacement is close to most found displacement
                            differences.append(difference)
                            cv2.circle(img, (int(round(old[0]+w/2)), int(round(old[1]+h/2))), 2, (255, 255, 0), cv2.FILLED) # plot current and previous matches for visualization
                            cv2.circle(img, (int(round(new[0]+w/2)), int(round(new[1]+h/2))), 2, (0, 0, 255), cv2.FILLED)
        if len(differences) > 1 and np.average(differences) < 10*delta:
            displacement = np.average(differences)
            print(str(delta) + "-frame displacement found!")
        else:
            displacement, delta = compare_matches(matches, inclination, delta+1)
    else:
        print("No matches found within the last 20 frames!")
        delta = 0
    return displacement, delta


def draw_rectangle(img, points, w, h, color, thickness):
    for point in points:
        cv2.rectangle(img, (int(round(point[0])), int(round(point[1]))), (int(round(point[0]))+w, int(round(point[1]))+h), color, thickness)


def px_to_cm(px, ref_cm, ref_px):
    return px/ref_px*ref_cm


images, times = load_images_from_folder(path)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # turn template into grayscale
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)    # turn template into HSV
template_sat = template_hsv[:, :, 1]                        # extracting only the saturation channel of the HSV template
all_matches = []                                            # list where all matches of all frames will be safed
total_displacement = []                               # list for plotting where all cumulative displacements will be stored
all_lines = []                                              # list for visualizing displacements between consecutive frames
x = []                                                      # x coordinate for plotting
scale_array = []
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
        collinearMatches, pole_inclination = find_collinear(filteredMatches)
        all_matches.append(collinearMatches)
        # draw_rectangle(img, matches, w, h, (150, 250, 255), 1)
        # draw_rectangle(img, filteredMatches, w, h, (0, 255, 0), 1)
        draw_rectangle(img, collinearMatches, w, h, (255, 0, 0), 1)
        scale = get_scale(collinearMatches, scale_array)
    else:
        all_matches.append([])

    if len(all_matches[-1]) > 1:                    # if there are matches in the current frame
        if frame_nr > 0:                            # special case for 1st frame because no displacement can be calculated there.
            frame_disp, delta = compare_matches(all_matches, pole_inclination, 1)
            if frame_disp != 0:
                x.append(times[frame_nr])
                if frame_nr > 1:
                    prev_total = total_displacement[x.index(times[frame_nr-delta])]
                    total_displacement.append(prev_total + px_to_cm(frame_disp, 4, scale))
                else:
                    total_displacement.append(px_to_cm(frame_disp, 4, scale))
                print("Displacement: " + str(px_to_cm(frame_disp, 4, scale)) + " px. Total: " + str(total_displacement[-1]) + " px.")
                disp.setData(x, total_displacement, symbolBrush=('b'))
            else:
                print("No displacements found!")
    else:
        print("No matches found in current frame!")

    cv2.imshow("img", img)
    cv2.waitKey(wait)

cv2.destroyAllWindows()
