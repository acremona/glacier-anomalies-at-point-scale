import cv2
import numpy as np
import os
import math
import pyqtgraph as pg
import matplotlib.pyplot as plt
import datetime
from pyqtgraph.Qt import QtGui


def load_images_from_folder(folder, first_date, end_date=None):
    """
    function loading all image files inside a specified folder. File names must contain date and time with delimiters - or _

    Parameters
    ----------
    folder : string
        path of the folder (string). can be a relative or absolute path.

    Returns
    -------
    images : list
        list of opencv images
    times : list of float
        list of time differences to the first image in hours. e.g. [0, 0.5, 1.0, ...]
    """
    print("Images loading...")
    images = []
    times = []
    #first_date = '2020-07-26_06-07'

    for filename in sorted(os.listdir(folder)):
        if filename < first_date:
            continue

        if end_date is not None:
            if filename > end_date:
                continue

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
    points : list of tuple of float
        list of x and y coordinates e.g. [(x1, y1), (x2, y2), ...]

    Returns
    -------
    collinear_points : list of tuple of float
        list of coordinates of all matches that are collinear (on a straight line)
    angle : float
        the inclination of the pole. returns 0 if no collinear matches.
    """
    angles = []
    origins = []
    collinear_points = []
    bin_angles = []
    bin_origins = []
    result = []

    if len(points) > 1:
        for a in range(len(points)):
            if a < len(points)-1:
                for b in range(a+1, len(points), 1):
                    dx = points[b][0] - points[a][0]       # getting distance in x direction
                    dy = points[b][1] - points[a][1]       # getting distance in y direction

                    if dy != 0:
                        angle = np.arctan(dx/dy)            # getting the angle of the connecting line between the 2 points
                        if abs(angle) < 35*np.pi/180:
                            origins.append(points[b][0]+(h_tot - points[b][1])*np.tan(angle))     # save angle and origin to a list, so the most common one can be picked later
                            angles.append(angle * 180 / np.pi)
        if len(angles) > 0:
            if len(set(angles)) > 1:
                density, bin_edges = np.histogram(angles, bins=np.arange(min(angles), max(angles) + threshAngle, threshAngle))     # generating a histogram of all found angles
                found_angle = bin_edges[np.argmax(density)]*np.pi/180         # choose the highest density of calculated angles
                density, bin_edges = np.histogram(origins, bins=np.arange(min(origins), max(origins) + 10, 10))  # generating a histogram of all found angles
                found_origin = bin_edges[np.argmax(density)]  # analog to angles
            else:
                found_angle = angles[0]
                found_origin = origins[0]
        else:
            found_angle = 0
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
                    origin = point_b[0]+(h_tot - point_b[1])*np.tan(angle)
                    if found_angle <= angle <= found_angle + threshAngle*np.pi/180 and found_origin <= origin <= found_origin + 20:  # if the angle is close to the most common angle and the same for the origin, the match is considered to be on the pole
                        collinear_points.append(point_a)
                        bin_angles.append(angle)
                        bin_origins.append(origin)
                        break                                           # if 1 pair of collinear points is found the iteration can be finished

        if len(bin_angles) > 0:
            o = np.average(bin_origins)
            an = np.average(bin_angles)
            if visualize:
                cv2.line(res_img, (int(round(o)), h_tot), (int(round(o-h_tot*np.tan(an))), 0), (0, 0, 255), 2)
            tuple_transform = [tuple(l) for l in collinear_points]                    # getting rid of duplicate values in the array by transforming into a tuple
            result = [t for t in (set(tuple(i) for i in tuple_transform))], an        # and then creating a set (can only contain unique values) before transforming back to a list

    if len(result) > 0:
        return result
    else:
        return [], 0           # if 1 or less points is given as an argument, no collinear points can be found


def get_scale(points, scale_array):
    """
    Gets the most common distance between all matches in 1 frame,
    Which corresponds to the distance between 2 tapes.

    Parameters
    ----------
    points : list of tuple of float
        list of all matches in a frame
    scale_array : list of float
        list of float numbers of scales from previous frames, so an average can be calculated

    Returns
    -------
    scale : float
        the distance between 2 tapes in px.
    """
    scale = rough_scale

    if len(points) > 1:
        averages = []
        current_scale_array = []
        for a in range(len(points)-1):
            for b in range(a+1, len(points), 1):
                dist = math.sqrt((points[a][0]-points[b][0])**2 + (points[a][1]-points[b][1])**2) #pythagoras to get distance
                current_scale_array.append(dist)
                scale_array.append(dist)
        if frame_nr > 50:
            for a in current_scale_array:
                if a > 1.5*rough_scale and (a % rough_scale < 10 or a % rough_scale > rough_scale-10): #this also adds all distances divided by 2, if the distance is a multiple of the expected scale
                                                                                                       #a.k.a. the distance between 3 tape stripes divided by 2 is the distance between 2 tapes
                                                                                                       #increases stability of histogram, especially when threshold is set very high
                    scale_array.append(a/2)

        if len(set(scale_array)) > 1 and min(scale_array) < 80:
            density, bin_edges = np.histogram(scale_array, bins=np.arange(min(scale_array), 80, 20))  # generating a histogram of all found distances
            try:
                diff = bin_edges[np.argmax(density)]
            except:
                diff = rough_scale #if no maximum in the histogram can be found, the scale equals the scale from the last image

            for d in scale_array:
                if diff <= d <= diff + 15: #if scale is within maximum bin
                    averages.append(d)
            if len(averages) > 0:
                scale = np.average(averages) #increases accuracy of scale
                # print(scale)
            if len(scale_array) > 200: # increases stability of scale calculation. Instead of calculating scales from all matches within the current image
                                       # calculate scales from the last 200 matches (corresponds to 2-4 hours of images) refer to report for more detail.
                del scale_array[:-200] # if the scale array has more than 200 matches, delete the first values until only 200 values are left
    else:
        print("Cannot calculate scale with less than 2 matches.")
    return scale


def get_distance(newmatches, oldmatches, angle):
    """Detects the most common distance between 2 sets of points. This function is beneficial because all tapes
    on the pole are supposed to move the same amount of distance between 2 frames. Therefore, if as input all matches
    in the old frame and all matches in the new frame are chosen, the most common distance between all combination of points
    will be the displacement of the pole.
    To be more precise, the distance projected to the pole is calculated (hence the pole inclination as input).

    Parameters
    ----------
    newmatches : list of tuple of float
        List of point coordinates (x, y)
    oldmatches : list of tuple of float
        List of point coordinates (x, y)
    angle : float
        The inclination of the pole (retrieved with the find_collinear function)

    Returns
    -------
    bin_lower : float
        lower limit of where the most common displacements were found.
    bin_upper : float
        upper limit of where the most common displacements were found.
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
                        distance = (oldmatch[-1]-newmatch[-1])/np.cos(angle) # get displacement projected on pole
                        if distance != 0:
                            differences.append(distance)   #append to list to find most common displacement
            if len(set(differences)) > 1:
                density, bin_edges = np.histogram(differences, bins=np.arange(min(differences), max(differences) + threshTracking, threshTracking))  # generating a histogram of all found distances
                diff = bin_edges[np.argmax(density)]                      # determining the bin with the highest amount of occurrences
    if diff != 0:
        return diff, diff+threshTracking
    else:
        return 0, 0


def remove_duplicates(points):
    """
    This function is approximating points that are very close together into 1 single point.
    If 2 or more points are close together (defined by a threshhold in px), the average x and y coordinates of all those points determine the result point.

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


def compare_matches(matches, inclination, delta):
    """
    This function compares the last two sets of points in a whole time series of points and calculates the displacement between those.
    If no displacement can be calculated, it recursively tries to compare earlier points with the current one.

    Parameters
    ----------
    matches : list of tuple of float
        A 2d list containing several points for each time step. e.g. [[[x11, y11], [x12, y12]], [[x21, y21], [x22, y22]]]
    inclination : float
        Inclination angle of the pole in rad
    delta : int
        difference between the positions of the images that are compared. default is 1 (last image compared to second last).
        delta is used to recursively iterate through past images to find good matches.

    Returns
    -------
    displacement : float
        displacement between the frames in px.

    delta : int
        end point of the recursive iteration. shows to which frame the last frame was compared to. Example: If the last frame is compared to the second last delta = 1.
        delta is used to calculate the total displacement in a later step of the algorithm. The information is needed, so that the found displacement of the current time step can be added to the correct value of the cumulative displacement.
    """
    displacement = 0
    if delta < 40 and (delta < len(matches)-1 or delta < 2):    # if no matches within the last 20 frames are found, the iteration is stopped.
        current_matches = matches[-1]
        old_matches = matches[-1-delta]
        differences = []

        diff_lower, diff_upper = get_distance(current_matches, old_matches, inclination)   # get most common distance between current and previous frame
        if diff_upper != 0 and (times[frame_nr-delta] in x or frame_nr <=1):
            for new in current_matches:                                     # the following loops visualize the displacement between 2 consecutive frames
                # h_new, s_new, v_new = get_colour(img, new[0], new[1], h, w)
                for old in old_matches:
                    if old[1] - new[1] > -5:                                     # only draw when the displacement is negative (glacier melts)
                        difference = (old[1] - new[1])/np.cos(inclination)      # displacement projected to pole axis
                        # h_old, s_old, v_old = get_colour(images[frame_nr-delta], old[0], old[1], h, w)
                        if diff_lower <= difference <= diff_upper:  # if displacement is close to most found displacement
                            differences.append(difference)
                            if visualize:
                                cv2.circle(res_img, (int(round(old[0]+w/2)), int(round(old[1]+h/2))), 2, (255, 255, 0), cv2.FILLED) # plot current and previous matches for visualization
                                cv2.circle(res_img, (int(round(new[0]+w/2)), int(round(new[1]+h/2))), 2, (0, 0, 255), cv2.FILLED)
        if len(differences) > 0 and np.average(differences) < 10*delta:
            displacement = np.average(differences)
            # print(str(delta) + "-frame displacement found!")
        else:
            displacement, delta = compare_matches(matches, inclination, delta+1)
    else:
        print("No matches found within the last 20 frames!")
        delta = 0
        print('lost')
    return displacement, delta


# def get_colour(image, x, y, h, w):
#     x_round = int(round(x))
#     y_round = int(round(y))
#     roi = image[y_round+5:y_round+h+-5, x_round+3:x_round+w-3]
#     channels = roi.transpose(2,0,1).reshape(3,-1)
#     hue = np.median(channels[0])
#     sat = np.median(channels[1])
#     val = np.median(channels[2])
#     return hue, sat, val
#
#
# def compare_colors(hue1, sat1, val1, hue2, sat2, val2):
#     diff_colors = True
#     if sat1 > 40 and sat2 > 40:
#         if abs(hue1-hue2) < 30 or abs(hue1-hue2) > 150:
#             diff_colors = False
#     else:
#         if sat1 < 40 and sat2 < 40 and abs(val1-val2) < 50:
#             diff_colors = False
#     return diff_colors


def draw_rectangle(image, points, w, h, color, thickness):
    """
    Draws a rectangles for a set of points. This function is mainly for visualizing and debugging purposes.

    Parameters
    ----------
    img : opencv_image
        The image where the recangles should be drawn
    points : list of tuple of float
        set of x and y coordinates. e.g. [(200, 100), (120, 150), ...]
    w : int
        width of the rectangles
    h : int
        height of the rectangles
    color : tuple of int
        color tuple in BGR
    thickness : int
        thickness of the drawn rectangle borders
    """
    for point in points:
        cv2.rectangle(image, (int(round(point[0])), int(round(point[1]))), (int(round(point[0]))+w, int(round(point[1]))+h), color, thickness)


def clean_scales(daily_disp, conversion_factors, boxlength):
    """
    This function makes the calculated conversion factors smoother by applying a convolution function (floating average). This way, outliers (errors)
    are removed. From the smoothened conversion factors and displacement rates, a new cumulative displacement array is calculated.

    Parameters
    ----------
    daily_disp : list of tuple of float
        list of displacement rates between frames. The first value is the displacement in px, the second value is the relative index to wich image the current one was compared with (this is needed for calculation of cumulative displacements)
        Example: if the displacements of the current frame was calculated by comparing the current frame with the previous one, the index is 1.
        If the previous frame did not contain any matches and the displacement was calculated by comparing the current frame with one frame earlier than the previous the index is 2.
    conversion_factors : list of float
        list of calculated distances between two tapes from every frame. Same length as daily_disp
    boxlength
        amount of values that should influence the floating average. Must be smaller than half of the length of conversion_factors.
    Returns
    -------
    clean_displacements : list of float
        list of cumulative displacements between frames in cm with smoothened conversion factor
    scale_smooth : list of float
        smoothened conversion factors after convolution
    """
    clean_displacements = []
    for i in range(len(conversion_factors)):
        if abs(rough_scale-conversion_factors[i])>15: # fixed hardcoded filter: if scale is 15px bigger or smaller than average of list, it is replaced by average value
            conversion_factors[i] = rough_scale

    if len(daily_disp) > 2*boxlength:
        box = np.ones(boxlength) / boxlength
        edge_correction = [rough_scale for i in range(boxlength)] #without edge correction the first and last scales would be 0 after convolution
                                                                  #thus, a certain amount of values equalling the average of the list are added at the beginning and end of the list
                                                                  #after convolution, the added values are removed again
        scale_temp = conversion_factors.copy()
        scale_temp[:0] = edge_correction # add values at beginning
        scale_temp.extend(edge_correction) # add values at end of list
        scale_smooth = np.convolve(scale_temp, box, mode='same') # convolution function
        scale_smooth = scale_smooth[boxlength:-boxlength] # remove values for edge correction again.
    else:
        scale_smooth = conversion_factors


    for j in range(len(daily_disp)): # this loop calculates cumulative displacements in cm.
        if j == 0: # case for the very first displacement
            clean_displacements.append(px_to_cm(daily_disp[j][0], tape_spacing, scale_smooth[j]))
        else:
            prev_total = clean_displacements[daily_disp[j][1]]
            clean_displacements.append(prev_total+px_to_cm(daily_disp[j][0], tape_spacing, scale_smooth[j]))
    return clean_displacements, scale_smooth


def px_to_cm(px, ref_cm, ref_px):
    """
    Converts a number from px to cm according to a defined scale.

    Parameters
    ----------
    px : float
        distance in px that is to be converted.
    ref_cm : float
        reference distance in cm (e.g. distance between 2 tape stripes).
    ref_px : float
        reference distance in px (same reference as ref_cm)

    Returns
    -------
    distance : float
        converted distance in cm.
    """
    return px/ref_px*ref_cm


def matchTemplate_hist(folder_path, template_path, thresh,first_date, end_date=None, wait=1, vis=False, plotting=False, csv=True):
    """
    Main function for Algorithm 1 (MatchTemplate with Histograms).

    Parameters
    ----------
    folder_path : str
        String with path to folder where the image series is located. can be an absolute or relative path
    template_path : str
        String with path to the image that should be used as template (image of one single tape)
    thresh : float
        Threshold for filtering matchTemplate. Recommended value is 0.7 - 0.75. See technical report for further info.
    wait : int, optional
        waiting time between 2 consecutive images if the calculation is visualized in [ms]. default is 1 ms.
    vis : bool, optional
        visualizes calculation (time-lapse with indicators). default is False
    plotting : bool, optional
        plots the calculation in real-time (parallel to vis). default is False. Note that real-time displacements are before post-processing, so smoothing of conversion factors with convolution is not applied yet!
        Requires PyQt5 library
    csv : bool, optional
        saves all output data into a csv file. default is True
    Returns
    -------
    x-axis : list of float
        time values in hours for every image
    total_displacement : list of float
        total cumulative displacements for every image in cm. same length as x-axis
    smooth_scales : list of float
        distances between two tapes in px for every image. same length as x-axis
    """

    global h, w, rough_scale, tape_height, tape_spacing, threshTracking, threshDuplicate, threshAngle, frame_nr, h_tot, w_tot, x, res_img, visualize, times
    ########################################################################################################
    # Warning: These thresholds are used for various filtering/calculation operations. They could be     ###
    #          changed in theory, but the algorithm is only tested for those default values listed below ###
    ########################################################################################################
    template = cv2.imread(template_path)  # change to path of RoI
    threshGray = thresh                         # threshold to match template in grayscale, 1 for perfect match, 0 for no match
    threshSat = thresh                          # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
    threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)
    threshAngle = 2                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
    threshTracking = 3                          # threshold to catch match from last frame (in pixel)
    tape_height = 2.0                           # height of tape in cm
    tape_spacing = 4.0                          # distance between two tapes in cm (center to center)
    ########################################################################################################
    images, times = load_images_from_folder(folder_path, first_date, end_date)
    print('ooo', times)
    visualize = vis
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # turn template into grayscale
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)    # turn template into HSV
    template_sat = template_hsv[:, :, 1]                        # extracting only the saturation channel of the HSV template
    template_canny = cv2.Canny(template_sat, 50, 100)
    h, w = template_gray.shape                                  # get width and height of template
    contours, _ = cv2.findContours(template_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rough_scale = 0
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        _, _, _, rough_scale = cv2.boundingRect(contour)
    if h*2/3 < rough_scale < h:
        rough_scale = rough_scale/tape_height*tape_spacing
    else:
        rough_scale = h/tape_height*(tape_spacing-1)

    all_matches = []                                            # list where all matches of all frames will be safed
    total_displacement = []                                     # list for plotting where all cumulative displacements will be stored
    past_mean_disp = []
    x_datetime = []
    daily_displacements = []
    x = []                                                      # x coordinate for plotting
    scale_array = []
    all_scales = []
    if plotting:
        win = pg.GraphicsWindow()                                   # initialize plotting
        pw = win.addPlot()
        pw.addLegend()

        label_style = {'color': '#EEE', 'font-size': '18pt'}
        pw.setLabel('bottom', "Time [Hours]", **label_style)
        pw.setLabel('left', "Daily mass balance [cm]", **label_style)

        font = QtGui.QFont()
        font.setPixelSize(20)
        pw.getAxis("bottom").tickFont = font

        pw.getAxis("bottom").setStyle(tickFont=font)
        pw.getAxis("left").setStyle(tickFont=font)

        #pw.setAxisItems({'bottom': pg.DateAxisItem()})

        # pc = win.addPlot()
        disp = pw.plot()
        #disp.setData([0], [5], symbolBrush=('r'))
        # sc = pc.plot()
        # st_n = pc.plot()
        # st_p = pc.plot()
    for frame_nr, img in enumerate(images):
        # print("========== Frame "+str(frame_nr)+" ==========")
        matches = []
        if visualize:
            res_img = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn image into grayscale
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # turn image into HSV
        sat_img = hsv_img[:, :, 1]                        # extracting only the saturation channel of the HSV image
        h_tot, w_tot = gray_img.shape

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
            if visualize:
                # draw_rectangle(img, matches, w, h, (150, 250, 255), 1)
                # draw_rectangle(img, filteredMatches, w, h, (0, 255, 0), 1)
                draw_rectangle(res_img, collinearMatches, w, h, (255, 0, 0), 1)
            collinearMatches.sort(key=lambda y: int(y[1]))
            scale = get_scale(collinearMatches, scale_array)
        else:
            all_matches.append([])

        if len(all_matches[-1]) > 1:                    # if there are matches in the current frame
            if frame_nr > 0:                            # special case for 1st frame because no displacement can be calculated there.
                frame_disp, delta = compare_matches(all_matches, pole_inclination, 1)
                if frame_disp != 0:
                    print(frame_disp)
                    x.append(times[frame_nr])
                    all_scales.append(scale)
                    rough_scale = np.median(all_scales)
                    if frame_nr > 1:
                        prev_total = total_displacement[x.index(times[frame_nr-delta])]
                        total_displacement.append(prev_total + px_to_cm(frame_disp, tape_spacing, scale))
                        daily_displacements.append((frame_disp, x.index(times[frame_nr-delta])))
                    else:
                        total_displacement.append(px_to_cm(frame_disp, tape_spacing, scale))
                        daily_displacements.append((frame_disp, 0))
                    # print("Displacement: " + str(px_to_cm(frame_disp, tape_spacing, scale)) + " cm. Total: " + str(total_displacement[-1]) + " cm.")
                    # #tweet mod=True
                    # #tweet if mod:
                    # #tweet   total_displacement[-1] = total_displacement[-1] + 0.0035

                    if plotting:
                        # x_datetime.append(datetime.datetime(year=2022, month=7, day=9, hour=5, minute=32) + datetime.timedelta(hours=times[frame_nr])) #'2022-07-09_05-32'
                        # print(x_datetime)
                        # #tweet if x[-1]%24 == 0:
                        # #tweet   print('x', x[-1], datetime.datetime(year=2022, month=7, day=13, hour=5, minute=32) + datetime.timedelta(hours=times[frame_nr]), 'tot disp', total_displacement[-1])
                        # #tweet   total_displacement[-1]= 0
                        # #tweet   #total_displacement[-2] = -(total_displacement[-1] - total_displacement[-2])/2
                        # #tweet else:
                        # #tweet   if x[-1] % 24 <= 0.3:
                        # #tweet       print('x', x[-1], 'tot disp', total_displacement[-1])
                        # #tweet      total_displacement[-1]= 0


                        # #tweet past_mean_disp.append(5.81)
                        # #tweet pw.addLegend()
                        # #tweet disp = pw.plot()
                         disp.setData(x, total_displacement, symbolBrush=('b'))
                        # #tweet pw.addLegend()
                        # #tweet disp = pw.plot()
                        # #tweet disp.setData(x, past_mean_disp, pen='r')
                        # sc.setData(x, all_scales, symbolBrush=('g'))
                        # std = np.std(all_scales)
                        # avg = np.average(all_scales)
                        # st_n.setData([0, x[-1]], [avg - std, avg - std])
                        # st_p.setData([0, x[-1]], [avg + std, avg + std])

                else:
                    print("No displacements found!")
        else:
            print("No matches found in current frame!")

        if visualize:
            # #tweet if frame_nr==0:
            # #tweet    cv2.imshow("img", res_img)
            # #tweet   cv2.waitKey(0)

            # #tweet cv2.putText(res_img, str(datetime.datetime(year=2022, month=7, day=13, hour=5, minute=32) + datetime.timedelta(hours=times[frame_nr]))
            # #tweet           , (5, 700), cv2.FONT_HERSHEY_SIMPLEX,
            # #tweet           1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("img", res_img)
            cv2.waitKey(wait)

            # #tweet if datetime.datetime(year=2022, month=7, day=13, hour=5, minute=32) + datetime.timedelta(hours=times[frame_nr]) > datetime.datetime(year=2022, month=7, day=19, hour=15, minute=00) and datetime.datetime(year=2022, month=7, day=13, hour=5, minute=32) + datetime.timedelta(hours=times[frame_nr]) < datetime.datetime(year=2022, month=7, day=19, hour=15, minute=40):
            # #tweet   total_displacement[-1] = total_displacement[-2]
            # #tweet   print('hi')

    # cv2.destroyAllWindows()
    print(total_displacement[-1])
    total_displacement, smooth_scales = clean_scales(daily_displacements, all_scales, 100)
    #smooth_scales = all_scales

    if csv:
        conversion_factors = [tape_spacing / val for val in smooth_scales]
        displacement_rate = [total_displacement[i + 1] - total_displacement[i] for i in range(len(total_displacement) - 1)]
        displacement_rate[:0] = [total_displacement[0]]
        out = np.vstack((np.array(x), np.array(total_displacement), np.array(displacement_rate), np.array(smooth_scales), np.array(conversion_factors)))
        out_transposed = np.transpose(out)
        np.savetxt("output.csv", out_transposed, comments='', header="time[h],cumulative displacements [cm],displacement rate [cm],distance between two tapes [px], conversion factor [cm/px]", delimiter=",", fmt='%f')
    return x, total_displacement, smooth_scales








