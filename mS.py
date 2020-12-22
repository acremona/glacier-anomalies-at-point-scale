import numpy as np
import cv2
from mT import match_template
from mask import red, yellow, blue, green, black
import pyqtgraph as pg

#############################################################
wait = 1
threshold = -0.002
#############################################################


def mS_same_frame(images, times, a, b, template, h, w):
    """
    This function recalls the matchTemplate and meanShift functions on the same frame to calculate eventual
    offsets of the templates (which will be later corrected).

    Parameters
    ----------
    images: list
        Time series of images.
    times: list
        List of time differences for every image to the first image in hours. e.g. [0, 0.5, 1.0, ...]
    a: float
        Slope of the function used to convert displacements into metric unit.
    b: float
        Y-intercept of the function used to convert displacements into metric unit.
    template: opencv-image
        Template used to identify the tapes.
    h: int
        Height of the track window.
    w: int
        Width of the track window

    Returns
    -------
    dy_list: list
        List of offset value between template and tape in each image.
    """
    #initialization of some variables
    count_no_matches = 0
    count_nomatches_notrack = 0
    x_coor = []
    x = []
    dy_list = []
    std_list = []
    dy_cum = []
    win = pg.GraphicsWindow()                                                                      # initialize plotting
    pw = win.addPlot()
    pw1 = win.addPlot()
    disp = pw.plot()
    disp1 = pw1.plot()

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0)                            # definition of max iteration for meanshift function
    kernel1 = np.ones((50, 50))                                                                     # definition of kernel for morphological operation

    for ii, img in enumerate(images):
        matches = match_template(img, template)                                                     # finding matches

        if not matches:
            count_no_matches = count_no_matches + 1

        elif len(matches) > 1:
            roi_hist_list = []
            roi = []
            track_window = []
            for j, mm in enumerate(matches):
                c = int(mm[0])                                                                      # definition of column (coordinate) of matches
                r = int(mm[1])                                                                      # definition of row (coordinate) of matches
                roi.append(img[r:r + h, c:c + w])                                                   # extraction of ROI from the initial window
                track_window.append((c, r, w, h))                                                   # definition of initial track window
                hsv_roi = cv2.cvtColor(roi[j], cv2.COLOR_BGR2HSV)                                   # conversion to hsv colorscale
                mask = yellow(hsv_roi) + red(hsv_roi) + green(hsv_roi) + blue(hsv_roi) + black(hsv_roi) # definition of mask of ROI to track only the color of the tape
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])                      # calculation of the histogram of ROI according to masked color (previous line)
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)                          # normalize it to 255
                roi_hist_list.append(roi_hist)
            ret = np.ones(len(roi))
            window_number = np.count_nonzero(ret)                                                   # definition of number of tapes to track

        else:
            # same procedure as the previous "elif block" but for only one tape/match
            c = int(matches[0][0])
            r = int(matches[0][1])
            roi = img[r:r + h, c:c + w]
            track_window = (c, r, w, h)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = yellow(hsv_roi) + red(hsv_roi) + green(hsv_roi) + blue(hsv_roi) + black(hsv_roi)
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            roi_hist_list = roi_hist
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)                              # normalize it to 255
            ret = 1
            window_number = 1

        # definition of x coordinates for plotting
        x_coor.append(ii)
        x.append(times[ii])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = yellow(hsv) + red(hsv) + green(hsv) + blue(hsv) + black(hsv)
        if isinstance(matches,list):
            old_y = np.zeros(len(roi))                                                              # preallocation for y location at old step
            new_y = np.zeros(len(roi))                                                              # preallocation for y location at new step
            dst = [None] * len(roi)                                                                 # preallocation for histogram backprojection
            conversion_factor = np.zeros(len(roi))                                                  # preallocation for conversion factor
            if window_number >= 1:
                for i, hsv_ro in enumerate(roi):
                    if ret[i]!=0:
                        dst[i] = cv2.calcBackProject([hsv], [0, 1], roi_hist_list[i], [0, 180, 0, 256], 1)  # calculation of histogram backprojection

                        # Filtering and morphological transformation of backprojection
                        _, dst[i] = cv2.threshold(dst[i], 0.85 * np.max(dst[i]), 255, cv2.THRESH_TOZERO)
                        _, dst[i] = cv2.threshold(dst[i], 0.5 * np.max(dst[i]), 255, cv2.THRESH_BINARY)
                        dst[i] = cv2.dilate(dst[i], kernel1)
                        dst[i] = np.where(mask_hsv == 255, dst[i], 0)

                        old_y[i] = track_window[i][1]                                               # assign y location at old step
                        ret[i], track_window1 = cv2.meanShift(dst[i], track_window[i],term_crit)    # apply meanshift to get the new location of the object/tape
                        x1, y1, w, h = track_window1
                        new_y[i] = y1                                                               # assign y location at new step
                        track_window[i] = track_window1

                        # calculation of conversion factor (eq. 8)
                        conversion_factor[i] = a*((new_y[i]+old_y[i])/2)+b
                        conversion_factor[i] = 1.9 / (conversion_factor[i] * 100)

                window_number = np.count_nonzero(ret)                                               # counting how many tapes are succesfully tracked
                dy = old_y - new_y                                                                  # calculation of displacement
                dy = np.where(ret != 0, dy, np.nan)                                                 # delete dy for tapes which get lost (ret = 0)
                dy = [d * conversion_factor[j] for j, d in enumerate(dy)]                           # converssion to metric unit
                std = np.nanstd(dy)                                                                 # calculation of standard deviation
                dy = np.asarray(dy)
                dy = np.nanmedian(dy)

                if np.isnan(dy):
                    count_nomatches_notrack = count_nomatches_notrack + 1                           # counting frames where all tepes get lost

            else:
                count_nomatches_notrack = count_nomatches_notrack + 1                               # counting frames where all tepes get lost

        else:
            count_nomatches_notrack = count_nomatches_notrack + 1                                   # counting frames where all tepes get lost
            dy = 0

        dy = np.nan_to_num(dy)
        dy_list.append(dy)
        std = np.nan_to_num(std)
        std_list.append(std)

        if ii == 0:
            dy_cum.append(dy)
        else:
            dy_cum.append(dy_cum[ii-1] + dy)

        # plot real time results --> error offset of the tapes with respect to the templates (to then correct it)
        disp.setData(x_coor, dy_list, symbolBrush=('b'))
        disp1.setData(x, dy_cum, symbolBrush=('r'))
        print('dy ', dy, ' pm ', std)
        cv2.imshow('img', img)
        cv2.waitKey(wait)
    print("[info] meanshift ended ...")

    return dy_list


def mS_different_frames(images, times, a, b, template, h, w, dy_cal):
    """
    This function recalls the matchTemplate function in one frame (to find the initial location of the tapes) and
    meanShift to track tapes into the consecutive frame. Displacement of tapes between two frames is calculated and
    cumulated over the hole series.

    Parameters
    ----------
    images: list
        Time series of images.
    times: list
        List of time differences for every image to the first image in hours. e.g. [0, 0.5, 1.0, ...]
    a: float
        Slope of the function used to convert displacements into metric unit.
    b: float
        Y-intercept of the function used to convert displacements into metric unit.
    template: opencv-image
        Template used to identify the tapes.
    h: int
        Height of the track window.
    w: int
        Width of the track window
    dy_cal: list
        Offsets of the templates.

    Returns
    -------
    dy_list: list
        Cumulative displacements for every image in m.
    std_list: list
        Cumulative standard deviation for every image in m.
    count_nomatches_notrack: int
        Number of images for which the displacement could not be calculated (value of 0 is assigned).
    """

    # initialization of some variables
    count_no_matches = 0
    count_nomatches_notrack = 0
    x_coor = []
    x = []
    dy_list = []
    std_list = []
    dy_cum = []
    win = pg.GraphicsWindow()                                                                       # initialize plotting
    pw = win.addPlot()
    pw1 = win.addPlot()
    disp = pw.plot()
    disp1 = pw1.plot()
    roi = None
    dy = 0
    std = 0

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0)                            # definition of max iteration for meanshift function
    kernel1 = np.ones((50, 50))                                                                     # definition of kernel for morphological operation

    for ii, img in enumerate(images):
        # definition of x coordinates for plotting
        x_coor.append(ii)
        x.append(times[ii])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                                                  # converssion to hsv colorscale
        mask_hsv = yellow(hsv) + red(hsv) + green(hsv) + blue(hsv) + black(hsv)                     # mask out tapes

        if isinstance(roi,list):
            # preallocation of some variable
            old_y = np.zeros(len(roi))
            new_y = np.zeros(len(roi))
            dst = [None] * len(roi)
            conversion_factor = np.zeros(len(roi))

            if window_number >= 1:
                for i, hsv_ro in enumerate(roi):
                    if ret[i]!=0:
                        dst[i] = cv2.calcBackProject([hsv], [0, 1], roi_hist_list[i], [0, 180, 0, 256], 1)  # calculation of histogram backprojection

                        # Filtering and morphological transformation of backprojection
                        _, dst[i] = cv2.threshold(dst[i], 0.85 * np.max(dst[i]), 255, cv2.THRESH_TOZERO)
                        _, dst[i] = cv2.threshold(dst[i], 0.5 * np.max(dst[i]), 255, cv2.THRESH_BINARY)
                        dst[i] = cv2.dilate(dst[i], kernel1)
                        dst[i] = np.where(mask_hsv == 255, dst[i], 0)

                        old_y[i] = track_window[i][1]                                                       # assign y location at old step
                        ret[i], track_window1 = cv2.meanShift(dst[i], track_window[i],term_crit)            # apply meanshift to get the new location of the object
                        x1, y1, w, h = track_window1
                        new_y[i] = y1                                                                       # assign y location at new step
                        track_window[i] = track_window1

                        # calculation of conversion factor (eq. 8)
                        conversion_factor[i] = a*((new_y[i]+old_y[i])/2)+b
                        conversion_factor[i] = 1.9 / (conversion_factor[i] * 100)

                window_number = np.count_nonzero(ret)                                                       # counting how many tapes are succesfully tracked
                dy = old_y - new_y                                                                          # calculation of displacement
                dy = np.where(ret != 0, dy, np.nan)                                                         # delete dy for tapes which get lost (ret = 0)
                dy = [d * conversion_factor[j] for j, d in enumerate(dy)]                                   # conversion to metric unit
                std = np.nanstd(dy)
                dy = np.asarray(dy)
                dy = np.nanmedian(dy)
                if np.isnan(dy):
                    count_nomatches_notrack = count_nomatches_notrack + 1
            else:
                count_nomatches_notrack = count_nomatches_notrack + 1
        elif roi is not None:
            # same procedure as the previous "elif block" but for only one tape/match
            if ret != 0:
                dst = cv2.calcBackProject([hsv], [0, 1], roi_hist_list, [0, 180, 0, 256], 1)
                _, dst = cv2.threshold(dst, 0.85 * np.max(dst), 255, cv2.THRESH_TOZERO)
                _, dst = cv2.threshold(dst, 0.5 * np.max(dst), 255, cv2.THRESH_BINARY)
                dst = cv2.dilate(dst, kernel1)
                dst = np.where(mask_hsv == 255, dst, 0)
                old_y = track_window[1]
                ret, track_window1 = cv2.meanShift(dst, track_window, term_crit)
                x1, y1, w, h = track_window1
                new_y = y1
                track_window = track_window1
                conversion_factor = a*((new_y+old_y)/2)+b
                conversion_factor = 1.9/(conversion_factor*100)
                dy = old_y - new_y
                dy = dy * conversion_factor
                std = 0
            else:
                # all tape get lost
                count_nomatches_notrack = count_nomatches_notrack + 1

        #recall matchtemplate and initialize variable
        matches = match_template(img, template)

        if not matches:
            count_no_matches = count_no_matches + 1

        elif len(matches) > 1:
            roi_hist_list = []
            roi = []
            track_window = []
            for j, mm in enumerate(matches):
                c = int(mm[0])                                                                              # definition of column (coordinate) of matches
                r = int(mm[1])                                                                              # definition of row (coordinate) of matches
                roi.append(img[r:r + h, c:c + w])                                                           # extraction of ROI from the initial window
                track_window.append((c, r, w, h))                                                           # definition of initial location of track window
                hsv_roi = cv2.cvtColor(roi[j], cv2.COLOR_BGR2HSV)
                mask = yellow(hsv_roi) + red(hsv_roi) + green(hsv_roi) + blue(hsv_roi) + black(hsv_roi)     # definition of mask of ROI to track only the color of the tape
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])                              # calculation of the histogram of ROI according to masked color (previous line)
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)                                  # normalize it to 255
                roi_hist_list.append(roi_hist)
            ret = np.ones(len(roi))
            window_number = np.count_nonzero(ret)                                                           # definition of number of tapes to track
        else:
            # same procedure as the previous "elif block" but for only one tape/match
            c = int(matches[0][0])
            r = int(matches[0][1])
            roi = img[r:r + h, c:c + w]
            track_window = (c, r, w, h)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = yellow(hsv_roi) + red(hsv_roi) + green(hsv_roi) + blue(hsv_roi) + black(hsv_roi)
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            roi_hist_list = roi_hist
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            ret = 1
            window_number = 1

        dy = np.nan_to_num(dy)
        dy = dy - dy_cal[ii]                                                                                # correct the displacement with offset from calibration

        # threshold is applied to reduce negative displacements
        if dy < threshold:
            dy = 0

        print('dy ', dy, ' pm ', std)
        dy_list.append(dy)
        std = np.nan_to_num(std)
        std_list.append(std)

        # cumulation of displacements
        if ii == 0:
            dy_cum.append(dy)
        else:
            dy_cum.append(dy_cum[ii-1] + dy)

        # plotting real time results
        disp.setData(x_coor, dy_list, symbolBrush=('b'))
        disp1.setData(x, dy_cum, symbolBrush=('r'))
        cv2.imshow('img', img)
        cv2.waitKey(wait)
        
    return dy_list, std_list, count_nomatches_notrack
