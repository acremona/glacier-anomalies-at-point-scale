import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matchtemplate_funct import match_template
from manual_mask import create_red_mask,create_yellow_mask
from automatic_mask import automatic_mask

#############################################################
path_template = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\template.jpg"
template = cv2.imread(path_template)
#############################################################

def mean_shift(roi1,roi2,images,track_window,track_window2):
    """Tracks two objects within a series of images.

    :param roi1: ndarray
        Region of interest including the first object (stripe) to track
    :param roi2: ndarray
        Region of interest including the second object (stripe) to track
    :param images: list
        List of images whithin the two objects are tracked
    :param track_window: tuple:4
        Initial search window for the first object
    :param track_window2: tuple:4
        Initial search window for the second object
    :return: None
        It have still to be changed!!
    """
    print("[info] meanshift running ...")
    hsv_roi = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
    hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

    # histogram to identify colors in the ROI (masks can then be adapted accordingly)
    hue, saturation, value = cv2.split(hsv_roi)
    hue2, saturation2, value2 = cv2.split(hsv_roi2)

    mask = automatic_mask(hsv_roi)
    mask2 = automatic_mask(hsv_roi2)

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180],[0, 180])  # kind of characterization of roi !!! provare mask = none
    roi_hist2 = cv2.calcHist([hsv_roi2], [0], mask2, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
    cv2.normalize(roi_hist2, roi_hist2, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

    kernel = np.ones((2, 2))
    kernel1 = np.ones((3, 3))
    ret1 = 1
    ret2 = 1
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if (ret1 != 0 or ret2 !=0):
            print('Tracking...')
            dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
            dst2 = cv2.calcBackProject([hsv], [0, 1], roi_hist2, [0, 180, 0, 256], 1)
            # Filtering and morphological transformation of backprojection
            _, dst = cv2.threshold(dst, 0.85 * np.max(dst), 255, cv2.THRESH_TOZERO)
            #dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
            dst = cv2.erode(dst, kernel)
            dst = cv2.dilate(dst, kernel1)
            _, dst2 = cv2.threshold(dst2, 0.85 * np.max(dst2), 255, cv2.THRESH_TOZERO)
            #dst2 = cv2.morphologyEx(dst2, cv2.MORPH_CLOSE, kernel)
            dst2 = cv2.erode(dst2, kernel)
            dst2 = cv2.dilate(dst2, kernel1)
            ret1, track_window1 = cv2.meanShift(dst, track_window,term_crit)  # apply meanshift to get the new location of the object
            x1, y1, w, h = track_window1
            ret2, track_window3 = cv2.meanShift(dst2, track_window2,term_crit)  # apply meanshift to get the new location of the object
            x3, y3, w, h = track_window3

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), 255, 2)
            cv2.putText(img,str(ret1),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x3, y3), (x3 + w, y3 + h), 255, 2)
            cv2.putText(img, str(ret2), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(dst, (x1, y1), (x1 + w, y1 + h), 255, 2)
            cv2.rectangle(dst2, (x3, y3), (x3 + w, y3 + h), 255, 2)

            # visualization of results
            cv2.imshow('img', img)
            cv2.imshow('dst1', dst)
            cv2.imshow('dst2', dst2)
            cv2.waitKey(0)
        else:
            print('Tracking Object Lost...')
            matches = match_template(img, template)
            c = []
            r = []
            for m in matches:
                c.append(int(m[0]))
                r.append(int(m[1]))

            cv2.rectangle(img, (c[0], r[0]), (c[0] + w, r[0] + h), 255, 2)
            cv2.rectangle(img, (c[1], r[1]), (c[1] + w, r[1] + h), 255, 2)

            roi1 = img[r[0]:r[0] + h, c[0]:c[0] + w]
            roi2 = img[r[1]:r[1] + h, c[1]:c[1] + w]
            track_window = c[0], r[0], w, h
            track_window2 = c[1], r[1], w, h
            cv2.imshow("roi1", roi1)
            cv2.imshow("roi2", roi2)
            hsv_roi = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
            hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
            mask = automatic_mask(hsv_roi)
            mask2 = automatic_mask(hsv_roi2)
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180],[0, 180])  # kind of characterization of roi !!! provare mask = none
            roi_hist2 = cv2.calcHist([hsv_roi2], [0], mask2, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
            cv2.normalize(roi_hist2, roi_hist2, 0, 255, cv2.NORM_MINMAX)
            ret1 = 1
            ret2 = 1
            cv2.waitKey(0)
    print("[info] meanshift ended ...")