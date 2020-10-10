import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def create_yellow_mask(image):
    low_yellow = np.array([20., 80., 0.])
    upper_yellow = np.array([50., 255., 255.])
    mask = cv2.inRange(image, low_yellow, upper_yellow)
    return mask

def create_red_mask(image):
    low_red = np.array([0., 80., 0.])
    upper_red = np.array([10., 255., 255.])
    mask = cv2.inRange(image, low_red, upper_red)
    return mask

def create_green_mask(image):
    low_green = np.array([50., 80., 0.])
    upper_green = np.array([70., 255., 255.])
    mask = cv2.inRange(image, low_green, upper_green)
    return mask

def create_blue_mask(image):
    low_blue = np.array([100., 80., 0.])
    upper_blue = np.array([165., 255., 255.])
    mask = cv2.inRange(image, low_blue, upper_blue)
    return mask

def mean_shift(roi1,roi2,images,track_window,track_window2):

    print("[info] meanshift running ...")
    hsv_roi = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
    hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

    # histogram to identify colors in the ROI (masks can then be adapted accordingly)
    hue, saturation, value = cv2.split(hsv_roi)
    hue2, saturation2, value2 = cv2.split(hsv_roi2)

    plt.hist(hue2, bins='auto')
    plt.show()

    mask = create_red_mask(hsv_roi)
    mask2 = create_red_mask(hsv_roi)

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180],[0, 180])  # kind of characterization of roi !!! provare mask = none
    roi_hist2 = cv2.calcHist([hsv_roi2], [0], mask2, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
    cv2.normalize(roi_hist2, roi_hist2, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0)

    kernel = np.ones((2, 2))
    kernel1 = np.ones((5, 5))

    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        dst2 = cv2.calcBackProject([hsv], [0, 1], roi_hist2, [0, 180, 0, 256], 1)

        # Filtering and morphological transformation of backprojection
        _, dst = cv2.threshold(dst, 0.6 * np.max(dst), 255, cv2.THRESH_TOZERO)
        dst = cv2.erode(dst, kernel)
        dst = cv2.dilate(dst, kernel1)
        _, dst2 = cv2.threshold(dst2, 0.6 * np.max(dst2), 255, cv2.THRESH_TOZERO)
        dst2 = cv2.erode(dst2, kernel)
        dst2 = cv2.dilate(dst2, kernel1)

        ret1, track_window1 = cv2.meanShift(dst, track_window,term_crit)  # apply meanshift to get the new location of the object
        x1, y1, w, h = track_window1
        ret2, track_window3 = cv2.meanShift(dst2, track_window2,term_crit)  # apply meanshift to get the new location of the object
        x3, y3, w, h = track_window3

        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), 255, 2)
        cv2.rectangle(img, (x3, y3), (x3 + w, y3 + h), 255, 2)
        cv2.rectangle(dst, (x1, y1), (x1 + w, y1 + h), 255, 2)
        cv2.rectangle(dst2, (x3, y3), (x3 + w, y3 + h), 255, 2)

        # visualization of results
        cv2.imshow('img', img)
        cv2.imshow('dst1', dst)
        cv2.imshow('dst2', dst2)
        cv2.waitKey(0)

    print("[info] meanshift ended ...")
