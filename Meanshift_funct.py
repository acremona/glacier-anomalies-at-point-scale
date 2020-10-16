import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matchtemplate_funct import match_template
from manual_mask import create_red_mask,create_yellow_mask,create_blue_mask,create_green_mask
from automatic_mask import automatic_mask

#############################################################
path_template = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\template_red.jpg"
template = cv2.imread(path_template)
#############################################################

def mean_shift(roi,images,track_window):
    """Tracks two objects within a series of images.

    Parameters
    ----------
    roi1: ndarray
       Region of interest including the first object (stripe) to track
    roi2: ndarray
        Region of interest including the second object (stripe) to track
    images: list
        List of images within the two objects are tracked
    track_window: tuple:4
        Initial search window for the first object
    track_window2: tuple:4
        Initial search window for the second object

    Returns
    -------
        None
            It have still to be changed!!
    """

    print("[info] meanshift running ...")
    print('matches',len(roi))
    mask_list = []
    hsv_roi_list = []
    roi_hist_list = []
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)
    kernel = np.ones((2, 2))
    kernel1 = np.ones((20, 20))
    ret1 = 1
    ret2 = 1

    for m,ro in enumerate(roi):
        hsv_roi = cv2.cvtColor(ro, cv2.COLOR_BGR2HSV)
        hsv_roi_list.append(hsv_roi)
        # histogram to identify colors in the ROI (masks can then be adapted accordingly)
        hue, saturation, value = cv2.split(hsv_roi)
        mask = create_yellow_mask(hsv_roi)+create_red_mask(hsv_roi)+create_green_mask(hsv_roi)+create_blue_mask(hsv_roi)
        mask_list.append(mask)
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180],[0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
        roi_hist_list.append(roi_hist)
    ret = np.ones((len(roi)))
    dst_list = []
    track_window1 = []
    dy_list = []
    for ii,img in enumerate(images):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = create_yellow_mask(hsv) + create_red_mask(hsv) + create_green_mask(hsv) + create_blue_mask(hsv)
        old_y = np.zeros(len(roi))
        new_y = np.zeros(len(roi))
        print('Tracking...')
        if len(roi)>1:
            for i,(hsv_ro) in enumerate(zip(hsv_roi_list)):
                if (np.sum(ret) !=0): #any(ret) == 0
                    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist_list[i], [0, 180, 0, 256], 1)
                    #cv2.imshow('mask hsv',mask_hsv)
                    # Filtering and morphological transformation of backprojection
                    _, dst = cv2.threshold(dst, 0.85 * np.max(dst), 255, cv2.THRESH_TOZERO)
                    _, dst = cv2.threshold(dst, 0.5 * np.max(dst), 255, cv2.THRESH_BINARY)
                    dst = cv2.dilate(dst, kernel1)
                    dst = np.where(mask_hsv == 255, dst, 0)
                    dst_list.append(dst)
                    old_y[i] = track_window[i][1]
                    ret[i], track_window1 = cv2.meanShift(dst, track_window[i],term_crit)  # apply meanshift to get the new location of the object
                    x1, y1, w, h = track_window1
                    new_y[i] = y1
                    track_window[i] = track_window1
                    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), 255, 2)
                    cv2.putText(img,str(ret[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(dst, (x1, y1), (x1 + w, y1 + h), 255, 2)
                    cv2.putText(img,str(new_y[i]-old_y[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('img', img)
                    #cv2.imshow('dst'+ str(i), dst)

                else:
                    print('Tracking Object Lost...')
                    matches = match_template(img, template)
                    if not matches:
                        pass
                    else:
                        if len(matches)>1:
                            mask_list = []
                            hsv_roi_list = []
                            roi_hist_list = []
                            roi = []
                            track_window = []
                            for j,m in enumerate(matches):
                                c = int(m[0])
                                r = int(m[1])
                                cv2.rectangle(img, (c, r), (c + w, r + h), 255, 2)
                                roi.append(img[r:r + h, c:c + w])
                                track_window.append((c, r, w, h))
                                cv2.imshow("roi", roi[j])
                                hsv_roi = cv2.cvtColor(roi[j], cv2.COLOR_BGR2HSV)
                                hsv_roi_list.append(hsv_roi)
                                mask = create_yellow_mask(hsv_roi) + create_red_mask(hsv_roi) + create_green_mask(hsv_roi) + create_blue_mask(hsv_roi)
                                mask_list.append(mask)
                                cv2.imshow("mask", mask)
                                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
                                roi_hist_list.append(roi_hist)
                            else:
                                print('gino')
                                pass
                    ret = np.ones(len(roi))
        else:
            print(len(roi))

        dy = new_y -old_y
        #dy[(dy>0)] = np.nan
        dy = np.nanmean(dy)
        dy_list.append(dy)
        print('displ',dy_list[ii])
        cv2.waitKey(50)
    print("[info] meanshift ended ...")
    return dy_list