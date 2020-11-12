import numpy as np
import cv2
from matchtemplate_funct import match_template
from manual_mask import create_red_mask,create_yellow_mask,create_blue_mask,create_green_mask, create_black_mask
import pyqtgraph as pg

#############################################################
path_template = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\template_1001.jpg"
template = cv2.imread(path_template)
wait = 1
#############################################################


def retrack(img, template, count_no_matches):
    h, w = template.shape[0], template.shape[1]
    print(h,w)
    matches = match_template(img, template)

    if not matches:
        print('[info] Cannot find any object')
        count_no_matches = count_no_matches + 1
        print('count no matches', count_no_matches)
        roi = []
        dy = np.nan
        std = 0
        window_number = 0
        return [], [], [], [], dy, std, window_number

    elif len(matches) > 1:
        print('[info] Found ' + str(len(matches)) + ' objects')
        roi_hist_list = []
        roi = []
        track_window = []
        for j, mm in enumerate(matches):
            c = int(mm[0])
            r = int(mm[1])
            roi.append(img[r:r + h, c:c + w])
            track_window.append((c, r, w, h))
            hsv_roi = cv2.cvtColor(roi[j], cv2.COLOR_BGR2HSV)
            mask = create_yellow_mask(hsv_roi) + create_red_mask(hsv_roi) + create_green_mask(hsv_roi) + create_blue_mask(hsv_roi) + create_black_mask(hsv_roi)
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
            roi_hist_list.append(roi_hist)
        ret = np.ones(len(roi))
        window_number = np.count_nonzero(ret)
        dy = np.nan
        std = 0
        return roi, roi_hist_list, ret, track_window, dy, std, window_number

    else:
        print('[info] Found one object')
        c = int(matches[0][0])
        r = int(matches[0][1])
        roi = img[r:r + h, c:c + w]
        track_window = (c, r, w, h)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = create_yellow_mask(hsv_roi) + create_red_mask(hsv_roi) + create_green_mask(hsv_roi) + create_blue_mask(hsv_roi) + create_black_mask(hsv_roi)
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        roi_hist_list = roi_hist
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
        ret = 1
        window_number = 1
        dy = np.nan
        std = 0
        return roi, roi_hist_list, ret, track_window, dy, std, window_number


def mean_shift(roi, images, track_window, hour, a, b):
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
    #initialization of some variables
    count_no_matches = 0
    count_no_track = 0
    check = True
    count_next = 1
    x_coor = []

    dst = []
    dy_list = []
    std_list = []
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0)
    kernel1 = np.ones((50, 50))
    print(roi is list,len(roi))
    print(type(roi))
    if isinstance(roi,list):
        ret = np.ones((len(roi)))
        roi_hist_list = []
        for m,ro in enumerate(roi):
            hsv_roi = cv2.cvtColor(ro, cv2.COLOR_BGR2HSV)
            mask = create_yellow_mask(hsv_roi)+create_red_mask(hsv_roi)+create_green_mask(hsv_roi)+create_blue_mask(hsv_roi) + create_black_mask(hsv_roi)
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180],[0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
            roi_hist_list.append(roi_hist)
    else:
        ret = 1
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = create_yellow_mask(hsv_roi) + create_red_mask(hsv_roi) + create_green_mask(hsv_roi) + create_blue_mask(hsv_roi) +create_black_mask(hsv_roi)
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        roi_hist_list = roi_hist
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255

    window_number = np.count_nonzero(ret)
    for ii, img in enumerate(images):
        x_coor.append(ii)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = create_yellow_mask(hsv) + create_red_mask(hsv) + create_green_mask(hsv) + create_blue_mask(hsv) + create_black_mask(hsv)
        old_y = np.zeros(len(roi))
        new_y = np.zeros(len(roi))
        print('[info] Tracking...')
        if isinstance(roi,list):
            dst = [None] * len(roi)
            conversion_factor = np.zeros(len(roi))
            if window_number >= 1:
                for i, hsv_ro in enumerate(roi):
                    if ret[i]!=0:
                        dst[i] = cv2.calcBackProject([hsv], [0, 1], roi_hist_list[i], [0, 180, 0, 256], 1)
                        # Filtering and morphological transformation of backprojection
                        _, dst[i] = cv2.threshold(dst[i], 0.85 * np.max(dst[i]), 255, cv2.THRESH_TOZERO) #provare a cambiareee
                        _, dst[i] = cv2.threshold(dst[i], 0.5 * np.max(dst[i]), 255, cv2.THRESH_BINARY)
                        dst[i] = cv2.dilate(dst[i], kernel1)
                        dst[i] = np.where(mask_hsv == 255, dst[i], 0)
                        old_y[i] = track_window[i][1]
                        ret[i], track_window1 = cv2.meanShift(dst[i], track_window[i],term_crit)  # apply meanshift to get the new location of the object
                        x1, y1, w, h = track_window1
                        new_y[i] = y1
                        track_window[i] = track_window1
                        conversion_factor[i] = a*(y1+h/2)+b
                        conversion_factor[i] = 1.9 / (conversion_factor[i] * 100)
                        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), 255, 2)
                        #cv2.putText(img,str(ret[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                        #cv2.rectangle(dst[i], (x1, y1), (x1 + w, y1 + h), 255, 2)
                        cv2.putText(img,str(new_y[i]-old_y[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        #cv2.imshow('dst'+str(i), dst[i])
                window_number = np.count_nonzero(ret)
                dy = new_y - old_y
                dy = np.where(ret != 0, dy, np.nan)
                dy = np.where(dy < 1, dy, np.nan)
                dy = [d * conversion_factor[j] for j, d in enumerate(dy)]
                std = np.nanstd(dy)
                dy = np.asarray(dy)
                if hour[ii] < 21 and hour[ii] > 6:
                    if hour[ii] < 15 and hour[ii] > 11:
                        dy = np.where(dy > -0.01, dy, np.nan)
                    else:
                        dy = np.where(dy > -0.005, dy, np.nan)

                dy = np.nanmean(dy)
                print('dy ',dy, ' pm ', std)
            else:
                print('[info] Tracking Object Lost...')
                roi, roi_hist_list, ret, track_window, dy, std, window_number = retrack(img, template, count_no_matches)
                print('count no track ', count_no_track)
                dst = [None] * len(roi)

        elif roi is not None: # if only one matches --> roi is not a list but an image, therefore len(roi)==33 ---> i thinck we could make it more elegant
            conversion_factor = 0
            if ret != 0:
                dst = cv2.calcBackProject([hsv], [0, 1], roi_hist_list, [0, 180, 0, 256], 1)
                _, dst = cv2.threshold(dst, 0.85 * np.max(dst), 255, cv2.THRESH_TOZERO)
                _, dst = cv2.threshold(dst, 0.5 * np.max(dst), 255, cv2.THRESH_BINARY)
                dst = cv2.dilate(dst, kernel1)
                dst = np.where(mask_hsv == 255, dst, 0)
                old_y = track_window[1]
                ret, track_window1 = cv2.meanShift(dst, track_window, term_crit)  # apply meanshift to get the new location of the object
                x1, y1, w, h = track_window1
                new_y = y1
                track_window = track_window1
                conversion_factor = a * (y1 + h / 2) + b
                conversion_factor = 1.9/(conversion_factor*100)
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), 255, 2)
                cv2.putText(img, str(new_y - old_y), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,cv2.LINE_AA)
                dy = new_y-old_y
                dy = dy * conversion_factor
                std = 0
                if dy > 0:
                    dy = 0
                if hour[ii] < 21 and hour[ii] > 6:
                    if dy < -0.007:
                        dy = 0

            else:
                print('Tracking Object Lost...')
                roi, roi_hist_list, ret, track_window, dy, std, window_number = retrack(img, template, count_no_matches)
                dst = None
        else:
            dst = None
            roi, roi_hist_list, ret, track_window, dy, std, window_number = retrack(img, template, count_no_matches)

        if ii != 0 and ii % 20 == 0 or window_number < 3:
            print('gino')
            cv2.putText(img, 'Retracking...', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            print('ooo',len(template))
            matches = match_template(img, template)
            print('frame', ii)

            if not matches:
                print('[info] Cannot find any object')
                count_no_matches = count_no_matches + 1
                print('count no matches', count_no_matches) # vedere se mettere roi = []
            else:
                if len(matches) > 1:
                    print('[info] Found' + str(len(matches)) + 'objects')
                    roi_hist_list = []
                    roi = []
                    track_window = []
                    for j, mm in enumerate(matches):
                        c = int(mm[0])
                        r = int(mm[1])
                        #cv2.rectangle(img, (c, r), (c + w, r + h), 255, 2)
                        roi.append(img[r:r + h, c:c + w])
                        track_window.append((c, r, w, h))
                        hsv_roi = cv2.cvtColor(roi[j], cv2.COLOR_BGR2HSV)
                        mask = create_yellow_mask(hsv_roi) + create_red_mask(hsv_roi) + create_green_mask(hsv_roi) + create_blue_mask(hsv_roi) + create_black_mask(hsv_roi)
                        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
                        roi_hist_list.append(roi_hist)
                    ret = np.ones(len(roi))
                    window_number = np.count_nonzero(ret)

                else:
                    print('[info] Found one object')
                    c = int(matches[0][0])
                    r = int(matches[0][1])
                    cv2.rectangle(img, (c, r), (c + w, r + h), 255, 2)
                    roi = img[r:r + h, c:c + w]
                    track_window = (c, r, w, h)
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = create_yellow_mask(hsv_roi) + create_red_mask(hsv_roi) + create_green_mask(hsv_roi) + create_blue_mask(hsv_roi) + create_black_mask(hsv_roi)
                    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                    roi_hist_list = roi_hist
                    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize it to 255
                    cv2.rectangle(img, (c, r), (c + w, r + h), 255, 2)
                    ret = 1
                    window_number = 1
        dy = np.nan_to_num(dy)
        dy_list.append(dy)
        std = np.nan_to_num(std)
        std_list.append(std)
        cv2.imshow('img', img)
        cv2.waitKey(wait)
    print("[info] meanshift ended ...")
    return dy_list, std_list