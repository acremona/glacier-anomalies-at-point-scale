import numpy as np
import cv2
import os
import datetime
import math
import imutils
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
from matchtemplate_funct import match_template
from Meanshift_funct import mean_shift_diff_frames
from meanshift_same_frame import mean_shift_same_frame
from manual_mask import create_red_mask,create_yellow_mask,create_blue_mask,create_green_mask, create_black_mask
from sklearn.linear_model import LinearRegression
import pylab



########################################################################################################
path = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1001"           # change path to folder with images
path_cal = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\calibration_1001"
path_template = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\templates_1"
path_first_img = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1001\\2019-06-27_11-59.jpg"
path_measure = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1001\\2019-06-27_11-59.jpg"
first_img = cv2.imread(path_first_img)
template = cv2.imread(path_template)
img_mes = cv2.imread(path_measure)
threshGray = 0.6                            # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8                             # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)
threshAngle = 2                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
threshNigth = 50
wait = 1                                    # time between every frame in ms, 0 for manual scrolling
#global h, w
#h,w = template.shape[0],template.shape[1]
########################################################################################################


def load_good_images_from_folder(folder):
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
    hours = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))

        gray = cv2.imread(os.path.join(folder, filename),0)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        idx = np.where(hist == hist.max())
        idx = idx[0]
        max_idx = np.max(idx)

        if image is not None and max_idx > threshNigth:
            images.append(image)
            if not 'template' in filename:
                filename = filename.replace('_', '-')
                time = list(map(int, filename.split('.')[0].split('-')))                # remove file ending (eg. .jpg) and split string into a list of y, m, d, h, s
                dt = datetime.datetime(time[0], time[1], time[2], time[3], time[4])     # convert into datetime format
                hour = dt.hour
                hours.append(hour)
                if len(times) == 0:                                                     # exception for first frame
                    first_time = dt.timestamp()                                         # convert datetime into seconds
                    times.append(0)
                else:
                    times.append((dt.timestamp()-first_time)/3600)                      # get time difference from first frame in hours
    print("Images loaded")
    return images, times, hours


#def measure(event, x,y,flags,params):
 #   if event == cv2.EVENT_LBUTTONDOWN:
  #      p.append((x,y))
   #     print('p',p)


def find_conversion_factor(img):
    kernel = np.ones((2, 2))
    kernel2 = np.ones((3, 3))
    h_over_w = 0.6

    stripe_area = []
    stripe_center = []
    stripe_center_y = []
    stripe_higth = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = create_yellow_mask(hsv) + create_red_mask(hsv) + create_green_mask(hsv) + create_blue_mask(hsv)
    mask_hsv = cv2.erode(mask_hsv, kernel)
    mask_hsv = cv2.dilate(mask_hsv, kernel2)

    # find and create contour of the found object
    contours = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 600 and area < 2200:
            stripe_area.append(area)
            M = cv2.moments(cont)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            #cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
            stripe_center.append([cx, cy])
            stripe_center_y.append(cy)
            stripe_higth.append(math.sqrt((h_over_w)*area))
    stripe_higth.sort(key=lambda y: y)
    np.asarray(stripe_higth)
    stripe_higth = np.reshape(stripe_higth,(-1,1))
    stripe_center_y.sort(key=lambda y: -y)
    np.asarray(stripe_center_y)
    stripe_center_y = np.reshape(stripe_center_y,(-1,1))
    if len(stripe_center) > 1 and len(stripe_higth) > 1:
        reg = LinearRegression().fit(stripe_center_y, stripe_higth)
        a = np.asscalar(reg.coef_)#np.ndarray.item()
        a = round(a,3)
        b = np.asscalar(reg.intercept_)
        b = round(b,1)
        #plt.scatter(stripe_center_y,stripe_higth)
        #plt.plot(np.arange(max(stripe_center_y)),a*np.arange(max(stripe_center_y))+b, color = 'orange')
        #plt.xlim(0,max(stripe_center_y))
        #plt.xlabel('y-coord of stripe center [px]')
        #plt.ylabel('stripe height [px]')
        #plt.show()
        #cv2.imshow('gino', img)
        #cv2.imshow('mask', mask_hsv)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return a,b
    else:
        return np.nan, np.nan



# calculating conversion px to m
#print('[info for the user] You have to select first one point on the top edge of the stripe (leftclick) then one on the bottom (leftclick), then press space botton')
#p = []
#cv2.namedWindow('first img')
#cv2.setMouseCallback('first img',measure)
#cv2.imshow('first img',img_mes)
#cv2.waitKey(0)
#cv2.line(img_mes,p[0],p[1],(0,0,255),2)
#cv2.imshow('first img',img_mes)

# actual program starts
a_list = []
b_list = []
#matches = match_template(first_img,template)
#matches2 = match_template(first_img,template2)
#a, b = find_conversion_factor(img_mes,h,w)
cal_set, _, _ = load_good_images_from_folder(path_cal)
templ,_,_ = load_good_images_from_folder(path_template)

#calculation of conversion factor
for cal in cal_set:
        a, b = find_conversion_factor(cal)
        a_list.append(a)
        b_list.append(b)

plt.scatter(np.arange(len(a_list)),a_list)
plt.show()
plt.scatter(np.arange(len(b_list)),b_list)
plt.show()

a = np.nanmedian(a_list)
b = np.nanmedian(b_list)

print(a, b)
#load images
images, times, time = load_good_images_from_folder(path)

# import validation data
validation_data = pd.read_excel("C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1009.xlsx",sheet_name='Final')
time_val = validation_data['time val'].tolist()
rate_val = validation_data['rate val'].tolist()
dy_cum_val = validation_data['dy cum val'].tolist()

print('images',len(images))
x_coord = np.arange(len(images))

writer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')

for count, tem in enumerate(templ):
    h, w = tem.shape[0], tem.shape[1]
    matches = match_template(first_img, tem)

    if not matches:
        print('No matches found in the first image. Try with another image with at least one match.')
        cv2.destroyAllWindows()

    elif len(matches) > 1:
        roi = []
        track = []
        for i, m in enumerate(matches):
            c = int(m[0])
            r = int(m[1])
            #cv2.rectangle(first_img, (c, r), (c + w, r + h), 255, 2)
            roi.append(first_img[r:r + h, c:c + w])
            track.append((c, r, w, h))
    else:
        c = int(matches[0][0])
        r = int(matches[0][1])
        #cv2.rectangle(first_img, (c, r), (c + w, r + h), 255, 2)
        roi = first_img[r:r + h, c:c + w]
        track = (c, r, w, h)


    dy_cal, std_cal, count_lost_cal = mean_shift_same_frame(roi, images, track, time, times, a, b, tem, h, w)
    dy, std, count_lost = mean_shift_diff_frames(roi, images, track, time, times, a, b, tem, h, w)
    dy_cal = np.asarray(dy_cal)
    dy = np.asarray(dy)

    stdcum = []
    for i, s in enumerate(std):
        if i == 0:
            stdcum.append(std[0])
        else:
            err = math.pow(stdcum[i-1],2) + math.pow(std[i],2)*10
            if err != 0:
                stdcum.append(math.sqrt(err))
            else:
                stdcum.append(0)

    dy = dy - dy_cal
    dycum = np.nancumsum(dy)

    #dy_day = reshape(44,:)? poi sommare o c'è un altro modo?
    # plt.errorbar(x_coord, dy, yerr=std, linestyle = 'None', ecolor = 'red', marker = '^')
    # plt.title('1002  27.07.19-27.08.19')
    # plt.xlabel('Frame Number [-]')
    # plt.ylabel('Displacement rates in vertical direction [m]')
    # plt.show()

    # plt.hist(dy,bins='auto')
    # plt.show()

    #plt.errorbar(times,dycum, yerr=stdcum, linestyle = '-', ecolor = 'red', marker = 'None')
    #plt.title('1002  27.07.19-27.08.19')
    #plt.xlabel('Time [-]')
    #plt.ylabel('Cumulative displacement in vertical direction [m]')
    #plt.show()

    print('Total displacement :',dycum[-1], ' pm ', stdcum[-1], ' m')
    print('Object could not be tracked in ',count_lost, ' frames ')

    df = pd.DataFrame({'time': times, 'rate': dy, 'std dev': std, 'dy cum': dycum, ' std dev cum': stdcum, 'count lost': count_lost})
    df.to_excel(writer, sheet_name = 'sheet ' + str(count))

writer.save()
cv2.waitKey(1)

cv2.destroyAllWindows()