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
from Meanshift_funct import mean_shift
from manual_mask import create_red_mask,create_yellow_mask,create_blue_mask,create_green_mask, create_black_mask
from sklearn.linear_model import LinearRegression
import pylab



########################################################################################################
path = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test"           # change path to folder with images
path_template = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\template_1001.jpg"
path_first_img = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test\\2019-06-27_11-59.jpg"
path_measure = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test\\2019-06-27_12-19.jpg"
first_img = cv2.imread(path_first_img)
template = cv2.imread(path_template)
img_mes = cv2.imread(path_measure)
threshGray = 0.6                            # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8                             # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)
threshAngle = 2                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
threshNigth = 50
wait = 0                                    # time between every frame in ms, 0 for manual scrolling
global h, w
h,w = template.shape[0],template.shape[1]
########################################################################################################
print(h,w)


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


def find_conversion_factor(img,h,w):
    kernel = np.ones((1, 4))

    stripe_area = []
    stripe_center = []
    stripe_center_y = []
    stripe_higth = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = create_yellow_mask(hsv) + create_red_mask(hsv) + create_green_mask(hsv) + create_blue_mask(hsv) + create_black_mask(hsv)
    mask_hsv = cv2.erode(mask_hsv, kernel)
    mask_hsv = cv2.dilate(mask_hsv, kernel)

    # find and create contour of the found object
    contours = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 500 and area < 2500:
            stripe_area.append(area)
            M = cv2.moments(cont)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
            stripe_center.append([cx, cy])
            stripe_center_y.append(cy)
            stripe_higth.append(math.sqrt((h/w)*area))
    stripe_higth.sort(key=lambda y: y)
    np.asarray(stripe_higth)
    stripe_higth = np.reshape(stripe_higth,(-1,1))
    stripe_center_y.sort(key=lambda y: -y)
    np.asarray(stripe_center_y)
    stripe_center_y = np.reshape(stripe_center_y,(-1,1))
    reg = LinearRegression().fit(stripe_center_y, stripe_higth)
    print(reg.coef_)
    a = reg.coef_
    a = -0.02
    b = max(stripe_higth)
    b=28
    plt.plot(stripe_center_y,stripe_higth)
    plt.plot(stripe_center_y,stripe_center_y*a+b)
    plt.show()
    cv2.imshow('mask masstab',mask_hsv)
    cv2.imshow('masstab',img)
    cv2.waitKey(0)
    return a,b



# calculating conversion px to m
#print('[info for the user] You have to select first one point on the top edge of the stripe (leftclick) then one on the bottom (leftclick), then press space botton')
#p = []
#cv2.namedWindow('first img')
#cv2.setMouseCallback('first img',measure)
#cv2.imshow('first img',first_img)
#cv2.waitKey(0)
#cv2.line(first_img,p[0],p[1],(0,0,255),2)
#conversion_factor = 1.9/((p[1][1]-p[0][1])*100)
#conversion_factor = 0.0007307692307692307
#print('1 px =',conversion_factor,'m')
#cv2.imshow('first img',first_img)

# actual program starts
images, times, time = load_good_images_from_folder(path)
print('images',len(images))
x_coord = np.arange(len(images))
matches = match_template(first_img,template)

if not matches:
        print('No matches found in the first image. Try with another image with at least one match.')
        cv2.destroyAllWindows()

elif len(matches)>1:
    roi = []
    track = []
    for i,m in enumerate(matches):
        c = int(m[0])
        r = int(m[1])
        cv2.rectangle(first_img, (c, r), (c + w, r + h), 255, 2)
        roi.append(first_img[r:r + h, c:c + w])
        track.append((c, r, w, h))
else:
    c = int(matches[0][0])
    r = int(matches[0][1])
    cv2.rectangle(first_img, (c, r), (c + w, r + h), 255, 2)
    roi = first_img[r:r + h, c:c + w]
    track = (c, r, w, h)

# Apply Meanshift_funct algorithm
a,b = find_conversion_factor(img_mes,h,w)
print(a,b)
dy, std = mean_shift(roi, images, track, time, a, b)
#dy = np.nan_to_num(dy)
#std = np.nan_to_num(std)


plt.errorbar(x_coord, dy, yerr=std, linestyle = 'None', ecolor = 'red', marker = '^')
plt.title('1002  27.07.19-27.08.19')
plt.xlabel('Frame Number [-]')
plt.ylabel('Displacement rates in vertical direction [m]')
plt.show()


plt.hist(dy,bins='auto')
plt.show()

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

dycum = np.nancumsum(dy)

plt.errorbar(times,dycum, yerr=stdcum, linestyle = '-', ecolor = 'red', marker = 'None')
plt.title('1002  27.07.19-27.08.19')
plt.xlabel('Time [-]')
plt.ylabel('Cumulative displacement in vertical direction [m]')
plt.show()

print('Total displacement :',dycum[-1], ' pm ', stdcum[-1], ' m')

workbook = xlsxwriter.Workbook('temp_1001.xlsx')
worksheet = workbook.add_worksheet()

for row, data in enumerate(dycum):
    worksheet.write(row, 0, times[row])
    worksheet.write(row, 1, data)
    worksheet.write(row, 2, stdcum[row])

workbook.close()

cv2.waitKey(1)

cv2.destroyAllWindows()