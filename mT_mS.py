import numpy as np
import cv2
import os
import datetime
import math
import imutils
import pandas as pd
from mS import mS_different_frames, mS_same_frame
from mask import red, yellow, blue, green, black
from sklearn.linear_model import LinearRegression

"""
Main code for the Algorithm 2: matchTemplate with meanShift (mT with mS).
In the first step the image series is imported with the function load_good_images_from_folder.py.
In the second step the a and b coefficients used later in the calculation of the conversion factor are calculated with
function find_conversion_factor.py.
Finally the combination of matchTemplate and meanShift is applied to the time series to calculate the displacement. With 
the function mean_shift_same_frame.py offset deriving from a not perfectly centered template are identified (and later
corrected). With the function mean_shift_diff_frames.py the actual displacement is calculated for the hole series of 
images.
"""

########################################################################################################
path = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\1006"                    # path to folder with images
path_cal = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\calibration_1006"    # path to folder with images to calibrateconversion factor
path_template = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\templates"      # path to folder with templates
threshNigth = 50                                                                    # darkness threshold to remove night images
wait = 1                                                                            # time between every frame in ms, 0 for manual scrolling
writer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
########################################################################################################


def load_good_images_from_folder(folder):
    """
    Function loading the image series inside a specified folder. File names must contain date and time with delimiters
    - or _. Aquisition taken during night are filtered out with help of a darkness threshold.

    Parameters
    ----------
    folder : string
        Path of the folder (string). can be a relative or absolute path.

    Returns
    -------
    images : list
        Time series of images.
    times : list
        List of time differences for every image to the first image in hours. e.g. [0, 0.5, 1.0, ...]
    hour: list
        List containing the hour when the image was taken.
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
                hour = dt.hour                                                          # save the hour value
                hours.append(hour)
                if len(times) == 0:                                                     # exception for first frame
                    first_time = dt.timestamp()                                         # convert datetime into seconds
                    times.append(0)
                else:
                    times.append((dt.timestamp()-first_time)/3600)                      # get time difference from first frame in hours
    print("Images loaded")
    return images, times, hours


def find_conversion_factor(img):
    """

    Parameters
    ----------
    img:

    Returns
    -------

    """
    kernel = np.ones((2, 2))                                                            # define kernel used for morphological application
    kernel2 = np.ones((3, 3))
    h_over_w = 0.6                                                                      # ratio height to width of the template

    stripe_area = []
    stripe_center = []
    stripe_center_y = []
    stripe_higth = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                                          #conversion to hsv colorscale
    mask_hsv = yellow(hsv) + red(hsv) + green(hsv) + blue(hsv)  # mask of the tapes
    mask_hsv = cv2.erode(mask_hsv, kernel)                                              # morphological operation to obtain a better mask
    mask_hsv = cv2.dilate(mask_hsv, kernel2)                                            # morphological operation to obtain a better mask

    # find and create contour of the found object
    contours = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for cont in contours:
        area = cv2.contourArea(cont)                                                    # calculation of the area of the tapes
        if area > 600 and area < 2200:
            stripe_area.append(area)
            M = cv2.moments(cont)                                                       # extract center location of the tapes
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            stripe_center.append([cx, cy])
            stripe_center_y.append(cy)
            stripe_higth.append(math.sqrt((h_over_w)*area))                             # calculation of theheight of every tape (eq. 6)
    stripe_higth.sort(key=lambda y: y)
    np.asarray(stripe_higth)
    stripe_higth = np.reshape(stripe_higth,(-1,1))
    stripe_center_y.sort(key=lambda y: -y)
    np.asarray(stripe_center_y)
    stripe_center_y = np.reshape(stripe_center_y,(-1,1))
    if len(stripe_center) > 1 and len(stripe_higth) > 1:
        reg = LinearRegression().fit(stripe_center_y, stripe_higth)                     # linear regression to extrapolate a and b coefficients
        a = reg.coef_.item()
        a = round(a,3)
        b = reg.intercept_.item()
        b = round(b,1)
        return a,b
    else:
        return np.nan, np.nan



# actual program starts
a_list = []
b_list = []

# load image time series, set for calibration of conversion factor and templates
images, times, time = load_good_images_from_folder(path)
cal_set, _, _ = load_good_images_from_folder(path_cal)
templ, _, _ = load_good_images_from_folder(path_template)

# calculation of a and b coefficients within the calibration set of images (images with good lighting conditions)
for cal in cal_set:
        a, b = find_conversion_factor(cal)
        a_list.append(a)
        b_list.append(b)

a = np.nanmedian(a_list)
b = np.nanmedian(b_list)
print(a, b)

print('Lenght of image time series: ',len(images))
x_coord = np.arange(len(images))

for count, tem in enumerate(templ):
    h, w = tem.shape[0], tem.shape[1]

    dy_cal_0 = [0]
    dy_cal = mS_same_frame(images, times, a, b, tem, h, w)
    dy_cal = dy_cal_0 + dy_cal
    dy_cal = dy_cal[0:len(dy_cal)-1]
    dy, std, count_lost = mS_different_frames(images, times, a, b, tem, h, w, dy_cal)
    dy = np.asarray(dy)

    # calculate standard deviation of cumulative sum according to gaussian error propagation (eq. 9)
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

    print('Total displacement :',dycum[-1], ' pm ', stdcum[-1], ' m')
    print('Object could not be tracked in ',count_lost, ' frames ')

    # saving data into excel
    df = pd.DataFrame({'time': times, 'rate': dy, 'std dev': std, 'dy cum': dycum, ' std dev cum': stdcum, 'count lost': count_lost})
    df.to_excel(writer, sheet_name = 'sheet ' + str(count))

writer.save()
cv2.waitKey(1)

cv2.destroyAllWindows()