import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


""" This programs takes two images and apply the meanshift algorithm to it to calculate the displacement of the tape stripes."""

def load_images_from_folder(folder):
    """
    function loading all image files inside a specified folder.

    :param folder: path of the folder (string). can be a relative or absolute path.
    :return: list of opencv images
    """
    print("Images loading...")
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    print("Images loaded")
    return images

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

def create_blue_mask(image):
    low_blue = np.array([100., 80., 0.])
    upper_blue = np.array([165., 255., 255.])
    mask = cv2.inRange(image, low_blue, upper_blue)
    return mask

# import images
path = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test"           # change path to folder with images
path_first_img = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test\\2019-08-03_07-51.jpg"
first_img = cv2.imread(path_first_img)
images = load_images_from_folder(path)

# setup initial location of window
c, r, w, h = 430, 280, 80, 120  # simply hardcoded the values (column, row, width, heigth)
track_window = (c, r, w, h)

cv2.rectangle(first_img, (c,r), (c+w,r+h), 255,2)
cv2.imshow("fisrt image", first_img)

# set up the ROI for tracking
roi = first_img[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)                                                                                            # extracting only the hue channel of the HSV template

hue , saturation, value = cv2.split(hsv_roi)

plt.hist(hue,bins='auto')
plt.show()

mask = create_red_mask(hsv_roi) + create_yellow_mask(hsv_roi)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])                           # kind of characterization of roi !!! provare mask = none
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)                              # normalize it to 255

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


for img in images:
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret1, track_window1 = cv2.meanShift(dst, track_window, term_crit)              # apply meanshift to get the new location of the object
    x1,y1,w,h = track_window1

    cv2.rectangle(img, (x1,y1), (x1+w,y1+h), 255,2)

    # visualization of results
    cv2.imshow('mask',mask)
    cv2.imshow('dst1',dst)
    cv2.imshow('roi',roi)
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()