import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matchtemplate_funct import match_template
from Meanshift_funct import mean_shift



########################################################################################################
path = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test"           # change path to folder with images
path_first_img = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test\\2019-08-03_07-51.jpg"
first_img = cv2.imread(path_first_img)

threshGray = 0.6                            # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8                             # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)
threshAngle = 1                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
wait = 0                                    # time between every frame in ms, 0 for manual scrolling
h, w = 40,80
########################################################################################################

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

template = first_img[320:320 + h, 430:430 + w]

images = load_images_from_folder(path)
matches = match_template(images,template)
print(matches)

c = []
r = []
for m in matches:
    c.append(int(m[0]))
    r.append(int(m[1]))

cv2.rectangle(first_img, (c[0],r[0]), (c[0]+w,r[0]+h), 255,2)
cv2.rectangle(first_img, (c[1],r[1]), (c[1]+w,r[1]+h), 255,2)

cv2.imshow("first img",first_img)

roi1 = first_img[r[0]:r[0] + h, c[0]:c[0] + w]
roi2 = first_img[r[1]:r[1] + h, c[1]:c[1] + w]
track1 = c[0],r[0],w,h
track2 = c[1],r[1],w,h
print(track1)
cv2.imshow("roi1",roi1)
cv2.imshow("roi2",roi2)

mean_shift(roi1,roi2,images,track1,track2)

cv2.waitKey(wait)

cv2.destroyAllWindows()