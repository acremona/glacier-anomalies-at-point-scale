import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matchtemplate_funct import match_template
from Meanshift_funct import mean_shift
import pylab
from automatic_mask import automatic_mask



########################################################################################################
path = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test"           # change path to folder with images
path_template = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\template_red.jpg"
path_first_img = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test\\2019-07-27_05-57.jpg"
first_img = cv2.imread(path_first_img)
template = cv2.imread(path_template)
threshGray = 0.6                            # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8                             # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
threshDuplicate = 25                        # threshold to find duplicates within matches (in pixel)
threshAngle = 1                             # threshold to check if matches are on straight line (a.k.a. the pole) in degrees
wait = 0                                    # time between every frame in ms, 0 for manual scrolling
global h, w
h,w = template.shape[0],template.shape[1]
########################################################################################################
print(h,w)
def load_images_from_folder(folder):
    """Function loading all image files inside a specified folder.

    Parameters
    ----------
    folder: string
        Path of the folder. (Can be a relative or absolute path).
    Returns
    -------
        list of opencv images
    """

    print("Images loading...")
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    print("Images loaded")
    return images

def measure(event, x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        p.append((x,y))
        print('p',p)

# calculating conversion px to m
print('[info for the user] You have to select first one point on the top edge of the stripe (leftclick) then one on the bottom (leftclick), then press space botton')
p = []
cv2.namedWindow('first img')
cv2.setMouseCallback('first img',measure)
cv2.imshow('first img',first_img)
cv2.waitKey(0)
cv2.line(first_img,p[0],p[1],(0,0,255),2)
conversion_factor = 1.9/((p[1][1]-p[0][1])*100)
print('1 px =',conversion_factor,'m')
cv2.imshow('first img',first_img)

# actual program starts
images = load_images_from_folder(path)
x_coord = np.arange(len(images))
matches = match_template(first_img,template)

if len(matches)>1:
    roi = []
    track = []
    for i,m in enumerate(matches):
        c = int(m[0])
        r = int(m[1])
        cv2.rectangle(first_img, (c, r), (c + w, r + h), 255, 2)
        roi.append(first_img[r:r + h, c:c + w])
        track.append((c, r, w, h))
else:
    if not matches:
        print('No matches found in the first image. Try with another image with at least one match.')
        cv2.destroyAllWindows()
    else:
        c = int(matches[0][0])
        r = int(matches[0][1])
        cv2.rectangle(first_img, (c, r), (c + w, r + h), 255, 2)
        roi = first_img[r:r + h, c:c + w]
        track = (c, r, w, h)

# Apply Meanshift_funct algorithm
dy = mean_shift(roi,images,track)
dy = np.nan_to_num(dy)
dy = [j * conversion_factor for j in dy]

dycum = np.nancumsum(dy) #multiply with conversion_factor
plt.plot(x_coord,dycum)
plt.title('1001  27.07.19-27.08.19')
plt.xlabel('Frame Number [-]')
plt.ylabel('Displacement in vertical direction [m]')
plt.show()

print('Total displacement:',dycum[-1],'m')

cv2.waitKey(wait)

cv2.destroyAllWindows()