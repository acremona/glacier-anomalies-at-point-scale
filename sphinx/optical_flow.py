import numpy as np
import cv2
import imutils

# import images
imgname1 = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\Holfuy_Findel\\H_F_2020-07-27_15_44.jpg"
imgname2 = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\Holfuy_Findel\\H_F_2020-07-27_16_04.jpg"
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

# Apply a gaussian filter to smooth the image
blurred_img1 = cv2.GaussianBlur(img1,(5,5),0)
blurred_img2 = cv2.GaussianBlur(img2,(15,15),0)

# convert to hsv format
hsv1 = cv2.cvtColor(blurred_img1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(blurred_img2, cv2.COLOR_BGR2HSV)

# make a mask of yellow color
low_yellow = np.array([20., 80.,0.])
upper_yellow = np.array([50.,255.,255.])
mask1 = cv2.inRange(hsv1, low_yellow, upper_yellow)
mask2 = cv2.inRange(hsv2, low_yellow, upper_yellow)

# make some morphological transformation to obtain a better mask
#still to do

# find and create contour of the found object
contours1 = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours1 = imutils.grab_contours(contours1)
contours2 = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours2 = imutils.grab_contours(contours2)

#variable initiation which are used in the optical flow algorithm
global point, pont_selected, old_points
pont_selected = False
point = ()
old_points = []

for c1 in contours1:
    area1 = cv2.contourArea(c1)

    if area1 > 1000 and area1 < 2500:
        cv2.drawContours(img1,[c1],-1, (0,255,0),2)
        M1 = cv2.moments(c1)
        cx1 = int(M1["m10"]/M1["m00"])
        cy1 = int(M1["m01"]/M1["m00"])
        cv2.circle(img1,(cx1,cy1),3,(255,255,255),-1)
        pont_selected = True
        old_points.append([[cx1,cy1]])

old_points = np.array(old_points, dtype=np.float32)

for c2 in contours2:
    area2 = cv2.contourArea(c2)

    if area2 > 1000 and area2 < 1700:
      cv2.drawContours(img2,[c2],-1, (0,255,0),2)
      M2 = cv2.moments(c2)
      cx2 = int(M2["m10"]/M2["m00"])
      cy2 = int(M2["m01"]/M2["m00"])
      cv2.circle(img2,(cx2,cy2),3,(255,255,255),-1)

# Optical Flow
old_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
new_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Lucas Kanade params
lk_params = dict(winSize = (10,10),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if pont_selected is True:
    new_point, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params) # why use the gray??

# Select good points
good_new = new_point[status==1]
good_old = old_points[status==1]

#define arrays containing x and y values of points
x_new = np.zeros(np.size(good_new,0))
y_new = np.zeros(np.size(good_new,0))
x_old = np.zeros(np.size(good_new,0))
y_old = np.zeros(np.size(good_new,0))

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    x_new[i] = a
    y_new[i] = b
    x_old[i] = c
    y_old[i] = d
    cv2.circle(img2, (a, b), 3, (0, 255, 0), -1)  # ATTENTION: changing the circle size changes the displacement!!! why?
    cv2.circle(img2, (c, d), 3, (255, 0, 0), -1)


# Displacement with optical flow
dy_lc = np.mean(y_new-y_old)

# Results
print("Displacement Optical flow Lucas Kanade = ", dy_lc, "px")

cv2.imshow('res1',img1)
cv2.imshow('res2',img2)
cv2.imshow('mask',mask1)


cv2.waitKey(0)
cv2.destroyAllWindows()