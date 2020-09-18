import numpy as np
import cv2

# import images
imgname1 = "H_F_2020-07-27_15_44.jpg"
imgname2 = "H_F_2020-07-27_16_04.jpg"
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

# setup initial location of window
c, r, w, h = 430, 180, 80, 40  # simply hardcoded the values (column, row, width, heigth)
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = img1[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # change color from Blue Green Red to hsv type
mask = cv2.inRange(hsv_roi, np.array((20., 100.,100.)), np.array((30.,255.,255.)))  # mask with yellow color
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])                           # kind of characterization of roi
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)                              # normalize it to 255


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# calculation of back projection
hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)                                         # change color from Blue Green Red to hsv type
dst1 = cv2.calcBackProject([hsv1],[0],roi_hist,[0,180],1)                            # create like a mask with higher values where the region in the image is similar to the roi
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
dst2 = cv2.calcBackProject([hsv2],[0],roi_hist,[0,180],1)

# apply meanshift to get the new location of the object
ret1, track_window1 = cv2.meanShift(dst1, track_window, term_crit)
ret2, track_window2 = cv2.meanShift(dst2, track_window, term_crit)
x1,y1,w,h = track_window1
x2,y2,w,h = track_window2

# Draw the rectangle containing the object on image
img1 = cv2.rectangle(img1, (x1,y1), (x1+w,y1+h), 255,2)
img2 = cv2.rectangle(img2, (x2,y2), (x2+w,y2+h), 255,2)

# calculation of the displacement
dx = np.abs(x2-x1)
dy = np.abs(y2-y1)
displ = np.sqrt(np.square(dx)+np.square(dy))

print(displ)

# visualization of results
cv2.imshow('mask',mask)
cv2.imshow('dst1',dst1)
cv2.imshow('dst2',dst2)
cv2.imshow('roi',roi)
cv2.imshow('res1',img1)
cv2.imshow('res2',img2)

cv2.waitKey(0)
cv2.destroyAllWindows()