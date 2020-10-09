import cv2
import imutils
import numpy as np
import os


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

def find_collinear(points):
    """
    This function searches for points that are collinear (on 1 straight line). If there are several lines, the one with
    the most points on it is chosen.

    :param points: list of x and y coordinates e.g. [[x1, y1], [x2, y2], ...]
    :return: list of x and y coordinates of points that are collinear.
    """
    angles = []
    origins = []
    collinear_points = []
    if len(points) > 1:
        for point_a in points:
            for point_b in points:
                dx = abs(point_a[0] - point_b[0])       # getting distance in x direction
                dy = abs(point_a[1] - point_b[1])       # getting distance in y direction

                if dy > 0 and dx > 0:
                    angle = np.arctan(dx/dy)      # getting the angle of the connecting line between the 2 points
                    angles.append(angle*180/np.pi)
                    if angle < 45*np.pi/180:
                        origins.append(point_b[0]-point_b[1]*np.tan(angle))

        density, bin_edges = np.histogram(angles, bins=100)     # generating a histogram of all found angles
        found_angle = bin_edges[np.argmax(density)]*np.pi/180         # choose the highest density of calculated angles
        density, bin_edges = np.histogram(origins, bins=100)     # generating a histogram of all found angles
        found_origin = bin_edges[np.argmax(density)]

        for point_a in points:
            for point_b in points:             # 2 loops comparing all points with each other
                dx = abs(point_a[0] - point_b[0])       # getting distance in x direction
                dy = abs(point_a[1] - point_b[1])       # getting distance in y direction
                if dy > 0 and dx > 0:
                    angle = np.arctan(dx/dy)                            # getting the angle of the connecting line between the 2 points
                    origin = point_b[0]-point_b[1]*np.tan(angle)
                    if abs(angle-found_angle) < threshAngle*np.pi/180 and abs(origin-found_origin) < 10:  # if the angle is close to the angle of the chosen line, the point lies on the line
                        collinear_points.append(point_a)
                        break                                           # if 1 pair of collinear points is found the iteration can be finished

        tuple_transform = [tuple(l) for l in collinear_points]              # getting rid of duplicate point in the array by transforming into a tuple
        return [t for t in (set(tuple(i) for i in tuple_transform))]        # and then creating a set (can only contain unique values) before transforming back to a list
    else:
        return points

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

def create_green_mask(image):
    low_green = np.array([50., 80., 0.])
    upper_green = np.array([70., 255., 255.])
    mask = cv2.inRange(image, low_green, upper_green)
    return mask

def create_blue_mask(image):
    low_blue = np.array([100., 80., 0.])
    upper_blue = np.array([140., 255., 255.])
    mask = cv2.inRange(image, low_blue, upper_blue)
    return mask


############################
path = "C:\\Users\\User\\Desktop\\Eth\\MasterIII\\Project\\images_test"
wait = 0
threshAngle = 1
images = load_images_from_folder(path)
############################

for img in images:
    center = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masky = create_yellow_mask(hsv)
    maskr = create_red_mask(hsv)
    maskg = create_green_mask(hsv)
    maskb = create_blue_mask(hsv)

    mask = maskr + masky + maskg +maskb

    kernel = np.ones((5,5))
    kernel1 = np.ones((7,7))

    mask = cv2.erode(mask,kernel)
    mask = cv2.dilate(mask,kernel1)

    cv2.imshow("mask",mask)

    # find and create contour of the found object
    contours = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)


    for c1 in contours:
        approx = cv2.approxPolyDP(c1,0.05*cv2.arcLength(c1,True),True)
        area1 = cv2.contourArea(approx)

        if area1 > 1000 and area1 < 2500 and len(approx) == 4:
            cv2.drawContours(img,[approx],-1, (0,255,0),2)
            M1 = cv2.moments(c1)                                # va bene anche approx invece di c1?
            cx1 = int(M1["m10"]/M1["m00"])
            cy1 = int(M1["m01"]/M1["m00"])
            cv2.circle(img,(cx1,cy1),3,(255,255,255),-1)
            center.append([cx1,cy1])
    print(center)
    #print("number of center detected",len(center))
    vertical_points = find_collinear(center)
    #print("number of collinear center detected",len(vertical_points))


    cv2.imshow("img", img)
    cv2.waitKey(wait)

cv2.destroyAllWindows()