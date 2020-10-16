import cv2
import numpy as np
from find_collinear import find_collinear
from remove_duplicates import remove_duplicates

threshGray = 0.6  # threshold to match template in grayscale, 1 for perfect match, 0 for no match
threshSat = 0.8  # threshold to match template in HSV saturation channel, 1 for perfect match, 0 for no match
wait = 200  # time between every frame in ms, 0 for manual scrolling


def draw_rectangle(img, points, w, h, color, thickness):
    for count, point in enumerate(points):
        cv2.rectangle(img, (int(round(point[0])), int(round(point[1]))), (int(round(point[0]))+w, int(round(point[1]))+h), color, thickness)


def match_template(im,temp):
    """Finds areas of an image that match (are similar) to a template image.

    Parameters
    ----------
    im: ndarray
       (Source image) The image in which we expect to find a match to the template image.
    temp: ndarray
        (Template image) The image which will be compared to the source image.
    Returns
    -------
        list
            The coordinates of the matching points found (Left uppermost corner of the ROI).
    """

    print("[info] matchtemplate running ...")

    template_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # turn template into grayscale
    template_hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)    # turn template into HSV
    template_sat = template_hsv[:, :, 1]                        # extracting only the saturation channel of the HSV template

    if type(im) is list:
        for img in im:
            matches = []
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn image into grayscale
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # turn image into HSV
            sat_img = hsv_img[:, :, 1]                        # extracting only the saturation channel of the HSV image
            h, w = template_gray.shape                        # get width and height of image

            resultGray = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
            locGray = np.where(resultGray >= threshGray)  # filter out bad matches
            for pt in zip(*locGray[::-1]):                # save all matches to a list
                matches.append([pt[0], pt[1]])

            resultSat = cv2.matchTemplate(sat_img, template_sat, cv2.TM_CCOEFF_NORMED)
            locSat = np.where(resultSat >= threshSat)     # filter out bad matches
            for pt in zip(*locSat[::-1]):                 # add all matches to the list
                matches.append([pt[0], pt[1]])

            sorted(matches, key=lambda x: x[0])           # sort the matched points by x coordinate

            #print("Matches: " + str(len(matches)))
            filteredMatches = remove_duplicates(matches)
            #print("After duplicate check: " + str(len(filteredMatches)))
            verticalMatches = find_collinear(filteredMatches)

            # print("After collinearity check: " + str(len(verticalMatches)))
            verticalMatches = verticalMatches[0:2]  # select the number of matches you want

            # draw_rectangle(img, matches, w, h, (150, 250, 255), 1)
            # draw_rectangle(img, filteredMatches, w, h, (0, 255, 0), 1)
            draw_rectangle(img, verticalMatches, w, h, (255, 0, 0), 2)
            cv2.imshow("img", img)
            cv2.waitKey(wait)
        print("[info] matchtemplate ended ...")
        return verticalMatches

    else:
        matches = []
        gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # turn image into grayscale
        hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # turn image into HSV
        sat_img = hsv_img[:, :, 1]  # extracting only the saturation channel of the HSV image
        h, w = template_gray.shape

        resultGray = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
        locGray = np.where(resultGray >= threshGray)  # filter out bad matches
        for pt in zip(*locGray[::-1]):  # save all matches to a list
            matches.append([pt[0], pt[1]])

        resultSat = cv2.matchTemplate(sat_img, template_sat, cv2.TM_CCOEFF_NORMED)
        locSat = np.where(resultSat >= threshSat)  # filter out bad matches
        for pt in zip(*locSat[::-1]):  # add all matches to the list
            matches.append([pt[0], pt[1]])

        sorted(matches, key=lambda x: x[0])  # sort the matched points by x coordinate

        # print("Matches: " + str(len(matches)))
        filteredMatches = remove_duplicates(matches)
        # print("After duplicate check: " + str(len(filteredMatches)))
        verticalMatches = find_collinear(filteredMatches)

        #print("After collinearity check: " + str(len(verticalMatches)))
        verticalMatches = verticalMatches[0:3]                            #select the number of matches you want
        print('matches',verticalMatches)
        if len(verticalMatches) > 0:
            #draw_rectangle(im, matches, w, h, (150, 250, 255), 1)
            #draw_rectangle(im, filteredMatches, w, h, (0, 255, 0), 1)
            draw_rectangle(im, verticalMatches, w, h, (0, 0, 255), 2)
            cv2.imshow("img", im)
            cv2.waitKey(wait)
            print("[info] matchtemplate ended ...")
            return verticalMatches
        else:
            print('Canot find any match') #!!! to change and quasi skip to next img
