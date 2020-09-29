import cv2
import numpy as np
import os


########################################################################################################
path = "resources/1001"                             # change path to folder with images
template = cv2.imread("resources/roi.jpg")          # change to path of RoI
thresh = 0.6                                        # threshold to match template, 1 for perfect match, 0 for no match
wait = 500                                          # time between every frame in ms
########################################################################################################


def load_images_from_folder(folder):    # function loading all files from a directory and saving them to an array
    print("Images loading...")
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    print("Images loaded")
    return images


images = load_images_from_folder(path)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)          # turn template into grayscale (optional)

for img in images:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # turn image into grayscale (optional)
    h, w = template_gray.shape                                      # get width and height of image

    result = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= thresh)                                # filter out bad matches

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 1)

    cv2.imshow("img", img)
    cv2.waitKey(wait)
cv2.destroyAllWindows()
