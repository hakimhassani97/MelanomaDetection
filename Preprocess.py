import cv2
import numpy as np

class Preprocess:

    @staticmethod
    def removeHair(img):
        # perform closing to remove hair and blur the image
        kernel = np.ones((15,15),np.uint8)
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 2)
        blur = cv2.blur(closing,(15,15))
        # apply OTSU threshold
        imgray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return ret, thresh