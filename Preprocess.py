import cv2
import numpy as np

class Preprocess:

    '''
        removes hair (artifacts) from an image
    '''
    @staticmethod
    def removeArtifact(img):
        # perform closing to remove hair
        kernel = np.ones((15,15),np.uint8)
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 2)
        # blur the image
        blur = cv2.blur(closing,(15,15))
        # apply OTSU threshold
        imgray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return ret, thresh
    
    '''
        equalizes the Y channel of an YUV image
    '''
    @staticmethod
    def equalizeHistYChannel(img):
        # convert image to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert image to RGB
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img