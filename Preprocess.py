import cv2
import numpy as np

class Preprocess:

    '''
        removes hair (artifacts) from an image
        morphologic close transformation
    '''
    @staticmethod
    def removeArtifact(img):
        # perform closing to remove hair
        kernel = np.ones((15,15),np.uint8)
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 2)
        return closing
    
    '''
        ASLM Noise Removal
        equalizes the Y channel of an YUV image, Y contains the intensity information
        https://www.opencv-srf.com/2018/02/histogram-equalization.html
        http://users.diag.uniroma1.it/bloisi/papers/bloisi-CMIG-2016-draft.pdf
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
    
    '''
        DR noise removal
        apply morphologic close transformation on each channel of RGB image
        kernel of (11,11) based on hair size
    '''
    @staticmethod
    def removeArtifactRGB(img):
        # median filtre
        imgMedian = cv2.medianBlur(img, 5, 5)
        # split RGB channels
        imgB, imgG, imgR = cv2.split(imgMedian)
        # kernel of 11 * 11
        kernel = np.ones((11, 11), np.uint8)
        # perform morphologic closing on each RGB channel
        imgClosingB = cv2.morphologyEx(imgB, cv2.MORPH_CLOSE, kernel)
        imgClosingG = cv2.morphologyEx(imgG, cv2.MORPH_CLOSE, kernel)
        imgClosingR = cv2.morphologyEx(imgR, cv2.MORPH_CLOSE, kernel)
        # merge the 3 channels
        imgResult = cv2.merge((imgClosingB, imgClosingG, imgClosingR))
        return imgResult

    '''
        DR noise removal
        1) convert BGR to YUV, 2) process, 3) convert YUV to RGB for OTSU
        apply morphologic close transformation on each channel of YUV image
        kernel of (11,11) based on hair size
    '''
    @staticmethod
    def removeArtifactYUV(img):
        # median filtre
        imgMedian = cv2.medianBlur(img, 5, 5)
        # split YUV channels
        img_yuv = cv2.cvtColor(imgMedian, cv2.COLOR_RGB2YUV)
        imgV, imgU, imgY = cv2.split(img_yuv)
        # kernel of 11 * 11
        kernel = np.ones((11, 11), np.uint8)
        # perform morphologic closing on each RGB channel
        imgClosingV = cv2.morphologyEx(imgV, cv2.MORPH_CLOSE, kernel)
        imgClosingU = cv2.morphologyEx(imgU, cv2.MORPH_CLOSE, kernel)
        imgClosingY = cv2.morphologyEx(imgY, cv2.MORPH_CLOSE, kernel)
        # merge the 3 channels
        imgResult = cv2.merge((imgClosingV, imgClosingU, imgClosingY))
        # back to RGB
        imgResult = cv2.cvtColor(imgResult, cv2.COLOR_YUV2RGB)
        return imgResult
    
    '''
        apply OTSU threshold
    '''
    @staticmethod
    def OTSUThreshold(img):
        # blur the image
        blur = cv2.blur(img,(15,15))
        # apply OTSU threshold
        imgray = cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return ret, thresh