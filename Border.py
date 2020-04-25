import cv2
import numpy as np
import pandas as pd
import imutils
from scipy import ndimage
from Preprocess import Preprocess
from Data import Data
from Caracteristics import Caracteristics
from Contours import Contours
import math
from matplotlib import pyplot as plt

class Border:
    '''
        all Border methods
    '''
    @staticmethod
    def borderRoundness(img,contour):
        '''
            get roundness of the Border
        '''
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # lesion perimeter
        lesionPerimeter = cv2.arcLength(contour, True)
        # get roundness
        roundness = (4 * np.pi * lesionArea) / (lesionPerimeter ** 2) * 100
        roundness = round(roundness, 2)
        return roundness
    
    @staticmethod
    def borderLength(img,contour):
        '''
            get the length of the Border
        '''
        # lesion perimeter
        lesionPerimeter = cv2.arcLength(contour, True)
        length = len(contour)
        return length

    @staticmethod
    def borderRegularityIndex(contour):
        '''
            get regularity index
        '''
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        regularityIndex = contourPerimeter / contourArea
        regularityIndex = round(regularityIndex, 2)
        return regularityIndex

    @staticmethod
    def borderRegularityIndexRatio(i, contour):
        '''
            get regularity index ratio
        '''
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        # get circle with same piremeter as contour
        radius = int(contourPerimeter / (2 * np.pi))
        # circle area
        circleArea = np.pi * (radius ** 2)
        regularityIndex = contourArea / circleArea
        regularityIndex = round(regularityIndex, 2)
        return regularityIndex
    
    @staticmethod
    def borderCompactIndex(contour):
        '''
            get Compact Index of a lesion
        '''
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        # get compactness
        compactIndex = (contourPerimeter**2) / (4*np.pi*contourArea)
        compactIndex = round(compactIndex, 2)
        return compactIndex
    
    @staticmethod
    def borderHeywoodCircularityIndex(img,contour):
        '''
            get Heywood's Circularity Index of the Border
            https://www.spacewx.com/Docs/ISO_DIS_10788_(E)_review.pdf
        '''
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # lesion perimeter
        lesionPerimeter = cv2.arcLength(contour, True)
        # get HCI result
        hci = lesionPerimeter / (2 * np.sqrt(np.pi * lesionArea))
        hci = round(hci, 2)
        return hci
    
    @staticmethod
    def borderHarrisCorner(img,contour):
        '''
            get corners with Harris corner detection
        '''
        # img dimensions
        h, w = img.shape[:2]
        # blank img
        blank = np.zeros((h, w), np.uint8)
        # draw the contour on the blank img
        blank = cv2.drawContours(blank, [contour], -1, (255, 255, 255), -1)
        # get harris corners
        blank = cv2.cornerHarris(blank, 2, 3, 0.04)
        #result is dilated for marking the corners, not important
        # blank = cv2.dilate(blank, None)
        # Threshold for an optimal value, it may vary depending on the image.
        ret, blank = cv2.threshold(blank, 0.01*blank.max(), 255, 0)
        blank = np.uint8(blank)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(blank)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(blank, np.float32(centroids), (5, 5), (-1, -1), criteria)
        return len(corners)
    
    @staticmethod
    def borderFractalDimension(img,contour):
        '''
            get Heywood's Circularity Index of the Border
        '''
        # convert img to gray
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # dilatation core
        s = 4
        m = s ** 2
        k = int(m / s)
        kernel = np.ones((k, k), np.uint8)
        # img dimensions
        h, w = imgray.shape[:2]
        # blank img
        blank = np.zeros((h, w), np.uint8)
        # draw the contour on the blank img
        blank = cv2.drawContours(blank, [contour], -1, (255, 255, 255), 1)
        # dilate
        # blank = cv2.dilate(blank, kernel, iterations=1)
        # count blocks of s*s
        nTotal = 0
        n = 0
        i = 0
        while i < h:
            j = 0
            while j < w:
                cell = blank[i:i + s, j:j + s]
                nTotal += 1
                if np.sum(cell != 0) != 0:
                    n = n + 1
                j = j + s
            i = i + s
        # lesion perimeter
        lesionPerimeter = cv2.arcLength(contour, True) / s
        # fd = round(n / (nTotal / lesionPerimeter), 2)
        fd = np.log(n) / np.log(k)
        fd = round(fd / lesionPerimeter * 100, 2)
        return fd

if __name__ == '__main__' :
    '''
        test program
    '''
    TYPE = 'Melanoma'
    # TYPE = 'Nevus'
    BDD = 'ISIC'
    # BDD = 'PH2'
    BDD_LOCATION = 'D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'
    DATA = BDD_LOCATION+BDD+'/'+TYPE+'/'
    files = Data.loadFilesAsArray(DATA)
    t = []
    i = 0
    for file in files:
        sImg = DATA+file
        img = cv2.imread(sImg,cv2.IMREAD_COLOR)
        # get contours
        contour = Contours.contours2(img)
        # draw contours
        cv2.drawContours(img, contour, -1, (0, 255, 255), 1)
        # get boundings
        Contours.boundingRectangle(img,contour)
        # get border
        borderCompactIndex = Border.borderCompactIndex(contour)
        print('['+str(i)+'/'+str(len(files))+']',borderCompactIndex)
        t.append(borderCompactIndex)
        i += 1
        # draw text
        x, y = 0, np.shape(img)[0]-3
        cv2.putText(img,'border : '+str(borderCompactIndex), (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 0), lineType = 1)
        cv2.imshow('img',img)
        if cv2.waitKey() == ord('c'):
            break

    # t = pd.DataFrame(t)
    # print('---------------------------------------------------------------------------------')
    # print('TYPE = ',TYPE)
    # print('BDD = ',BDD)
    # print('max = ',t.max(), ', min = ', t.min(), ', mean = ', t.mean(), ', median = ', t.median())
    # print(t)
    # # t = t.values
    # print('---------------------------------------------------------------------------------')
    cv2.destroyAllWindows()