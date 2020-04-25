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

class Diameter:
    '''
        all Asymmetry methods
    '''
    @staticmethod
    def diameterMinEnclosingCircle(img, contour):
        '''
            caluclate Diameter by Diameter of Circle around contour
        '''
        # get enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        d = radius * 2
        return d

    @staticmethod
    def diameterOpenCircle(img, contour):
        '''
            caluclate diameter of open circle of the lesion
        '''
        # lesion perimeter
        lesionPerimeter = cv2.arcLength(contour, True)
        # get diameter of circle with same perimeter as lesionPerimeter
        d = lesionPerimeter / np.pi
        d = round(d, 2)
        return d

    @staticmethod
    def diameterLengtheningIndex(img, contour):
        '''
            caluclate Lengthening Index of the lesion
        '''
        # get moments of contour
        M = cv2.moments(contour)
        # moments of inertia
        lamda1 = (M["m20"] + M["m02"] - np.sqrt(np.power(M["m20"] - M["m02"], 2) + 4 * (np.power(M["m11"], 2)))) / 2
        lamda2 = (M["m20"] + M["m02"] + np.sqrt(np.power(M["m20"] - M["m02"], 2) + 4 * (np.power(M["m11"], 2)))) / 2
        li = lamda1 / lamda2 * 100
        li = round(li, 2)
        return li

if __name__ == '__main__' :
    '''
        test program
    '''
    # TYPE = 'Melanoma'
    TYPE = 'Nevus'
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
        # Contours.boundingRectangle(img,contour)
        # get color
        diameterOpenCircle = Diameter.diameterOpenCircle(img,contour)
        print('['+str(i)+'/'+str(len(files))+']',diameterOpenCircle)
        t.append(diameterOpenCircle)
        i += 1
        # draw text
        x, y = 0, np.shape(img)[0]-3
        cv2.putText(img,'color : '+str(diameterOpenCircle), (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 0), lineType = 1)
        cv2.imshow('img',img)
        if cv2.waitKey() == ord('c'):
            break
    cv2.destroyAllWindows()