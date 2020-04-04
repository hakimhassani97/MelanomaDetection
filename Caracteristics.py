import cv2
import numpy as np
import imutils

'''
    get ABCD, 7 points, menzies caracteristics
'''
class Caracteristics:
    '''
        needed for Assymetry A
        get roundness of the contour
    '''
    @staticmethod
    def roundness(img,contour):
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ellipse = cv2.fitEllipse(contour)
        # mask = cv2.ellipse(img, ellipse, (255, 255, 255), 1)
        blankImg = np.zeros(np.shape(imgray))
        # draw ellipse on empty image
        cv2.ellipse(blankImg, ellipse, (255, 255, 255), -1)
        # draw solid contour on empty image
        cv2.drawContours(blankImg, [contour], -1, (0, 0, 0), -1)
        # surface of intersection area
        contourArea = cv2.contourArea(contour)
        intersectionArea = contourArea - np.count_nonzero(blankImg)
        # get the ratio between intersectionArea and the contourArea
        roundness = intersectionArea/contourArea
        return roundness
    
    '''
        get Assymetry A of a lesion
    '''
    @staticmethod
    def assymetry(img,contour):
        # get fitted ellipse
        (cx, cy), (_, _), angle = cv2.fitEllipse(contour)
        x, y, w, h = cv2.boundingRect(contour)
        blankImg = np.zeros((h, w))
        contour = Caracteristics.translateContour(contour,x,y)
        cv2.drawContours(blankImg, [contour], -1, (255, 255, 255), -1)
        # cv2.line(blankImg, (int(w/2), int(h/2)), (int(w/2)+h, int(h/2)+h), (0, 255, 0), 1)
        blankImg = imutils.rotate_bound(blankImg, angle+90)
        v = np.sum(blankImg, axis=0)
        h = np.sum(blankImg, axis=1)
        # print(v)
        # print(h)
        cv2.imshow('t',blankImg)
    
    '''
        translates the contour Vector by dx,dy
    '''
    @staticmethod
    def translateContour(contour,dx,dy):
        for p in contour:
            p[0][0]-=dx
            p[0][1]-=dy
        return contour
    
    '''
        needed for Border B
        get Compact Index of a lesion
    '''
    @staticmethod
    def compactIndex(contour):
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        return (contourPerimeter**2) / (4*np.pi*contourArea)
    
    '''
        needed for Border B
        get regularity index
    '''
    @staticmethod
    def regularityIndex(contour):
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        return contourPerimeter / contourArea
    
    '''
        needed for Border B
        get regularity index 2
    '''
    @staticmethod
    def regularityIndexPercentage(contour):
        # get contour perimeter
        contourPerimeter = cv2.arcLength(contour, True)
        # get contour area
        contourArea = cv2.contourArea(contour)
        # get circle with same piremeter as contour
        radius = int(contourPerimeter / (2 * np.pi))
        # circle area
        circleArea = np.pi * (radius ** 2)
        return contourArea / circleArea
    
    '''
        needed for Color C
        get number of colors
    '''
    @staticmethod
    def nbColors(img, contour):
        lesion = Caracteristics.extractLesion(img, contour)
        cv2.imshow('nb colors', lesion)
    
    '''
        needed for Color C
        extract lesion
    '''
    @staticmethod
    def extractLesion(img, contour):
        mask = np.zeros(img.shape, dtype='uint8')
        mask = cv2.drawContours(mask, [contour], -1, (255 , 255 , 255),thickness=cv2.FILLED)
        # mask = cv2.bitwise_not(mask)
        img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        lesion = cv2.bitwise_and(img, img, mask=mask)
        return lesion