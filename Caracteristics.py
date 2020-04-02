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
    def roundness(img, contour):
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    def assymetry(img, contour):
        # get fitted ellipse
        (cx, cy), (_, _), angle = cv2.fitEllipse(contour)
        x, y, w, h = cv2.boundingRect(contour)
        blankImg = np.zeros((h, w))
        contour = Caracteristics.translateContour(contour, x, y)
        cv2.drawContours(blankImg, [contour], -1, (255, 255, 255), -1)
        # cv2.line(blankImg, (int(w/2), int(h/2)), (int(w/2)+h, int(h/2)+h), (0, 255, 0), 1)
        blankImg = imutils.rotate(blankImg, angle+90)
        v = np.sum(blankImg, axis=0)
        h = np.sum(blankImg, axis=1)
        # print(v)
        # print(h)
        cv2.imshow('t', blankImg)

    '''
        translates the contour Vector by dx,dy
    '''
    @staticmethod
    def translateContour(contour, dx, dy):
        for p in contour:
            p[0][0] -= dx
            p[0][1] -= dy
        return contour
