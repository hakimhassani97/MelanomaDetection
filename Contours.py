import cv2
import numpy as np
from Data import Data
from Preprocess import Preprocess

'''
    method 1
    get contours
'''
def contours1(img):
    # convert img to grayscale
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply threshold
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # get contours
    c,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

'''
    method 2
    get contours
'''
def contours2(img):
    ret, thresh = Preprocess.removeHair(img)
    # search for contours and select the biggest one
    c, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    return cnt

'''
    draw bounding circle
'''
def boundingCircle(img,contour):
    # get perimeter of contour
    perimeter = cv2.arcLength(contour, True)
    # get moment of contour
    M = cv2.moments(contour)
    # get center of gravity of contour
    xe = int(M["m10"] / M["m00"])
    ye = int(M["m01"] / M["m00"])
    # get center of circle around the contour
    radius = int(perimeter / (2 * np.pi))
    # draw the circle and its center
    cv2.circle(img, (xe, ye), radius=1, color=(0, 255, 255), thickness=1)
    cv2.circle(img, (xe, ye), radius=radius, color=(0, 255, 255), thickness=1)

'''
    draw bounding rectangle
'''
def boundingRectangle(img,contour):
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(img,(x,y),(x+w,y+h), color=(0, 255, 255), thickness=2)


'''
    get roundiness
'''
def roundness(contour):
    # get surface of contour
    area = cv2.contourArea(contour)
    # get perimeter of contour
    perimeter = cv2.arcLength(contour, True)
    # get roundness
    roundness = (4 * np.pi * area) / (perimeter * perimeter) * 100
    roundness = round(roundness, 2)

'''
    main program
'''
files=Data.loadFilesAsArray('data/')
for file in files:
    sImg='data/'+file
    img=cv2.imread(sImg,cv2.IMREAD_COLOR)
    # get contours
    contour=contours2(img)
    # draw contours
    # img=np.zeros(np.shape(img))
    cv2.drawContours(img, contour, -1, (0, 255, 255), 1)
    boundingRectangle(img,contour)
    cv2.imshow('img',img)
    if cv2.waitKey() == ord('c'):
        break
cv2.destroyAllWindows()