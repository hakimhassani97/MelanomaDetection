import cv2
import numpy as np

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
    # perform closing to remove hair and blur the image
    kernel = np.ones((15,15),np.uint8)
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 2)
    blur = cv2.blur(closing,(15,15))
    # apply OTSU threshold
    imgray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # search for contours and select the biggest one
    c, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    return cnt

'''
    main program
'''
sImg='data/5.jpg'
img=cv2.imread(sImg,cv2.IMREAD_COLOR)
# get contours
contours=contours2(img)
# draw contours
# img=np.zeros(np.shape(img))
cv2.drawContours(img, contours, -1, (0,255,0), 1)
cv2.imshow('img',img)

cv2.waitKey()
cv2.destroyAllWindows()