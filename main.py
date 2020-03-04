import cv2
import numpy as np
from Contours import Contours
from Data import Data
from Preprocess import Preprocess

'''
    main program
'''
files=Data.loadFilesAsArray('data/')
for file in files:
    sImg='data/'+file
    img=cv2.imread(sImg,cv2.IMREAD_COLOR)
    # get contours
    contour=Contours.contours2(img)
    # draw contours
    # img=np.zeros(np.shape(img))
    cv2.drawContours(img, contour, -1, (0, 255, 255), 1)
    Contours.boundingRectangle(img,contour)
    cv2.imshow('img',img)
    if cv2.waitKey() == ord('c'):
        break
cv2.destroyAllWindows()