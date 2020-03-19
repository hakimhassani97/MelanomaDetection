import cv2
import numpy as np
from Contours import Contours
from Data import Data
from Preprocess import Preprocess
from Caracteristics import Caracteristics

'''
    main program
'''
# TYPE='Melanoma'
TYPE='Nevus'
# BDD='ISIC'
BDD='PH2'
BDD_LOCATION='D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'
DATA=BDD_LOCATION+BDD+'/'+TYPE+'/'
files=Data.loadFilesAsArray(DATA)
for file in files:
    sImg=DATA+file
    img=cv2.imread(sImg,cv2.IMREAD_COLOR)
    # get contours
    contour=Contours.contours2(img)
    # draw contours
    # img=np.zeros(np.shape(img))
    cv2.drawContours(img, contour, -1, (0, 255, 255), 1)
    # get boundings
    Contours.boundingRectangle(img,contour)
    # Contours.boundingCircle(img,contour)
    # get roundness
    roundness=Caracteristics.roundness(img,contour)
    roundness=round(roundness,2)
    x, y = 0, np.shape(img)[0]-3
    cv2.putText(img,'roundness : '+str(roundness), (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), lineType=1)
    # get assymetry
    # Caracteristics.assymetry(img,contour)
    cv2.imshow('img',img)
    if cv2.waitKey() == ord('c'):
        break
cv2.destroyAllWindows()