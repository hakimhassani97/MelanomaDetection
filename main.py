import cv2
import numpy as np
from Contours import Contours
from Data import Data
from Preprocess import Preprocess
from Caracteristics import Caracteristics

'''
    main program
'''
file_object = open('fichier.txt', 'a')

i = 0.0
moy = 0.0
max = -200
min = 200
#TYPE = 'Melanoma'
TYPE = 'Nevus'
#BDD = 'ISIC'
BDD = 'PH2'


file_object.write(
    'BDD = ' + BDD + ' / ' + 'TYPE = ' + TYPE + '\n')
file_object.write('\n')

file_object.write(
    'Assymetry methode of Distance Between the center of gravity of contour and  center of circle around the contour\n')
file_object.write('\n')


BDD_LOCATION = 'D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'
DATA = BDD_LOCATION+BDD+'/'+TYPE+'/'
files = Data.loadFilesAsArray(DATA)
for file in files:
    sImg = DATA+file
    img = cv2.imread(sImg, cv2.IMREAD_COLOR)
    # get contours
    contour = Contours.contours2(img)
    # draw contours
    # img=np.zeros(np.shape(img))
    cv2.drawContours(img, contour, -1, (0, 255, 255), 1)
    # get boundings
    # Contours.boundingRectangle(img, contour)
    # Contours.boundingCircle(img,contour)
    # get Compact Index
    # print(Caracteristics.regularityIndexPercentage(contour))
    # get number of colors
    # Caracteristics.colorThreshold(img, contour)
    # get roundness
    # roundness = Caracteristics.roundness(img, contour)
    # roundness = round(roundness, 2)
    # x, y = 0, np.shape(img)[0]-3
    # cv2.putText(img, 'roundness : '+str(roundness), (x, y),
    # cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), lineType=1)

    # cv2.putText(img, 'Asymétrique : '+str(resultOfAsymétrique), (x, y),
    #             cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), lineType=1)

    # get assymetry
    # Caracteristics.assymetry(img,contour)

    #
    #
    #
    #
    #
    #
    #
    #

    ################################################ Start All About Statistic ########################################################################
    '''
        1 -Call Assymetry A Functions  :
    '''

    # 1.1 Distance Between the center of gravity of contour and  center of circle around the contour
    AssymetryByDistanceByCircle = Caracteristics.AssymetryByDistanceByCircle(
        img, contour)

    i = i+1
    moy = moy + AssymetryByDistanceByCircle
    if AssymetryByDistanceByCircle > max:
        max = AssymetryByDistanceByCircle
    if AssymetryByDistanceByCircle < min:
        min = AssymetryByDistanceByCircle
    file_object.write("image : "+str(i) + " , value : " +
                      str(AssymetryByDistanceByCircle) + '\n')

    '''
         2 -Call Border B Functions :
    '''

    # 2.1 Between perimeter and contour area
    # regularityIndex = Caracteristics.regularityIndex(contour)

    # 2.2 Between circle with same piremeter as contour and contour area
    # regularityIndexPercentage = Caracteristics.regularityIndexPercentage(
    #     contour)

    '''
         3 -Call Color C Functions  :
    '''

    # 3.1 gets number of colors from kmeans centers
    # Caracteristics.nbColors(img, contour)

    # 3.2 gets number of colors from color histogram
    # nbColorsHist = Caracteristics.nbColorsHist(img, contour)

    # 3.3 Kurtosis, color distribution
    # kurtosis = Caracteristics.kurtosis(img, contour)

    # 3.4 color thresholds
    # colorThreshold = Caracteristics.colorThreshold(img, contour)

    # 3.5 extracts the lesion
    # extractLesion = Caracteristics.extractLesion(img, contour)

    '''
        4 -Call Diameter D Functions
    '''

    # DiameterByCircle = Caracteristics.DiameterByCircle(img, contour)
    ################################################ End All About statistic #####################################################################

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    cv2.imshow('img', img)
    if cv2.waitKey() == ord('c'):
        break
moy = moy/i
center = (max+min)/2.0
file_object.write("\n")
file_object.write(
    "----------------------------------------------------------------------------------------------------------------------------------------\n")
file_object.write("Moyenn : "+str(moy) + " , Max value : "+str(max) +
                  " , Min value : "+str(min)+" ,Center between Max Min : "+str(center) + "\n")
file_object.write(
    "----------------------------------------------------------------------------------------------------------------------------------------\n")
file_object.write("\n")
file_object.close()
cv2.destroyAllWindows()
