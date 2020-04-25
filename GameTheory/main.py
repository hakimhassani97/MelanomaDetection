import sys
sys.path.append('')
import time
import numpy as np
import nashpy as nash
import cv2
from Data import Data
from Contours import Contours
from Caracteristics import Caracteristics
from NashEnumSupport import NashEnumSupport
from NashEnumSommet import NashEnumSommet
from NashLemkHowson import NashLemkHowson

# for i in range(25,26,1):
#     A = np.int8(10 * np.random.random((10, 10)))
#     B = np.int8(10 * np.random.random((10, 10)))
#     t1 = time.time()
#     # J1 = NashEnumSupport(A,B).EQ()
#     #J1 = NashEnumSommet(A,B).EQ()
#     J1 = NashLemkHowson(A,B).EQs()
#     #J1 = NashLemkHowson(A,B).EQ()
#     t2 = time.time()
#     print (J1)
#     print(t2-t1)
# img data
BDD = 'PH2'
types = ['Nevus', 'Melanoma']
TYPE = types[0]
DATA = 'D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'
DATA = DATA+BDD+'/'+TYPE+'/'
files = Data.loadFilesAsArray(DATA)
for file in files:
    sImg = DATA+file
    img = cv2.imread(sImg, cv2.IMREAD_COLOR)
    contour = Contours.contours2(img)
    cv2.imshow(TYPE, img)
    car1 = round(Caracteristics.DiameterByCircle(img, contour), 4)
    car2 = round(Caracteristics.compactIndex(contour), 4)
    car3 = round(Caracteristics.regularityIndex(contour), 4)
    car4 = round(Caracteristics.regularityIndexPercentage(contour), 4)
    car5 = round(Caracteristics.colorThreshold(img, contour), 4)
    car6 = round(Caracteristics.nbColors(img, contour), 4)
    car7 = round(Caracteristics.kurtosis(img, contour), 4)
    car8 = round(Caracteristics.AssymetryByDistanceByCircle(img, contour), 4)
    car9 = round(Caracteristics.roundness(img, contour), 4)
    X = [car1, car2, car3, car4, car5, car6, car7, car8, car9]
    # game
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    B = -A
    g = nash.Game(A,B)
    if cv2.waitKey() == ord('c'):
        break
cv2.destroyAllWindows()
