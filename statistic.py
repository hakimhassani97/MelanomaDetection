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
TYPE = 'Nevus'
# BDD='ISIC'
BDD = 'PH2'

BDDS = ['ISIC', 'PH2']
TYPES = ['Melanoma', 'Nevus']
#  ------------------------------------choisir le nom de fichier----------------------
fichierName = "Asymetry"
#  -----------------------------------------------------------------------------------
fichier = open('outputs/'+fichierName+".txt", "w")
fichier. write(
    "--------------------------------All About Statistic -----------------------------------\n")
moyenn = 0.0
i = 0.0
max = -10000000
min = 10000000

for BD in BDDS:
    for TY in TYPES:
        BDD = BD
        TYPE = TY
        BDD_LOCATION = 'D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'
        DATA = BDD_LOCATION+BDD+'/'+TYPE+'/'
        files = Data.loadFilesAsArray(DATA)
        moyenn = 0.0
        i = 0.0
        fichier. write("\n")
        fichier. write("BDD : "+BDD + ", TYPE : "+TYPE + "\n")
        fichier. write("\n")
        for file in files:
            sImg = DATA+file
            img = cv2.imread(sImg, cv2.IMREAD_COLOR)
            # get contours
            contour = Contours.contours2(img)

            # --------------------- choisir la fonction-------------------------------
            result = Caracteristics.DiameterByCircle(img, contour)
            # -----------------------------------------------------------------------

            moyenn = moyenn + result
            i = i+1
            if result > max:
                max = result
            if result < min:
                min = result
            fichier. write("img : "+str(i) + " result : " + str(result) + "\n")

            # cv2.imshow('img', img)
            # if cv2.waitKey() == ord('c'):
            #     break
        moyenn = moyenn/i
        center = (max+min)/2.0
        fichier. write(
            "-------------------------------------------------------------------------------------------------------------------------\n")
        fichier. write("moyenn = "+str(moyenn) + " ; max value = "+str(max) +
                       " ; min value = "+str(min) + " ; center between min and max = "+str(center) + "\n")

        fichier. write(
            "-------------------------------------------------------------------------------------------------------------------------\n")


fichier. close()
cv2.destroyAllWindows()
