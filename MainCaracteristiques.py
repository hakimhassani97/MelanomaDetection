import cv2
import numpy as np
import pandas as pd
import csv
import math
from matplotlib import pyplot as plt
from Data import Data
from Contours import Contours
from Asymmetry import Asymmetry
from Border import Border
from Color import Color
from Diameter import Diameter

class MainCaracteristiques:
    '''
        this class calculates the caracteristiques for each image
    '''
    @staticmethod
    def getData(out):
        f = open('outputs/'+out, 'r', newline='')
        data = []
        with f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        try:
            i = int(data[len(data)-1][0]) + 1
        except:
            i = 0
        return data, i

    @staticmethod
    def fillDbTypeNames(out, DATA, BDD):
        '''
            fills names, DB and types of the images into the 'out'
        '''
        DATA = DATA+BDD+'/'
        # read old file
        old, i = MainCaracteristiques.getData(out)
        f = open('outputs/'+out, 'w', newline='')
        with f:
            writer = csv.writer(f)
            # write old data
            writer.writerows(old)
            types = ['Melanoma', 'Nevus']
            for t in types:
                files = Data.loadFilesAsArray(DATA+t)
                for file in files:
                    sImg = DATA+t+'/'+file
                    writer.writerow([i, BDD, t, file])
                    i+=1
    
    @staticmethod
    def addColumn(out, DATA_SRC):
        '''
            adds a column to each row
        '''
        # read old file
        old, i = MainCaracteristiques.getData(out)
        colIndex = len(old[0])
        total = 1200
        curr = 1
        f = open('outputs/'+out, 'w', newline='')
        with f:
            writer = csv.writer(f)
            BDDS = ['PH2', 'ISIC']
            for BDD in BDDS:
                DATA = DATA_SRC+BDD+'/'
                types = ['Melanoma', 'Nevus']
                for t in types:
                    files = Data.loadFilesAsArray(DATA+t)
                    for file in files:
                        sImg = DATA+t+'/'+file
                        r = MainCaracteristiques.find(old, 3, file)
                        if r!=None:
                            # del r[-1]
                            img = cv2.imread(sImg, cv2.IMREAD_COLOR)
                            contour = Contours.contours2(img)
                            car = []
                            # append asymmetrys
                            # car.append(Asymmetry.asymmetryByBestFitEllipse(img, contour))
                            # car.append(Asymmetry.asymmetryByDistanceByCircle(img, contour))
                            # car.append(Asymmetry.asymmetryIndex(img, contour))
                            # car.append(Asymmetry.asymmetryBySubRegion(img, contour))
                            # car.append(Asymmetry.asymmetryBySubRegionCentered(img, contour))
                            # car.append(Asymmetry.asymmetryBySubRegionCentered2(img, contour))
                            # append borders
                            # car.append(Border.borderRoundness(img, contour))
                            # car.append(Border.borderLength(img, contour))
                            # car.append(Border.borderRegularityIndex(contour))
                            # car.append(Border.borderRegularityIndexRatio(img, contour))
                            # car.append(Border.borderCompactIndex(contour))
                            # car.append(Border.borderHeywoodCircularityIndex(img, contour))
                            # car.append(Border.borderHarrisCorner(img, contour))
                            # car.append(Border.borderFractalDimension(img, contour))
                            # append colros
                            # car.append(Color.colorHSVIntervals(img, contour))
                            # car.append(Color.colorYUVIntervals(img, contour))
                            # car.append(Color.colorYCbCrIntervals(img, contour))
                            # car.append(Color.colorSDG(img, contour))
                            # car.append(Color.colorKurtosis(img, contour))
                            # append diameters
                            # car.append(Diameter.diameterMinEnclosingCircle(img, contour))
                            # car.append(Diameter.diameterOpenCircle(img, contour))
                            # car.append(Diameter.diameterLengtheningIndex(img, contour))
                            r = np.append(r, car)
                            writer.writerow(r)
                            print('['+str(curr)+'/'+str(total)+']')
                            curr += 1
    
    @staticmethod
    def find(data, col, x):
        '''
            finds x in the data[col]
        '''
        for d in data:
            if len(d)>col and d[col]==x:
                return d
        return None

'''
    fill all caracteristiques
'''
BDD_LOCATION = 'D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'

# MainCaracteristiques.fillDbTypeNames('resnew.csv', BDD_LOCATION, 'PH2')
# MainCaracteristiques.fillDbTypeNames('resnew.csv', BDD_LOCATION, 'ISIC')
# MainCaracteristiques.addColumn('resnew.csv', BDD_LOCATION)