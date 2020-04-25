import csv
import cv2
import numpy as np
from Contours import Contours
from Caracteristics import Caracteristics
from Data import Data

class Calculator:
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
        old, i = Calculator.getData(out)
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
        old, i = Calculator.getData(out)
        colIndex = len(old[0])
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
                        r = Calculator.find(old, 3, file)
                        if r!=None:
                            # del r[-1]
                            img = cv2.imread(sImg, cv2.IMREAD_COLOR)
                            contour = Contours.contours2(img)
                            # car = Caracteristics.DiameterByCircle(img, contour)
                            # car = Caracteristics.compactIndex(contour)
                            # car = Caracteristics.regularityIndex(contour)
                            # car = Caracteristics.regularityIndexPercentage(contour)
                            # car = Caracteristics.colorThreshold(img, contour)
                            # car = Caracteristics.nbColors(img, contour)
                            # car = Caracteristics.kurtosis(img, contour)
                            # car = Caracteristics.AssymetryByDistanceByCircle(img, contour)
                            # car = Caracteristics.roundness(img, contour)
                            car = round(car, 4)
                            r.append(car)
                            writer.writerow(r)
    
    @staticmethod
    def find(data, col, x):
        '''
            finds x in the data[col]
        '''
        for d in data:
            if len(d)>col and d[col]==x:
                return d
        return None

BDD_LOCATION = 'D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'

# Calculator.fillDbTypeNames('res.csv', BDD_LOCATION, 'PH2')
# Calculator.fillDbTypeNames('res.csv', BDD_LOCATION, 'ISIC')
# Calculator.addColumn('res.csv', BDD_LOCATION)