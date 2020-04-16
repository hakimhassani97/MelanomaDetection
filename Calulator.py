import csv
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

BDD_LOCATION = 'D:/HAKIM/MIV M2/PFE/fichiers prof/MIV 96-2019/Application MIV 96-2019/Code/BDD/'

Calculator.fillDbTypeNames('res.csv', BDD_LOCATION, 'PH2')
Calculator.fillDbTypeNames('res.csv', BDD_LOCATION, 'ISIC')