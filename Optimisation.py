from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import cv2
from Data import Data
from Contours import Contours
from Caracteristics import Caracteristics
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from Calulator import Calculator

class Optimisation:
    '''
        dimensionnality reduction
    '''
    def train():
        '''
            do training on data
        '''
        XData = pd.read_csv('outputs/res binary.csv', header=None)
        target = XData.loc[:,2]
        X = XData.loc[:,4:]
        scaler = StandardScaler()
        # XStandard = scaler.fit_transform(X)
        # pca = PCA(n_components=5)
        # xpca = pca.fit_transform(XStandard)
        # xpcaDf = pd.DataFrame(data = xpca)
        # test_size: what proportion of original data is used for test set
        train_data, test_data, train_lbl, test_lbl = train_test_split( X, target, test_size=1/7.0, random_state=0)
        # Fit on training set only.
        scaler.fit(train_data)
        # Apply transform to both the training set and the test set.
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        # Make an instance of the Model
        pca = PCA(0.8)
        pca.fit(train_data)
        # print(pca.components_)
        # print(pd.DataFrame(pca.components_, columns=[4,5,6,7,8,9,10,11,12], index = ['pc1','pc2','pc3','pc4']))
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)
        # default solver is incredibly slow which is why it was changed to 'lbfgs'
        logisticRegr = LogisticRegression(solver = 'lbfgs')
        logisticRegr.fit(train_data, train_lbl)
        # Predict for One Observation (image)
        prediction = logisticRegr.predict(test_data[40].reshape(1,-1))
        # score of training
        score = logisticRegr.score(test_data, test_lbl)
        print(score)
        # save the classifier
        version = 2
        _ = joblib.dump(logisticRegr, 'outputs/models/model'+str(version)+'.pkl', compress=9)
        _ = joblib.dump(scaler, 'outputs/models/scaler'+str(version)+'.pkl', compress=9)
        _ = joblib.dump(pca, 'outputs/models/pca'+str(version)+'.pkl', compress=9)
    
    def trainOneCol(col):
        '''
            do training on one column of data
        '''
        XData = pd.read_csv('outputs/res binary.csv', header=None)
        target = XData.loc[:,2]
        X = XData.loc[:,col:col]
        scaler = StandardScaler()
        XStandard = scaler.fit_transform(X)
        # test_size: what proportion of original data is used for test set
        train_data, test_data, train_lbl, test_lbl = train_test_split( X, target, test_size=1/7.0, random_state=0)
        # Fit on training set only.
        scaler.fit(train_data)
        # Apply transform to both the training set and the test set.
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        # default solver is incredibly slow which is why it was changed to 'lbfgs'
        logisticRegr = LogisticRegression(solver = 'lbfgs')
        logisticRegr.fit(train_data, train_lbl)
        # Predict for One Observation (image)
        prediction = logisticRegr.predict(test_data[40].reshape(1,-1))
        # score of training
        score = logisticRegr.score(test_data, test_lbl)
        print('col = ', col, ', score = ', score)
        # save the classifier
        version = 2
        _ = joblib.dump(logisticRegr, 'outputs/models/model col'+str(col)+' '+str(version)+'.pkl', compress=9)
        _ = joblib.dump(scaler, 'outputs/models/scaler col'+str(col)+' '+str(version)+'.pkl', compress=9)
    
    def trainSvm():
        '''
            use SVM for classification
        '''
        XData = pd.read_csv('outputs/res binary.csv', header=None)
        target = XData.loc[:,2]
        X = XData.loc[:,4:]
        X_train, X_test, target_train, target_test = train_test_split(X, target, test_size = 0.2, random_state=0)
        # init scaler
        scaler = StandardScaler()
        # Fit on training set only.
        scaler.fit(X_train)
        # Apply transform to both the training set and the test set.
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Make an instance of the Model
        pca = PCA(0.96)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        clf = svm.SVC(gamma='auto')
        r = clf.fit(X_train, target_train)
        print(r.score(X_test, target_test))
        # test = X.loc[40,:]
        # test = np.reshape(np.array(test), (1, -1))
        # print(test)
        # print(r.predict(test))
    
    def trainNN():
        '''
            use Neural network for classification
        '''
        XData = pd.read_csv('outputs/res binary.csv', header=None)
        target = XData.loc[:,2]
        X = XData.loc[:,4:]
        X_train, X_test, target_train, target_test = train_test_split(X, target, test_size = 0.2, random_state=0)
        # init scaler
        scaler = StandardScaler()
        # Fit on training set only.
        scaler.fit(X_train)
        # Apply transform to both the training set and the test set.
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Make an instance of the Model
        pca = PCA(0.95)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 30, 5), random_state=1)
        NN.fit(X_train, target_train)
        r = round(NN.score(X_test,target_test), 4)
        print(r)

    @staticmethod
    def test():
        '''
            test saved models
        '''
        version = 2
        logisticRegr = joblib.load('outputs/models/model'+str(version)+'.pkl')
        # scaler
        scaler = joblib.load('outputs/models/scaler'+str(version)+'.pkl')
        # pca
        pca = joblib.load('outputs/models/pca'+str(version)+'.pkl')
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
            X = scaler.transform([X])
            X = pca.transform(X)
            cars = X.reshape(1,-1)
            prediction = logisticRegr.predict(cars)
            print('truth =', TYPE, 'prediction =', types[int(prediction)])
            if cv2.waitKey() == ord('c'):
                break
        cv2.destroyAllWindows()

    @staticmethod
    def applyPca():
        '''
            calculates the pca matrix for a given dataset
        '''
        # X, i = Calculator.getData('res.csv')
        X = pd.read_csv('outputs/res.csv',header=None)
        X = X.loc[:,4:]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components=5)
        xpca = pca.fit_transform(X)
        xpcadf = pd.DataFrame(data = xpca)
        print(xpcadf)
        print(pca.explained_variance_ratio_)
        # y = ['DiameterByCircle','compactIndex','regularityIndex','regularityIndexPercentage','colorThreshold','nbColors','kurtosis','AssymetryByDistanceByCircle','roundness']
        # y = np.array(y)
        # # Standardizing the features
        # xpca = pca.fit_transform(X)
        # print(xpca[0])

# Optimisation.trainOneCol(11)
Optimisation.train()
# Optimisation.trainSvm()
# Optimisation.trainNN()
# Optimisation.test()