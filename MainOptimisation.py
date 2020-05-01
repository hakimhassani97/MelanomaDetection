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

class MainOptimisation:
    '''
        dimensionnality reduction
    '''
    def train():
        '''
            do training on data
        '''
        XData = pd.read_csv('outputs/resnew.csv', header=None)
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
        version = ' new'
        _ = joblib.dump(logisticRegr, 'outputs/models/model'+str(version)+'.pkl', compress=9)
        _ = joblib.dump(scaler, 'outputs/models/scaler'+str(version)+'.pkl', compress=9)
        _ = joblib.dump(pca, 'outputs/models/pca'+str(version)+'.pkl', compress=9)
    
    def trainOneCol(col, BDD=None):
        '''
            do training on one column of data
        '''
        if BDD == None:
            BDD = ''
        else:
            BDD = ' '+BDD
        XData = pd.read_csv('outputs/resnew'+BDD+'.csv', header=None)
        target = XData.loc[:,2]
        X = XData.loc[:,col:col]
        # scaler = StandardScaler()
        # XStandard = scaler.fit_transform(X)
        # test_size: what proportion of original data is used for test set
        train_data, test_data, train_lbl, test_lbl = train_test_split( X, target, test_size=1/7.0, random_state=0)
        # Fit on training set only.
        # scaler.fit(train_data)
        # Apply transform to both the training set and the test set.
        # train_data = scaler.transform(train_data)
        # test_data = scaler.transform(test_data)
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        # default solver is incredibly slow which is why it was changed to 'lbfgs'
        logisticRegr = LogisticRegression(solver = 'lbfgs')
        logisticRegr.fit(train_data, train_lbl)
        # Predict for One Observation (image)
        # prediction = logisticRegr.predict(test_data[40].reshape(1,-1))
        # score of training
        score = logisticRegr.score(test_data, test_lbl)
        print('col = ', col, ', score = ', score)
        # count number of FP and FN
        predictions = logisticRegr.predict(test_data)
        T = 0
        test_lbl = np.array(test_lbl)
        results = pd.DataFrame()
        for i in range(0, len(predictions)):
            p = predictions[i]
            t = test_lbl[i]
            df = pd.DataFrame({'car':test_data[i] , 'target':t, 'prediction':p})
            results = results.append(df, ignore_index=True)
        # use pandas
        targets = results['target']
        total = len(targets)
        predics = results['prediction']
        truths = results.loc[targets == predics]
        T = len(truths) / total
        fns = results.loc[(targets != predics) & (predics == 0)]
        FN = len(fns) / total
        fps = results.loc[(targets != predics) & (predics == 1)]
        FP = len(fps) / total
        print('fn = ', FN, ', fp = ', FP, ', T = ', T)
        # save the classifier
        version = BDD
        _ = joblib.dump(logisticRegr, 'outputs/finals/logisticRegr col'+str(col)+' '+str(version)+'.pkl', compress=9)
        # _ = joblib.dump(scaler, 'outputs/finals/scaler col'+str(col)+' '+str(version)+'.pkl', compress=9)
        return results, fns, fps
    
    def trainSvm():
        '''
            use SVM for classification
        '''
        XData = pd.read_csv('outputs/resnew.csv', header=None)
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
        pca = PCA(.92)
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
        XData = pd.read_csv('outputs/resnew.csv', header=None)
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
        pca = PCA(0.94)
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
    def testFinals(col, seuil, BDD=None, op=None):
        '''
            test final results and thresholds
        '''
        if BDD == None:
            BDD = ''
        else:
            BDD = ' '+BDD
        XData = pd.read_csv('outputs/resnew'+BDD+'.csv', header=None)
        # bdd = XData.loc[:,1].values
        # print(bdd[bdd=='PH2'])
        target = XData.loc[:,2].values
        X = XData.loc[:,col:col].values
        # predict
        version = 1
        logisticRegr = joblib.load('outputs/finals/logisticRegr col'+str(col)+' '+str(version)+'.pkl')
        predictions = logisticRegr.predict(X)
        T = 0
        target = np.array(target)
        results = pd.DataFrame()
        for i in range(0, len(predictions)):
            p = predictions[i]
            t = target[i]
            df = pd.DataFrame({'car':X[i] , 'target':t, 'prediction':p})
            results = results.append(df, ignore_index=True)
            print(X[i][0], p)
        # verify thresholds
        # print(len(results[results['car']<seuil]['target']))
        # print('target = 1, <seuil,',len(results[(results['car']<seuil) & (results['target']==1)]['target'])/len(results[results['car']<seuil]['target']))
        # print('target = 0, <seuil,',len(results[(results['car']<seuil) & (results['target']==0)]['target'])/len(results[results['car']<seuil]['target']))
        # print('-------------')
        # print(len(results[results['car']>=seuil]['target']))
        # print('target = 1, >=seuil,',len(results[(results['car']>=seuil) & (results['target']==1)]['target'])/len(results[results['car']>=seuil]['target']))
        # print('target = 0, >=seuil,',len(results[(results['car']>=seuil) & (results['target']==0)]['target'])/len(results[results['car']>=seuil]['target']))
        # verify accuracy
        # op means the default target when < seuil
        total = len(results['target'])
        if op == 0:
            truths = len(results[((results['car']<seuil) & (results['target']==0)) | ((results['car']>=seuil) & (results['target']==1))]['target'])
        else:
            truths = len(results[((results['car']>=seuil) & (results['target']==0)) | ((results['car']<seuil) & (results['target']==1))]['target'])
        print('total =',total)
        print('accuracy =',truths/total)

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
    
    def plotCol(col, BDD=None):
        '''
            draws the graph of col
        '''
        if BDD == None:
            BDD = ''
        else:
            BDD = ' '+BDD
        XData = pd.read_csv('outputs/resnew'+BDD+'.csv', header=None)
        data = XData.loc[1:,col:col]
        labels = XData.loc[1:,2:2]
        plt.plot(labels, data, 'b^', ms=1)
        minY = int(np.min(data, axis=0))
        maxY = int(np.max(data, axis=0)+1)
        plt.axis([-1, 2, minY, maxY])
        plt.ylabel('asymmetryByBestFitEllipse')
        plt.xlabel('class')
        plt.show()
    
    def plotResults(results, fns, fps):
        '''
            draws the results of training, false negatives and positives
        '''
        rest = results[results['prediction'] == results['target']]
        # plt.plot(results['target'], results['car'], 'y.', ms=2)
        plt.plot(rest['prediction'], rest['car'], 'y.',
            fns['prediction'], fns['car'], 'r.',
            fps['prediction'], fps['car'], 'g.', ms=2)
        # print(np.max(fns['car']), np.min(fps['car']))
        neg = results[results['prediction']==0]['car']
        pos = results[results['prediction']==1]['car']
        print(np.max(neg), np.min(pos))
        print(np.min(neg), np.max(pos))
        plt.ylabel('asymmetryByBestFitEllipse')
        plt.xlabel('class')
        # plt.show()
    
    def saveThreshClassification(thresholds, ops, BDD=None):
        '''
            save the thresholds classification results to csv file
        '''
        if BDD == None:
            BDD = ''
        else:
            BDD = ' '+BDD
        XData = pd.read_csv('outputs/resnew'+BDD+'.csv', header=None)
        data = XData.loc[:,4:25]
        labels = XData.loc[:,2:2]
        print(len(labels))
        col = 7
        if ops[col-4] == 0:
            TN = len(data[(data.loc[:,col]<thresholds[col-4]) & (labels.loc[:,2]==0)][col])
            TP = len(data[(data.loc[:,col]>=thresholds[col-4]) & (labels.loc[:,2]==1)][col])
            print(TN, TP)
        else:
            TN = len(data[(data.loc[:,col]>=thresholds[col-4]) & (labels.loc[:,2]==0)][col])
            TP = len(data[(data.loc[:,col]<thresholds[col-4]) & (labels.loc[:,2]==1)][col])
            print(TN, TP)
    
    @staticmethod
    def testThresholds(col, seuil, BDD=None, op=None):
        '''
            test thresholds
        '''
        if BDD == None:
            BDD = ''
        else:
            BDD = ' '+BDD
        XData = pd.read_csv('outputs/resnew'+BDD+'.csv', header=None)
        target = XData.loc[:,2].values
        data = XData.loc[:,col:col].values
        # predict
        target = np.array(target)
        data = np.array(data)
        results = pd.DataFrame()
        for i in range(0, len(data)):
            if op == 0:
                if data[i] < seuil:
                    p = 0
                else:
                    p = 1
            else:
                if data[i] >= seuil:
                    p = 0
                else:
                    p = 1
            t = target[i]
            df = pd.DataFrame({'car':data[i] , 'target':t, 'prediction':p})
            results = results.append(df, ignore_index=True)
        # verify accuracy
        # op means the default target when < seuil
        total = len(results['target'])
        if op == 0:
            truths = len(results[((results['car']<seuil) & (results['target']==0)) | ((results['car']>=seuil) & (results['target']==1))]['target'])
        else:
            truths = len(results[((results['car']>=seuil) & (results['target']==0)) | ((results['car']<seuil) & (results['target']==1))]['target'])
        print('total =',total)
        print('accuracy =',truths/total)
        return results['prediction']

# for col in range(4,26):
#     MainOptimisation.trainOneCol(col)
# MainOptimisation.plotCol(4, BDD='PH2')
# for col in range(4,26):
#     print('-----------------------------')
#     results, fns, fps = MainOptimisation.trainOneCol(col, BDD='ISIC')
#     MainOptimisation.plotResults(results, fns, fps)
# MainOptimisation.testFinals(7, 22.2, BDD='PH2')
# test thresholds for each column (ISIC and PH2)
# thresholds = [5.83, 90.265, 10.695, 13.63, 21.61, 60.255, 65.115, 1087.5, 0.03, 0.645, 1.66, 1.28, 166.5, 1.45, 3.5, 1.5, 3.5, 9.37, 61.35, 390.0, 396.22, 4.37]
# thresholdsPH2 = [5.325, 91.415, 9.375 , 22.2  , 41.59 , 58.57, 41.745, 2240.0, 0.015, 0.575, 2.22, 1.445, 291.5, 0.67, 3.0, 4.0, 5.5, 9.54, 63.84, 697.0, 801.485, 9.035]
thresholdsPH2 = np.array([8.26, 84.69, 14.21, 17.83, 34.77, 16.93, 51.2, 1939, 0.01, 0.51, 1.96, 1.4, 315, 0.8, 1, 4, 6, 7.26, 46.87, 722, 743.65, 9.27])
thresholdsISIC = np.array([4.23, 93.61, 7.31, 12.28, 16.17, 10.18, 73.42, 900, 0.02, 0.71, 1.37, 1.2, 145, 1.6, 3, 2, 3, 10.25, 66.93, 342, 323.27, 3.63])
opsPH2 = [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
opsISIC = [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
predictions = pd.DataFrame()
for col in range(4,26):
    # col = 4
    thresh = thresholdsPH2[col-4]
    # thresh = 8.96
    print('-------------------------------------')
    print('col =', col, thresh)
    predictions[col] = MainOptimisation.testThresholds(col, thresh, BDD='PH2', op=opsPH2[col-4])
print(predictions)
# MainOptimisation.saveThreshClassification(thresholdsPH2, opsPH2, BDD='PH2')
# MainOptimisation.train()
# MainOptimisation.trainSvm()
# MainOptimisation.trainNN()
# MainOptimisation.test()