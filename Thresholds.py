import pandas as pd
import numpy as np

def getBestThresholds(col, seuil, BDD=None, op=None):
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
    accuracies = np.array([])
    total = len(data)
    results = pd.DataFrame()
    for i in range(0, len(data)):
        t = target[i]
        df = pd.DataFrame({'car':data[i] , 'target':t})
        results = results.append(df, ignore_index=True)
    for seuil in data:
        seuil = seuil[0]
        # verify accuracy
        # op means the default target when < seuil
        if op == 0:
            truths = len(results[((results['car']<seuil) & (results['target']==0)) | ((results['car']>=seuil) & (results['target']==1))]['target'])
        else:
            truths = len(results[((results['car']>=seuil) & (results['target']==0)) | ((results['car']<seuil) & (results['target']==1))]['target'])
        # print('total =',total)
        # print('accuracy =',truths/total)
        # print(np.mean(results[results['target']==1]['car']))
        accuracies = np.append(accuracies, truths/total)
    mx = np.argmax(accuracies)
    print(accuracies[mx], data[mx])
    if len(data[(data<data[mx])]) < len(data[(data>=data[mx])]):
        print(1)
    else:
        print(0)

# old thresholds
# thresholdsPH2 = [5.325, 91.415, 9.375 , 22.2  , 41.59 , 58.57, 41.745, 2240.0, 0.015, 0.575, 2.22, 1.445, 291.5, 0.67, 3.0, 4.0, 5.5, 9.54, 63.84, 697.0, 801.485, 9.035]
# thresholdsISIC = [4.705, 92.85, 9.53, 12.62, 19.515, 14.025, 68.535, 925.0, 0.035, 0.685, 1.525, 1.235, 144.5, 1.615, 2.5, 1.5, 2.5, 9.96, 64.73, 338.0, 335.815, 3.725]
# new thresholds
thresholdsPH2 = [8.26, 84.69, 14.21, 17.83, 34.77, 16.93, 51.2, 1939, 0.01, 0.51, 1.96, 1.4, 315, 0.8, 1, 4, 6, 7.26, 46.87, 722, 743.65, 9.27]
thresholdsISIC = [4.23, 93.61, 7.31, 12.28, 16.17, 10.18, 73.42, 900, 0.02, 0.71, 1.37, 1.2, 145, 1.6, 3, 2, 3, 10.25, 66.93, 342, 323.27, 3.63]
opsPH2 = [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
opsISIC = [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

for col in range(4,26):
    # col = 4
    # thresh = thresholdsPH2[col-4]
    thresh = 19
    print('-------------------------------------')
    print('col =', col, thresh)
    getBestThresholds(col, thresh, BDD='ISIC', op=opsISIC[col-4])