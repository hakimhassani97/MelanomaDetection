import pandas as pd
import numpy as np
import nashpy as nash
import  matplotlib.pyplot as plt

def load():
    '''
        load data from csv
    '''
    XData = pd.read_csv('outputs/resnew PH2.csv', header=None)
    target = XData.loc[:,2].values
    data = XData.loc[:,4:25].values
    return target, data

def getDataMatrix(s, t, minimum):
    '''
        fill the M matrix with data and (M, data, target)\n
        s is an array of methodes for a strategy s = [0, 1, ..., 21]
    '''
    results = pd.DataFrame(data)
    # ABCD of melanome
    AMelanome = []
    for i in range(0, len(data)):
        cols = results.loc[i].values
        # melanome sStarts[s]:sEnds[s]
        A = cols[s]
        if t==1:
            carMelanome = A[
                ((opsPH2[s]==0) & (A>=thresholdsPH2[s]))
                | ((opsPH2[s]==1) & (A<thresholdsPH2[s]))
            ]
        else:
            carMelanome = A[
                ((opsPH2[s]==0) & (A<thresholdsPH2[s]))
                | ((opsPH2[s]==1) & (A>=thresholdsPH2[s]))
            ]
        if len(carMelanome) >= minimum:# and target[i]==t:
            AMelanome.append(A)
            # print(carMelanome)
    return AMelanome

def getColumnsToUse(T):
    '''
        get the columns to use in each strategy methods for a sample image caracteristiques T
    '''
    cars = ((opsPH2==0) & (T<thresholdsPH2)) | ((opsPH2==1) & (T>=thresholdsPH2))
    cars = np.logical_not(cars)
    cars = np.array(cars, dtype=np.int)
    return cars

def getColsFromStrategy(s, colsToUse):
    '''
        return columns for a strategy s
    '''
    return colsToUse[(colsToUse>=sStarts[s]) & (colsToUse<sEnds[s])]

def distance(T, AMelanome, t):
    '''
        returns the distance of the caracteristiques vector T and the AMelanomes of target == t
    '''
    AMelanome = np.array(AMelanome)
    mean = np.mean(AMelanome, axis=0)
    sigma = np.std(AMelanome, axis=0, ddof=1)
    Z = np.subtract(AMelanome, mean)
    sigma[sigma==0] = 1
    Z = np.divide(Z, sigma)
    R = np.dot(Z.T, Z)
    R = np.multiply(R, 1/len(AMelanome))
    Tz = np.subtract(T, mean)
    Tz = np.divide(Tz, sigma)
    diff = np.subtract(Tz, Z)
    nn = np.linalg.norm(diff, axis=1)
    # nn = np.sqrt(np.add(np.power(Tz, 2), np.power(Z, 2)))
    # print(np.shape(nn))
    Y = np.argmin(nn, axis=0)
    Y = Z[Y]
    V = np.subtract(Tz, Y)
    d = np.dot(R, V)
    d = np.dot(V.T, d)
    if t==0:
        if d != 0:
            d = 1/d
        else:
            # d = 999999
            pass
    return d

def Utility(d1, d2):
    '''
        Utility functions between S1 and S2
    '''
    return d1 - d2

# information
cars = range(4, 26)
cars = np.array(cars)
thresholdsPH2 = np.array([8.26, 84.69, 14.21, 17.83, 34.77, 16.93, 51.2, 1939, 0.01, 0.51, 1.96, 1.4, 315, 0.8, 1, 4, 6, 7.26, 46.87, 722, 743.65, 9.27])
thresholdsISIC = np.array([4.23, 93.61, 7.31, 12.28, 16.17, 10.18, 73.42, 900, 0.02, 0.71, 1.37, 1.2, 145, 1.6, 3, 2, 3, 10.25, 66.93, 342, 323.27, 3.63])
opsPH2 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
opsISIC = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
# strategies info
sLengths = [6, 8, 5, 3]
sStarts = [0, 6, 14, 19]
sEnds = [6, 14, 19, 22]
# load data
target, data = load()
ds = []
i = 130
for i in range(0, len(data)):
    # take a sample T
    T = data[i]
    cars = getColumnsToUse(T)
    cols = range(0, 22)
    cols = np.array(cols)
    sMelanome = cols[cars==1]
    sNonMelanome = cols[cars==0]
    print(cars, sMelanome, sNonMelanome)
###################################
# # strategy
# s = 0
# # target (player)
# t = 1
# sMelanome0 = getColsFromStrategy(s, sMelanome)
# AMelanome = getDataMatrix(sMelanome0, t=t, minimum=len(sMelanome0))
# # get the distance between T and AMelanome
# d = distance(T[sMelanome0], AMelanome, t)
# print(target[i], d)
###################################
# fill distances for player 1 (t==1)
    t = 1
    d1 = []
    for s1 in range(0, 4):
        sMelanome1 = getColsFromStrategy(s1, sMelanome)
        if len(sMelanome1)>0:
            mins = [[6, 8, 5, 3], [6, 7, 2, 3]]
            maximum = min([len(sMelanome1), mins[t][s1]])
            M = getDataMatrix(sMelanome1, t=t, minimum=maximum)
            M = np.array(M)
            # get the distance between T and M
            if(len(M)>0):
                d = distance(T[sMelanome1], M, t)
                d1.append(d)
    d1 = np.array(d1)
    print('d1 =', d1)
    # fill distances for player 2 (t==0)
    t = 1
    d2 = []
    for s2 in range(0, 4):
        sMelanome2 = getColsFromStrategy(s2, sNonMelanome)
        if len(sMelanome2)>0:
            mins = [[6, 8, 5, 3], [6, 7, 2, 3]]
            maximum = min([len(sMelanome2), mins[t][s2]])
            M = getDataMatrix(sMelanome2, t=t, minimum=maximum)
            M = np.array(M)
            # get the distance between T and M
            if(len(M)>0):
                d = distance(T[sMelanome2], M, t)
                d2.append(d)
    d2 = np.array(d2)
    print('d2 =', d2)
# construct the game
game = np.zeros((len(d1), len(d2)))
for i in range(0, len(d1)):
    for j in range(0, len(d2)):
        game[i, j] = Utility(d1[i], d2[j])
print(game)



##################
# ds = np.array(ds)
# df = pd.DataFrame()
# df[0] = target
# df[1] = ds
# df = df.values
# print('1 =',np.count_nonzero(df[df[:,1]==0][:,0]==1), ', 0 =',np.count_nonzero(df[df[:,1]==0][:,0]==0))
# print(df[df[:,1]==0])
# data = ds
# labels = target
# plt.plot(labels, data, 'g.', ms=1)
# minY = int(np.min(data, axis=0))
# maxY = int(np.max(data, axis=0)+1)
# plt.axis([-1, 2, minY, maxY])
# plt.ylabel('asymmetryByBestFitEllipse')
# plt.xlabel('class')
# plt.show()