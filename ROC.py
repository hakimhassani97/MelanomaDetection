import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# XData = pd.read_csv('D:/HAKIM/MIV M2/PFE/stats/results PH2 used.txt', header=None)
# print(XData)
# y_test = XData.loc[:,1].values
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(0,7):
#     y_score = XData.loc[:,i+2].values
#     # Compute ROC curve and ROC area for each class
#     fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # plot results
# plt.figure()
# lw = 2
# plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='Théorie des jeux (surface = %0.2f)' % roc_auc[0], linestyle='-')
# plt.plot(fpr[1], tpr[1], color='grey', lw=lw, label='Asymetrie (sufrace = %0.2f)' % roc_auc[1], linestyle='--')
# plt.plot(fpr[2], tpr[2], color='red', lw=lw, label='Bordure (sufrace = %0.2f)' % roc_auc[2], linestyle=':')
# plt.plot(fpr[3], tpr[3], color='green', lw=lw, label='Couleur (sufrace = %0.2f)' % roc_auc[3], linestyle='-.')
# plt.plot(fpr[4], tpr[4], color='blue', lw=lw, label='Diametre (sufrace = %0.2f)' % roc_auc[4], linestyle='-.')
# plt.plot(fpr[5], tpr[5], color='brown', lw=lw, label='7PCL (sufrace = %0.2f)' % roc_auc[5], linestyle='-.')
# plt.plot(fpr[6], tpr[6], color='violet', lw=lw, label='Menzies (sufrace = %0.2f)' % roc_auc[6], linestyle=':')
# # plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='-')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('taux de faux positifs')
# plt.ylabel('taux de vrais positifs')
# plt.title('Fonction d’efficacité du récepteur (courbe ROC)')
# plt.legend(loc="lower right")
# plt.show()

XData = pd.read_csv('D:/HAKIM/MIV M2/PFE/stats/fake.txt', header=None)
print(XData)
y_test = XData.loc[:,1].values
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,6):
    y_score = XData.loc[:,i+2].values
    # Compute ROC curve and ROC area for each class
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plot ISIC and PH2 results
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='Notre approche THJ (surface = %0.2f)' % roc_auc[0], linestyle='-')
plt.plot(fpr[1], tpr[1], color='grey', lw=lw, label='Djelouah (sufrace = %0.2f)' % roc_auc[1], linestyle='--')
plt.plot(fpr[2], tpr[2], color='red', lw=lw, label='Adabbost, Pennisi et al (sufrace = %0.2f)' % roc_auc[2], linestyle=':')
plt.plot(fpr[3], tpr[3], color='green', lw=lw, label='Random Trees, Pennisi et al (sufrace = %0.2f)' % roc_auc[3], linestyle='-.')
plt.plot(fpr[4], tpr[4], color='blue', lw=lw, label='Bayes, Pennisi et al (sufrace = %0.2f)' % roc_auc[4], linestyle='-.')
plt.plot(fpr[5], tpr[5], color='brown', lw=lw, label='KNN, Pennisi et al (sufrace = %0.2f)' % roc_auc[5], linestyle='-.')
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='-')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('taux de faux positifs')
plt.ylabel('taux de vrais positifs')
plt.title('Fonction d’efficacité du récepteur (courbe ROC)')
plt.legend(loc="lower right")
plt.show()

