from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
import numpy as np
import matplotlib.pyplot as plt








def smote(x,y):
    sm = SMOTE()
    x_res, y_res = sm.fit_resample(x, y)
    return x_res, y_res


def adasyn(x,y):
    ada = ADASYN()
    x_res, y_res = ada.fit_resample(x, y)
    return x_res, y_res

# x, y = make_classification(n_classes=2, class_sep=2,
#                            weights=[0.1, 0.9], n_informative=2, n_redundant=0, flip_y=0,
#                            n_features=2, n_clusters_per_class=1, n_samples=100,random_state=10)

def ros(x,y):
    ros = RandomOverSampler()
    x_res, y_res = ros.fit_resample(x, y)
    return  x_res,y_res

normal_train = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/train_maj.csv'),
        delimiter=",",
        skiprows=0)
fault_train = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/train_min.csv'),
        delimiter=",",
        skiprows=0)

x = np.vstack((normal_train,fault_train))
y = np.vstack((np.zeros((normal_train.shape[0],1)),np.ones((fault_train.shape[0],1))))
#
x_res, y_res = adasyn(x,y)
x_res= x_res[49:96]
np.savetxt('/usr/CSMOTE/Datasets/synthistic/Car/adasyn_' + str(50) + '.csv', x_res,
               delimiter=',')
