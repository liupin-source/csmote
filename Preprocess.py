

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


train = pd.read_csv('/usr/project/Tensorflow/UCR/FordA/FordA_TRAIN.tsv',sep='\t', header=None).to_numpy()
train_min=np.empty((0,500))
train_maj=np.empty((0,500))
for i in range(train.shape[0]):
    if (train[i, 0:1] == 1):
        train_min = np.vstack((train_min, train[i, 1:501]))
    else:
        train_maj = np.vstack((train_maj, train[i, 1:501]))

test = pd.read_csv('/usr/project/Tensorflow/UCR/FordA/FordA_TEST.tsv',sep='\t', header=None).to_numpy()

test_min=np.empty((0,500))
test_maj=np.empty((0,500))
for i in range(test.shape[0]):
    if (test[i, 0:1] == 1):
        test_min = np.vstack((test_min, test[i, 1:501]))
    else:
        test_maj = np.vstack((test_maj, test[i, 1:501]))

car = np.vstack((train_maj,train_min,test_maj,test_min))

scalar = MinMaxScaler(feature_range=(0, 1)) 
car = scalar.fit_transform(car)

train_maj=car[0:train_maj.shape[0],]
train_min=car[train_maj.shape[0]:(train_maj.shape[0]+train_min.shape[0]),]
test_maj=car[(train_maj.shape[0]+train_min.shape[0]):(train_maj.shape[0]+train_min.shape[0]+test_maj.shape[0]),]
test_min=car[(train_maj.shape[0]+train_min.shape[0]+test_maj.shape[0]):(train_maj.shape[0]+train_min.shape[0]+test_maj.shape[0]+test_min.shape[0]),]


np.random.shuffle(train_min)
train_min_used = train_min[0:184]
train_min_remain = train_min[184:1755]
# #
np.random.shuffle(test_min)
test_min_used = test_min[0:68]
test_min_remain = test_min[68:639]
#
np.savetxt('/usr/SiScort/Datasets/origin/FordA/train_maj.csv', train_maj, delimiter=',')
np.savetxt('/usr/SiScort/Datasets/origin/FordA/train_min.csv', train_min, delimiter=',')
np.savetxt('/usr/SiScort/Datasets/origin/FordA/test_maj.csv', test_maj, delimiter=',')
np.savetxt('/usr/SiScort/Datasets/origin/FordA/test_min.csv', test_min, delimiter=',')

np.savetxt('/usr/SiScort/Datasets/origin/FordA/train_min_used.csv', train_min_used, delimiter=',')
np.savetxt('/usr/SiScort/Datasets/origin/FordA/train_min_remain.csv', train_min_remain, delimiter=',')
np.savetxt('/usr/SiScort/Datasets/origin/FordA/test_min_used.csv', test_min_used, delimiter=',')
np.savetxt('/usr/SiScort/Datasets/origin/FordA/test_min_remain.csv', test_min_remain, delimiter=',')

