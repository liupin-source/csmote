
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score,roc_auc_score


def fit_svm(train_x,train_y,C):
    model=svm.SVC(C,kernel='linear',gamma=10,decision_function_shape='ovo')
    model.fit(train_x,train_y.ravel())   
    return model

def score(model,x,y):
    confusionMatrix = confusion_matrix(y, model.predict(x), labels=[1, 0])
    tp = confusionMatrix[0, 0]
    fp = confusionMatrix[0, 1]
    fn = confusionMatrix[1, 0]
    tn = confusionMatrix[1, 1]
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    specificity = tnr = tn / (tn + fp)
    recall = tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall)
    gmean = (recall * specificity) ** 0.5

    recall=recall_score(y, model.predict(x), labels=[1, 0])
    f1 = f1_score(y, model.predict(x), labels=[1, 0])
    auc =roc_auc_score(y, model.predict(x), labels=[1, 0])

    return recall,f1,gmean,auc

if __name__ == '__main__':

    normal_train = np.loadtxt(
            open('/usr/CSMOTE/Datasets/origin/Car/train_maj.csv'),
            delimiter=",",
            skiprows=0)
    normal_train = np.hstack((normal_train,np.zeros((normal_train.shape[0],1))))
    fault_train = np.loadtxt(
            open('/usr/CSMOTE/Datasets/origin/Car/train_min.csv'),
            delimiter=",",
            skiprows=0)
    fault_train = np.hstack((fault_train, np.ones((fault_train.shape[0], 1))))

    train_data = np.vstack((normal_train,fault_train))
    np.random.shuffle(train_data)
    train_x = train_data[:,0:577]
    train_y = train_data[:,577:578]

    normal_test = np.loadtxt(
            open('/usr/CSMOTE/Datasets/origin/Car/test_maj.csv'),
            delimiter=",",
            skiprows=0)
    normal_test = np.hstack((normal_test, np.zeros((normal_test.shape[0], 1))))
    fault_test  = np.loadtxt(
            open('/usr/CSMOTE/Datasets/origin/Car/test_min.csv'),
            delimiter=",",
            skiprows=0)
    fault_test = np.hstack((fault_test, np.ones((fault_test.shape[0], 1))))
    test_data = np.vstack((normal_test ,fault_test ))
    np.random.shuffle(test_data)
    test_x = test_data[:,0:577]
    test_y = test_data[:,577:578]

    C = 10
    model = fit_svm(train_x,train_y,C)
    print(score(model, train_x, train_y))
    print(score(model, test_x, test_y))


