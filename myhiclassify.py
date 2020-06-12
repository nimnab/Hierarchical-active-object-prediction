import logging

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sys
from sklearn.multiclass import OneVsRestClassifier
# from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from HierarchicalClassifier import Hierarchy
from data import Data
logging.basicConfig(level=logging.DEBUG)
from util import save_obj, load_obj


classifiers = [
    SVC(kernel="linear", C=0.025, class_weight='balanced'),
    SVC(kernel="rbf", gamma="auto", C=100, probability=True, class_weight='balanced'),
    tree.DecisionTreeClassifier(min_samples_split=10),
    RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced'),
    MLPClassifier(solver='lbfgs', hidden_layer_sizes=(256, 128), random_state=1,
                  max_iter=1000, learning_rate='adaptive', learning_rate_init=0.001, activation='tanh'),
]

names = ["linear", "rbf", "DecisionTreeClassifier", "RandomForestClassifier", "MLPClassifier"]



def myclassifier(index):
    # bclf = OneVsRestClassifier(classifiers[2])
    # clf = classifiers[index]
    model = Hierarchy(data=mydata,  ytrain=Y_train, xtrain=X_train, xtest=X_test, ytest=Y_test)
    alphas = np.logspace(-10, 3, 10)
    for alpha in alphas:
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(256, 128), random_state=1,
                      max_iter=2000, learning_rate='adaptive', learning_rate_init=0.001, activation='tanh', alpha=alpha)
        print(suffix, alpha)
        model.testclf(clf)
    # model.fit(xtrain=X_train, ytrain=Y_train, xtest=X_test)
    # for level in (1,2,3):
    #     preds = model.predict(xtest=X_test)
    #     print(level, model.accuracy(preds=preds,targets=Y_test,level=level))


if __name__ == '__main__':
    suffix = 'tanhsig'
    # suffix = ''
    data = 'mac_tools'
    # data = 'mac_parts'
    seeds = [15, 896783, 9, 12, 45234]
    filepath = '/hri/localdisk/nnabizad/hidata/'
    # mydata = Data(usew2v=False, title=False, tool_output=True, obj=data)
    # save_obj(mydata, filepath+'mydata'+data)
    mydata = load_obj( filepath+'mydata'+data)
    seed = seeds[0]
    X_train = np.load(filepath + '{}{}_xtrain_{}.npy'.format(suffix,data, seed))
    Y_train = np.load(filepath + '{}{}_ytrain_{}.npy'.format(suffix,data, seed))
    X_test = np.load(filepath + '{}{}_xtest_{}.npy'.format(suffix,data,  seed))
    Y_test = np.load(filepath + '{}{}_ytest_{}.npy'.format(suffix,data,  seed))
    index = 4
    myclassifier(index=index)
