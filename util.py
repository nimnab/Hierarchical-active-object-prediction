import pickle as pk
import sys

import numpy as np


# from nltk.util import ngrams


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pk.load(f)


pth = '/hri/localdisk/nnabizad/'


def output(input, filename=pth + "corpus.txt", func=print, nonextline=False):
    if func:
        file = None
        end = ''
        if func == 'write':
            file = open(filename, 'a')
            func = file.write
            end = '\n'
        elif nonextline:
            func = sys.stdout.write
        if type(input) == list or type(input) == np.ndarray:
            for l in input:
                func(str(l) + end)
        elif type(input) == dict:
            for i in input:
                func(str(i) + ' : ' + str(input[i]) + end)
        else:
            func(str(input) + end)
        if file: file.close()


def hierarchical_accuracy(preds, mydata):
    incorrect1 = incorrect2 = incorrect3 = correct1 = correct2 = correct3 = 0
    for man in range(len(mydata.dtest.target)):
        for obj in range(len(mydata.dtest.target[man])):
            if np.sum(mydata.dtest.target[man][obj]) == 0:
                break
            else:
                lev1 = mydata.level1[np.argmax(mydata.dtest.target[man][obj][mydata.level1])]
                lev1p = mydata.level1[np.argmax(preds[man][obj][mydata.level1])]
                if lev1 == lev1p:
                    correct1 += 1
                else:
                    incorrect1 += 1
                if not mydata.isleaf(mydata.decoddict_hir[lev1p]) or not mydata.isleaf(mydata.decoddict_hir[lev1]):
                    lev2 = mydata.level2[np.argmax(mydata.dtest.target[man][obj][mydata.level2])]
                    lev2p = mydata.level2[np.argmax(preds[man][obj][mydata.level2])]
                    if lev2 == lev2p:
                        correct2 += 1
                    else:
                        incorrect2 += 1
                    if not mydata.isleaf(mydata.decoddict_hir[lev2p], noparent=True) or not mydata.isleaf(
                            mydata.decoddict_hir[lev2], noparent=True):
                        lev3 = mydata.level3[np.argmax(mydata.dtest.target[man][obj][mydata.level3])]
                        lev3p = mydata.level3[np.argmax(preds[man][obj][mydata.level3])]
                        if lev3 == lev3p:
                            correct3 += 1
                        else:
                            incorrect3 += 1
    accu1, accu2, accu3 = correct1 / (correct1 + incorrect1), correct2 / (correct2 + incorrect2), correct3 / (
            correct3 + incorrect3)
    print('Level1:{} , level2:{}, level3:{}'.format(accu1, accu2, accu3))
    return accu1, accu2, accu3


#
def flat_accuracy(preds, mydata):
    incorrect1 = incorrect2 = incorrect3 = correct1 = correct2 = correct3 = 0
    for man in range(len(mydata.dtest.target)):
        for obj in range(len(mydata.dtest.target[man])):
            if np.sum(mydata.dtest.target[man][obj]) == 0:
                break
            else:
                target = np.argmax(mydata.dtest.target[man][obj])
                pred = np.argmax(preds[man][obj])
                if target == pred and target != mydata.encoddict_flat['UNK']:
                    correct3 += 1
                else:
                    incorrect3 += 1
                target_parents = mydata.reverse_hierarchy[mydata.decoddict_flat[target]]
                pred_parents = mydata.reverse_hierarchy[mydata.decoddict_flat[pred]]
                if len(target_parents) == 3 and len(pred_parents) == 3:
                    if pred_parents[0] == target_parents[0]:
                        correct1 += 1
                    else:
                        incorrect1 += 1
                    if pred_parents[1] == target_parents[1]:
                        correct2 += 1
                    else:
                        incorrect2 += 1
    accu1, accu2, accu3 = correct1 / (correct1 + incorrect1), correct2 / (correct2 + incorrect2), correct3 / (
            correct3 + incorrect3)
    print('Level1:{} , level2:{}, level3:{}'.format(accu1, accu2, accu3))
    return accu1, accu2, accu3
