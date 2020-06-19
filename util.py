import sys

import numpy as np


# from nltk.util import ngrams
import pickle as pk


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pk.load(f)


pth = '/hri/localdisk/nnabizad/'


def maxdic(dic):
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]

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
                if len(pred_parents) > 1:
                    if len(target_parents) > 1 and pred_parents[0] == target_parents[0]:
                        correct1 += 1
                    else:
                        incorrect1 += 1
                if len(pred_parents) > 2:
                    if len(target_parents) > 2 and pred_parents[1] == target_parents[1]:
                        correct2 += 1
                    else:
                        incorrect2 += 1
                # else:
                #     print(mydata.decoddict_flat[target])
    accu1, accu2, accu3 = correct1 /(correct1+incorrect1) , correct2/(correct2+incorrect2) , correct3/(correct3+incorrect3)
    print('Flat accuracy Level1:{} , level2:{}, level3:{}'.format(accu1, accu2, accu3))
    return accu1, accu2, accu3

def hierarchical_accuracy(preds, mydata, beam=1):
    incorrect1 = incorrect2 = incorrect3 = correct1 = correct2 = correct3 = 0
    for man in range(len(mydata.hdtest.target)):
        for obj in range(len(mydata.hdtest.target[man])):
            if np.sum(mydata.hdtest.target[man][obj]) == 0:
                break
            else:
                firstpreds = dict()
                secondpreds = dict()
                finalpreds = dict()
                rootinds = [mydata.encoddict_hir[i] for i in mydata.class_hierarchy['<ROOT>']]
                rootpred_args = np.argsort(-preds[man][obj][rootinds])
                for ind in rootpred_args[:beam]:
                    rootpred = rootinds[ind]
                    if mydata.isleaf(mydata.decoddict_hir[rootpred]):
                        finalpreds[rootpred] =  preds[man][obj][rootpred]
                    else:
                        firstpreds[rootpred] = preds[man][obj][rootpred]
                        inds_1 = [mydata.encode(i)[0]  for i in mydata.class_hierarchy[mydata.decoddict_hir[rootpred]]]
                        args_1 = np.argsort(-preds[man][obj][inds_1])
                        for arg in args_1[:beam]:
                            pred_1 = inds_1[arg]
                            full_obj = mydata.decoddict_hir[pred_1] + ' ' + mydata.decoddict_hir[rootpred]
                            if mydata.isleaf(full_obj):
                                finalpreds[pred_1] = preds[man][obj][pred_1]
                            else:
                                secondpreds[rootpred] = preds[man][obj][pred_1] * preds[man][obj][rootpred]
                                inds_2 = [mydata.encode(i)[0] for i in
                                           mydata.class_hierarchy[full_obj]]
                                args_2 = np.argsort(-preds[man][obj][inds_2])
                                for arg2 in args_2[:beam]:
                                    pred_2 = inds_2[arg2]
                                    finalpreds[pred_2] = preds[man][obj][pred_2]

                trueargs = np.argwhere(mydata.hdtest.target[man][obj] ==1)
                if firstpreds: first = maxdic(firstpreds)
                if secondpreds: sec = maxdic(secondpreds)
                final = maxdic(finalpreds)
                if final in trueargs:
                    correct3+=1
                else:
                    incorrect3+=1
                if len(trueargs) >1 and firstpreds:
                    if first in trueargs:
                        correct1+=1
                    else:
                        incorrect1+=1
                if len(trueargs)>2 and secondpreds:
                    if sec in trueargs:
                        correct2 +=1
                    else:
                        incorrect2 +=1

    accu1, accu2, accu3 = correct1 /(correct1+incorrect1) , correct2/(correct2+incorrect2) , correct3/(correct3+incorrect3)
    print('Hierarchical accuracy Level1:{} , level2:{}, level3:{}'.format(accu1, accu2, accu3))
    return accu1, accu2, accu3


if __name__ == '__main__':
    respath = '/home/nnabizad/code/hierarchical/res'
    # data = 'mac_tools'
    data = 'mac_parts'
    # mydata = Data(obj=data)
    # save_obj(mydata, '/hri/localdisk/nnabizad/' + data)
    mydata = load_obj('/hri/localdisk/nnabizad/mac_parts')
    mydata.generate_fold(15)
    preds = mydata.hdtest.target
    hierarchical_accuracy(preds, mydata, 20)