import pickle as pk

import numpy as np
from keras.models import load_model
from pandas import DataFrame

from data import Data
from models import *
from util import *


def save_obj(obj, name):
    """
    Saving the pickle object
    """
    with open(name + '.pkl', 'wb') as file:
        pk.dump(obj, file, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    loading the pickle object
    """
    with open(name + '.pkl', 'rb') as file:
        return pk.load(file)


def hierarchical_accuracy_beam(preds, mydata, beam=3):
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
                args = np.argsort(-preds[man][obj][mydata.level1])
                level2s = dict()
                level3s = dict()
                for _obj in args[:beam]:
                    objcandidate = mydata.decoddict_hir[mydata.level1[_obj]]
                    objpred = preds[man][obj][mydata.level1[_obj]]
                    if objcandidate in mydata.class_hierarchy:
                        children = mydata.class_hierarchy[objcandidate]
                        for child in children[:]:
                            ch, p = mydata.encode(child)
                            childpred = preds[man][obj][ch]
                            level2s[ch]= childpred * objpred
                            if child in mydata.class_hierarchy:
                                grandchildren = mydata.class_hierarchy[child]
                                for grandchild in grandchildren[:]:
                                    ch, _, _ = mydata.encode(grandchild)
                                    level3s[ch] = preds[man][obj][ch] * childpred * objpred
                if level2s:
                    lev2 = max(level2s.keys(), key=(lambda k: level2s[k]))
                    if lev2 == mydata.level2[np.argmax(mydata.dtest.target[man][obj][mydata.level2])]:
                        correct2 +=1
                    else:
                        incorrect2 +=1
                if level3s:
                    lev3 = max(level3s.keys(), key=(lambda k: level3s[k]))
                    if lev3 == mydata.level3[np.argmax(mydata.dtest.target[man][obj][mydata.level3])]:
                        correct3 +=1
                    else:
                        incorrect3 +=1

    accu1, accu2, accu3 = correct1 / (correct1 + incorrect1), correct2 / (correct2 + incorrect2), correct3 / (
            correct3 + incorrect3)
    print('Level1:{} , level2:{}, level3:{}'.format(accu1, accu2, accu3))
    return accu1, accu2, accu3


def write_result_hir(hidden_size, dens_size):
    respath = '/home/nnabizad/code/hierarchical/res'
    suffix = 'hir'
    seeds = [15, 896783, 9, 12, 45234]
    accu_seeds = np.zeros([3, len(seeds)])
    filename = '{}/{}_{}'.format(respath, suffix, data)
    for seedid in range(len(seeds)):
    # for seedid in [0]:
        seed = seeds[seedid]
        mydata.generate_fold(seed, hierarchical=True)
        modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(suffix, data, seed)
        # _, history = lstm_pred_hierarchical(mydata, modelname, seed, hidden_size, dens_size)
        trained = load_model(modelname)
        # DataFrame(history.history).to_csv(
        #     '{}/logs/{}_{}.csv'.format(respath, data, seed))
        predictions = trained.predict(mydata.dtest.input)
        accu1, accu2, accu3 = hierarchical_accuracy_beam(predictions, mydata)
        output([accu1, accu2, accu3], filename=filename, func='write')

def write_result_flat(hidden_size, dens_size):
    respath = '/home/nnabizad/code/hierarchical/res'
    suffix = 'flat'
    seeds = [15, 896783, 9, 12, 45234]
    accu_seeds = np.zeros([3, len(seeds)])
    filename = '{}/{}_{}'.format(respath, suffix, data)
    for seedid in range(len(seeds)):
    # for seedid in [0]:
        seed = seeds[seedid]
        mydata.generate_fold(seed, hierarchical=False)
        modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(suffix, data, seed)
        _, history = lstm_pred(mydata, modelname, seed, hidden_size, dens_size)
        trained = load_model(modelname)
        DataFrame(history.history).to_csv(
            '{}/logs/{}_{}.csv'.format(respath, data, seed))
        predictions = trained.predict(mydata.dtest.input)
        accu1, accu2, accu3 = flat_accuracy(predictions, mydata)
        output([accu1, accu2, accu3], filename=filename, func='write')


if __name__ == '__main__':
    data = 'mac_tools'
    # data = 'mac_parts'
    # mydata = Data(obj=data)
    # save_obj(mydata, '/hri/localdisk/nnabizad/' + data)
    mydata = load_obj('/hri/localdisk/nnabizad/'+data)
    write_result_hir(512, 512)
    # write_result_flat(512, 512)
    # save_layer(2)
