import numpy as np
from keras.models import load_model
from util import output
from data import Data
from models import lstm_pred , lstm_pred_hierarchical
import pickle as pk
from pandas import DataFrame


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
                if not mydata.isleaf(mydata.decoddict[lev1p]) or not mydata.isleaf(mydata.decoddict[lev1]):
                    lev2 = mydata.level2[np.argmax(mydata.dtest.target[man][obj][mydata.level2])]
                    lev2p = mydata.level2[np.argmax(preds[man][obj][mydata.level2])]
                    if lev2 == lev2p:
                        correct2 += 1
                    else:
                        incorrect2 += 1
                    if not mydata.isleaf(mydata.decoddict[lev2p], noparent=True) or not mydata.isleaf(mydata.decoddict[lev2], noparent=True):
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


def write_result(hidden_size, dens_size):
    respath = '/home/nnabizad/code/hierarchical/res'
    suffix = 'tsig'
    seeds = [15, 896783, 9, 12, 45234]
    accu_seeds = np.zeros([3, len(seeds)])
    filename = '{}/{}_{}'.format(respath, suffix, data)
    # for seedid in range(len(seeds)):
    for seedid in [0]:
        seed = seeds[seedid]
        mydata.generate_fold(seed)
        modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(suffix, data, seed)
        _, history = lstm_pred_hierarchical(mydata, modelname, seed, hidden_size, dens_size)
        trained = load_model(modelname)
        DataFrame(history.history).to_csv(
            '{}/logs/{}_{}.csv'.format(respath, data, seed))
        predictions = trained.predict(mydata.dtest.input)
        accu1, accu2, accu3 = hierarchical_accuracy(predictions, mydata)
        output([accu1, accu2, accu3], filename=filename, func='write')


if __name__ == '__main__':
    # data = 'mac_tools'
    data = 'mac_parts'
    mydata = Data(obj=data, hierarchical=True)
    save_obj(mydata, data)
    write_result(512, 512)
    # save_layer(2)
