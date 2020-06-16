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
        # _, history = lstm_pred(mydata, modelname, seed, hidden_size, dens_size)
        trained = load_model(modelname)
        # DataFrame(history.history).to_csv(
        #     '{}/logs/{}_{}.csv'.format(respath, data, seed))
        predictions = trained.predict(mydata.dtest.input)
        accu1, accu2, accu3 = hierarchical_accuracy(predictions, mydata)
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
        # _, history = lstm_pred(mydata, modelname, seed, hidden_size, dens_size)
        trained = load_model(modelname)
        # DataFrame(history.history).to_csv(
        #     '{}/logs/{}_{}.csv'.format(respath, data, seed))
        predictions = trained.predict(mydata.dtest.input)
        accu1, accu2, accu3 = flat_accuracy(predictions, mydata)
        output([accu1, accu2, accu3], filename=filename, func='write')


if __name__ == '__main__':
    data = 'mac_tools'
    # data = 'mac_parts'
    # mydata = Data(obj=data)
    # save_obj(mydata, '/hri/localdisk/nnabizad/' + data)
    mydata = load_obj('/hri/localdisk/nnabizad/'+data)
    # write_result_hir(512, 512)
    write_result_flat(512, 512)
    # save_layer(2)
