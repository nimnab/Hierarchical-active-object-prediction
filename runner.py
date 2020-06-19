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


def write_result_hir(hidden_size, dens_size, regul):
    suffix = 'hir_soft'
    seeds = [15, 896783, 9, 12, 45234]
    accu_seeds = np.zeros([len(seeds), 3])
    filename = '{}/{}_{}'.format(respath, suffix, data)
    for seedid in range(len(seeds)):
        # for seedid in [0]:
        seed = seeds[seedid]
        mydata.generate_fold(seed)
        modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}_{}'.format(suffix, data, regul, seed)
        _, history = lstm_pred_hierarchical(mydata, modelname, hidden_size, dens_size, regul)
        trained = load_model(modelname)
        DataFrame(history.history).to_csv(
            '{}/logs/{}_{}_{}.csv'.format(respath, data, regul, seed))
        predictions = trained.predict(mydata.hdtest.input)
        # for beam in [1,2,3,4,5,10]:
        accu1, accu2, accu3 = hierarchical_accuracy(predictions, mydata, beam=1)
        accu_seeds[seedid] = accu1, accu2, accu3
    means = np.mean(accu_seeds, 0)
    stds = np.std(accu_seeds, 0)
    output('regul: {}: mean accuracy l1:{}, l2:{}, l3:{}, sd1:{}, sd2:{}, sd3:{}'.format(regul, means[0], means[1],
                                                                                         means[2], stds[0], stds[1],
                                                                                         stds[2]), filename=filename,
           func='write')


def write_result_flat(hidden_size, dens_size, regul):
    suffix = 'flat'
    seeds = [15, 896783, 9, 12, 45234]
    accu_seeds = np.zeros([len(seeds), 3])
    filename = '{}/{}_{}'.format(respath, suffix, data)
    for seedid in range(len(seeds)):
        # for seedid in [0]:
        seed = seeds[seedid]
        mydata.generate_fold(seed)
        modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}_{}'.format(suffix, data, regul, seed)
        _, history = lstm_pred(mydata, modelname, hidden_size, dens_size, regul)
        trained = load_model(modelname)
        DataFrame(history.history).to_csv(
            '{}/logs/{}_{}_{}.csv'.format(respath, data, regul, seed))
        predictions = trained.predict(mydata.dtest.input)
        # for beam in [1,2,3,4,5,10]:
        accu1, accu2, accu3 = flat_accuracy(predictions, mydata)
        accu_seeds[seedid] = accu1, accu2, accu3
    means = np.mean(accu_seeds, 0)
    stds = np.std(accu_seeds, 0)
    output('regul: {}: mean accuracy l1:{}, l2:{}, l3:{}, sd1:{}, sd2:{}, sd3:{}'.format(regul, means[0], means[1],
                                                                                         means[2], stds[0], stds[1],
                                                                                         stds[2]), filename=filename,
           func='write')



# def compare():
#     seed = 15
#     flatmodel = load_model('/hri/localdisk/nnabizad/models/{}_{}_{}'.format('flat', data, seed))
#     hirmodel = load_model('/hri/localdisk/nnabizad/models/{}_{}_{}'.format('hir', data, seed))
#     mydata.generate_fold(seed)
#     flatpreds = flatmodel.predict(mydata.dtest.input)
#     # mydata.generate_fold(seed, hierarchical=True)
#     hirpreds = hirmodel.predict(mydata.hdtest.input)
#     mans , lens , _ = np.shape(flatpreds)
#     for man in range(mans):
#         for obj in range(lens):
#             if np.sum(mydata.dtest.target[man][obj]) != 0:
#                 flatpred = mydata.decoddict_flat[np.argmax(flatpreds[man][obj])]
#                 hirpred = mydata.decoddict_hir[np.argmax(hirpreds[man][obj])]
#                 inputflat = mydata.decoddict_flat[np.argmax(mydata.dtest.input[man][obj])]
#                 inputhir = mydata.decoddict_hir[np.argmax(mydata.hdtest.input[man][obj])]
#                 targetflat = mydata.decoddict_flat[np.argmax(mydata.dtest.target[man][obj])]
#                 targethir = mydata.decoddict_hir[np.argmax(mydata.hdtest.target[man][obj])]
#                 accu1, accu2, accu3 = flat_accuracy(flatpreds, mydata, man, obj)
#                 haccu1, haccu2, haccu3 = hierarchical_accuracy(hirpreds, mydata, man, obj)
#                 print('flat: ', accu1, accu2, accu3 )
#                 print('hit: ', haccu1, haccu2, haccu3 )
#                 print()

if __name__ == '__main__':
    respath = '/home/nnabizad/code/hierarchical/res'
    # data = 'mac_tools'
    data = 'mac_parts'
    # mydata = Data(obj=data)
    # save_obj(mydata, '/hri/localdisk/nnabizad/' + data)
    mydata = load_obj('/hri/localdisk/nnabizad/'+data)
    reguls = [0, 1e-6, 1e-4, 1e-2, 0.1, 0.5]
    for regul in reguls:
        write_result_hir(256, 256, regul)
        # write_result_flat(256, 256, regul)
