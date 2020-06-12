import time
# from keract import get_activations
from keras.models import load_model

from data import Data
from models import *
from util import output
from pandas import DataFrame

models = [lstm_pred, lstm_sum, lstm_gru, lstm_sif, lstm_sum_zeroh, lstm_contcat]



def accuracy(preds, targets, level=3):
    if level < 3:
        rev_hierarchy = mydata.hierarchy.inverse_hierachy('<ROOT>', level)
    preds = np.argmax(preds, 2)
    targets = np.argmax(targets, 2)
    mans, maxlen = np.shape(targets)
    correct = total = 0
    for i in range(mans):
        for j in range(maxlen):
            if targets[i][j] != 0:
                total += 1
                if level < 3:
                    if rev_hierarchy[targets[i][j]] == rev_hierarchy[preds[i][j]]:
                        correct += 1
                else:
                    if targets[i][j] == preds[i][j]:
                        correct += 1
            else:
                break
    print('correct: {}, total: {}, accuracy: {}'.format(correct, total, correct / total))
    return correct / total


def write_result(hidden_size, dens_size):
    respath = '/home/nnabizad/code/hierarchical/res'
    suffix = 'sig'
    model = models[0]
    seeds = [15, 896783, 9, 12, 45234]
    elapsed_training = []
    elapsed_testing = []
    accu_seeds = np.zeros([3, len(seeds)])
    filename = '{}/{}_{}'.format(respath,suffix,data)
    for seedid in range(len(seeds)):
        seed = seeds[seedid]
        mydata.generate_fold(seed)
        modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(suffix, data, seed)
        start_time = time.time()
        _, history = model(mydata, modelname, seed, hidden_size, dens_size)
        # trained = load_model(modelname)
        DataFrame(history.history).to_csv(
            '{}/logs/{}_{}.csv'.format(respath, data, seed))
    #     predictions = trained.predict(mydata.dtest.input)
    #     tested = time.time()
    #     elapsed_testing.append(tested - start_time)
    #     for level in [1, 2, 3]:
    #         # accu = trained.evaluate(mydata.dtest.input, mydata.dtest.target)[1]
    #         accu = accuracy(predictions, mydata.dtest.target, level)
    #         # elapsed_training.append(trained - start_time)
    #         print(level, accu)
    #         accu_seeds[level - 1][seedid] = accu
    #         # predictions = trained.predict(mydata.dtest.input)
    #         # np.save('{}_{}_lens'.format(data,seed), predictions)
    #         # DataFrame(history.history).to_csv(
    #         #     '/home/nnabizad/code/toolpred/res/logs/{}_seed_{}.csv'.format(data, seed))
    # output({'std: ': np.std(accu_seeds, 1), 'mean: ': np.mean(accu_seeds, 1)}, filename=filename, func='write')
    # output(accu_seeds, filename=filename, func='write')
    # # output('training elapsed time: {}'.format(np.mean(elapsed_training)), filename=filename, func='write')
    # # output(elapsed_training, filename=filename, func='write')
    # output('training time: {}'.format(np.mean(elapsed_testing)), filename=filename, func='write')
    # # output(elapsed_testing, filename=filename, func='write')
    # return filename

    # return 0

def save_layer(layerno):
    suffix = 'tanhsoft'
    seeds = [15, 896783, 9, 12, 45234]
    filepath = '/hri/localdisk/nnabizad/hidata/'
    for seed in seeds[:1]:
        modelname = '/hri/localdisk/nnabizad/models/{}_{}_{}'.format(maxlevel, data, seed)
        saved_model = load_model(modelname)
        mydata.generate_fold(seed)
        layer = [layer.name for layer in saved_model.layers][layerno]
        # predictions = saved_model.predict(mydata.dtrain.input)
        predictions = get_activations(saved_model, mydata.dtrain.input, layer_name=layer, nodes_to_evaluate=None,
                                      output_format='simple',
                                      auto_compile=True)[layer]

        _, _, featurelen = np.shape(predictions)
        Xs = np.empty((0, featurelen))
        Ys = []
        for i in range(len(mydata.train)):
            k=0
            for j in range(len(mydata.train[i])):
                for y in mydata.train[i][j]:
                    if sum(mydata.dtrain.input[i][k])==0:print(i,k)
                    Xs = np.append(Xs, [predictions[i][k]], axis=0)
                    Ys.append(y)
                    k+=1
                # Ys.append([noneadd(y) for y in mydata.train[i][j]])
        np.save(filepath + '{}{}_xtrain_{}'.format(suffix,data, seed), Xs)
        np.save(filepath + '{}{}_ytrain_{}'.format(suffix,data, seed),Ys)
        predictions = get_activations(saved_model, mydata.dtest.input, layer_name=layer, nodes_to_evaluate=None,
                                      output_format='simple',
                                      auto_compile=True)[layer]
        Xs = np.empty((0, featurelen))
        Ys = []
        for i in range(len(mydata.test)):
            for j in range(len(mydata.test[i])):
                k=0
                for y in mydata.test[i][j]:
                    Xs = np.append(Xs, [predictions[i][k]], axis=0)
                    Ys.append(y)
                    k+=1
                # Ys.append([noneadd(y) for y in mydata.test[i][j]])
        np.save(filepath + '{}{}_xtest_{}'.format(suffix, data,  seed), Xs)
        np.save(filepath + '{}{}_ytest_{}'.format(suffix, data,  seed),Ys)
    return 0

if __name__ == '__main__':
    data = 'mac_tools'
    # data = 'mac_parts'
    mydata = Data(obj=data)
    write_result(512, 512)
    # save_layer(2)
