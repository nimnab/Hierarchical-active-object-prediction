import numpy as np
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import TimeDistributed, Input, LSTM, Dense, Masking, Lambda, concatenate, RepeatVector, GRU, Dropout, \
    Permute
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# dr = 0.2

def lstm_pred_hierarchical(mydata, modelname,  hidden_size, dens2_size , regul, dr):
    # for seed in seeds:
    _, seqlength, obj_number = np.shape(mydata.hdtest.target)
    featurelen = np.shape(mydata.hdtest.input[0])[1]

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(seqlength, featurelen)))
    model.add(LSTM(hidden_size, return_sequences=True, recurrent_dropout=dr))

    model.add(TimeDistributed(Dense(dens2_size, activation='relu')))
    # model.add(Dropout(dr))
    model.add(Dense(obj_number, activation='sigmoid', activity_regularizer=regularizers.l2(regul)))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['binary_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(modelname , monitor='val_loss', mode='min', verbose=1,
                         save_best_only=True)
    h=model.fit(mydata.hdtrain.input, mydata.hdtrain.target, validation_data=(mydata.hdtest.input, mydata.hdtest.target),
              # validation_split=0.1,
              epochs=2000, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    # np.concatenate(mydata.dtest.input, axis=0)
    return model,h

def lstm_pred(mydata, modelname,  hidden_size, dens2_size, regul, dr):
    # for seed in seeds:
    _, seqlength, obj_number = np.shape(mydata.dtest.target)
    featurelen = np.shape(mydata.dtest.input[0])[1]

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(seqlength, featurelen)))
    model.add(LSTM(hidden_size, return_sequences=True, recurrent_dropout=dr))

    model.add(TimeDistributed(Dense(dens2_size, activation='relu')))
    model.add(Dropout(dr))
    model.add(Dense(obj_number, activation='softmax', activity_regularizer=regularizers.l2(regul)))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['categorical_accuracy'])
    # tool_freq = [y for x in mydata.train[counter] for y in x]

    # class_weights = class_weight.compute_class_weight('balanced',np.unique(mydata.alltools), mydata.alltools)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(modelname, monitor='val_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True)
    h=model.fit(mydata.dtrain.input, mydata.dtrain.target,
              validation_split=0.1,
              epochs=500, batch_size=10, verbose=2, callbacks=[es, mc])
    # make a prediction
    # np.concatenate(mydata.dtest.input, axis=0)
    return model,h