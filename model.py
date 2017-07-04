#! /usr/bin/env python
# -*- coding: utf-8 -*-
import getopt, sys, os
from collections import Counter
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.models import Sequential, save_model
from keras.optimizers import Adam
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve

AA = 'ARNDCQEGHILKMFPSTWYV'

HELP = """
USAGE
    python model.py [-t <training_set>] [-i <independent_set>] [-d] [-n <int>] [-e <int>]

OPTIONAL ARGUMENTS

    -t <training_set>    : the file of training set, which must have the same format as "dataset/chen_train.txt".
    -i <independent_set> : the file of independent set, which must have the same format as "dataset/chen_test.txt".
    -d                   : rebuilt the LSTM-based deep learning model simultaneously (very time-consuming).
    -n <int>             : the number of CPU to train the model.
    -e <int>             : the number of trees in random forest classifier.
    
    -v                   : Version information of this software.

    -h                   : Help information, print USAGE and ARGUMENTS messages.
"""

VERSION = """
DESCRIPTION
   Name             : LEMP_Model_Generation
   Version          : 1.0
   Update Time      : 2017-07-01
   Author           : Xuhan Liu & Zhen Chen
"""


class Config:
    def __init__(self, w_lstm, w_eaac, bias, cut90, cut95, cut99):
        self.w_lstm = w_lstm
        self.w_eaac = w_eaac
        self.bias = bias
        self.cut90 = cut90
        self.cut95 = cut95
        self.cut99 = cut99


def EAAC(frags):
    eaacs = []
    for frag in frags:
        eaac = []
        for i in range(24):
            count = Counter(frag[i: i + 8])
            if 20 in count: count.pop(20)
            sums = sum(count.values()) + 1e-6
            aac = [count[i] / sums for i in range(20)]
            eaac += aac
        eaacs.append(eaac)
    return np.array(eaacs)


def rf_train(X, y, indep=None, save=None, cpu=1, out=None, n_estimators=1000):
    print('INFO: Begin to train the EAAC-based random forest model...')
    cvs = np.zeros((X.shape[0], 2))
    if indep:
        inds = np.zeros((len(indep[1]), 2))
    print('INFO: The number of tree is %d in random forest.' % n_estimators)
    print('INFO: This model will be trained on %d CPUs.' % cpu)
    model = ensemble.RandomForestClassifier(n_estimators=n_estimators, n_jobs=cpu)
    model.fit(X, y)
    scores = model.predict_proba(X)
    cvs[:, 0], cvs[:, 1] = y, scores[:, 1]
    if save:
        joblib.dump(model, save + '.pkl', compress=9)
    if indep:
        inds[:, 0], inds[:, 1] = indep[1], model.predict_proba(indep[0])[:, 1]
    if indep:
        np.savetxt(out + '.ind.txt', inds, fmt='%f', delimiter='\t')
    np.savetxt(out + '.cv.txt', cvs, delimiter='\t', fmt='%f')


def RNN():
    model = Sequential()
    model.add(Embedding(21, 5, input_length=31))

    model.add(LSTM(32, implementation=2))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def rnn_train(X, y, out, indep=None, batch_size=512, epochs=300):
    print('INFO: Begin to train the LSTM-based deep learning model...')
    cvs = np.zeros((len(y), 2))
    folds = StratifiedKFold(5).split(X, y)
    if indep:
        inds = np.zeros((len(indep[1]), 2))
    for i, (trained, valided) in enumerate(folds):
        X_train, y_train = X[trained], y[trained]
        X_valid, y_valid = X[valided], y[valided]
        net = RNN()
        best_saving = ModelCheckpoint(filepath='%s.%d.h5' % (out, i), monitor='val_loss',
                                      verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        net.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), verbose=2,
                callbacks=[best_saving, early_stopping], batch_size=batch_size)
        save_model(net, '%s.%d.h5' % (out, i))
        print("Validation test:", net.evaluate(X_valid, y_valid, batch_size=batch_size))
        cvs[valided, 0], cvs[valided, 1] = y_valid, net.predict_proba(X_valid, batch_size=batch_size)[:, 0]
        if indep:
            print("Independent test:", net.evaluate(indep[0], indep[1], batch_size=batch_size))
            inds[:, 0] += indep[1]
            inds[:, 1] += net.predict_proba(indep[0], batch_size=batch_size)[:, 0]
    if indep:
        np.savetxt(out + '.ind.txt', inds / 5, fmt='%f', delimiter='\t')
    np.savetxt(out + '.cv.txt', cvs, fmt='%f', delimiter='\t')


def set_config(path_eaac, path_lstm):
    score_lstm = np.loadtxt(path_lstm)
    score_eaac = np.loadtxt(path_eaac)
    X, y = np.concatenate([score_eaac[:, 1:], score_lstm[:, 1:]], axis=1), score_eaac[:, 0]
    model = LogisticRegression()
    model.fit(X, y)
    fprs, tprs, thrs = roc_curve(y, model.predict_proba(X)[:, 1])
    cuts = [0, 0, 0]
    for i, fpr in enumerate(fprs):
        for j, value in enumerate([0.1, 0.05, 0.01]):
            if np.abs(fpr - value) < np.abs(fprs[cuts[j]] - value):
                cuts[j] = i
    config = {'w_eaac': model.coef_[:, 0],
              'w_lstm': model.coef_[:, 1],
              'bias': model.intercept_,
              'cut90': thrs[cuts[0]],
              'cut95': thrs[cuts[1]],
              'cut99': thrs[cuts[2]]}
    joblib.dump(config, 'model/config.pkl')


def pep1(path):
    seqs = open(path).readlines()
    X = [[AA.index(res.upper()) if res.upper() in AA else 20 for res in (seq.split()[0])]
         for seq in seqs if seq.strip() != '']
    y = np.array([int(seq.split()[-1]) for seq in seqs if seq.strip() != ''])
    return EAAC(X), y


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvdi:o:n:e:")
        OPT = dict(opts)
    except getopt.GetoptError:
        print('ERROR: Invalid arguments usage.')
        sys.exit(2)
    if '-h' in OPT:
        print(HELP + '\n')
    elif '-v' in OPT:
        print(VERSION + '\n')
    else:
        if '-t' not in OPT:
            print('INFO: Training set is not given, \'dataset/chen_train.txt\' will be used as default')
            OPT['-t'] = 'dataset/chen_train.txt'
        if not os.path.exists(OPT['-t']):
            print('ERROR: Training set file cannot be found.')
            sys.exit(2)
        if '-i' not in OPT:
            print('INFO: Independent set is not given, \'dataset/chen_test.txt\' will be used as default')
            OPT['-i'] = 'dataset/chen_test.txt'
        if not os.path.exists(OPT['-i']):
            print('ERROR: Independent file cannot be found.')
            sys.exit(2)
        if '-n' not in OPT:
            OPT['-n'] = 1
        if '-e' not in OPT:
            OPT['-e'] = 1000
        X, y = pep1(OPT['-t'])
        indep = pep1(OPT['-i'])
        rf_train(X, y, indep=indep, out='dataset/eaac', save='model/eaac', cpu=int(OPT['-n']), n_estimators=int(OPT['-e']))
        if '-d' in OPT:
            rnn_train(X, y, indep=indep, out='dataset/lstm')
        set_config('dataset/eaac.ind.txt', 'dataset/lstm.ind.txt')

        print('=== SUCCESS ===')

