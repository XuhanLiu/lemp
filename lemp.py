#! /usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
import os
import sys
from collections import Counter
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib

AA = 'ARNDCQEGHILKMFPSTWYV'
CFG = joblib.load('model/config.pkl')

USAGE = """
USAGE
    python lemp.py  -i <input_file> [-o <output_file>] [-h] [-v]

OPTIONAL ARGUMENTS
    -i <input_file>  : dataset file containing protein sequences as FASTA file format.
    -o <output_file> : a directory containing the results of prediction of each sample.
    
    -v               : version information of this software.

    -h               : Help information, print USAGE and ARGUMENTS messages.

Note: Please designate each protein sequence in FASTA file with distinct name!
"""

VERSION = """
DESCRIPTION
   Name             : LEMP (LSTM-based Ensemble Malonylation Predictor)
   Version          : 1.0
   Update Time      : 2017-07-01
   Author           : Xuhan Liu & Zhen Chen
"""


def Snippet(fasta):
    fas = open(fasta).read().replace('\r\n', '\n')[1:].split('>')
    dic = {}
    for fa in fas:
        lines = fa.split('\n')
        kb = lines[0].split()[0]
        seq = ''.join(lines[1:]).upper()
        frags = []
        for i, res in enumerate(seq):
            if res != 'K': continue
            frag = [i+1]
            for j in range(i - 15, i + 16):
                if j < 0 or j >= len(seq):
                    frag.append(20)
                else:
                    frag.append(AA.index(seq[j]) if seq[j] in AA else 20)
            frags.append(frag)
            # print(str(i+1) + '\t' + tmpStr + '\n')
        dic[kb] = np.array(frags)
    return dic


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


def ScoreEAAC(dic):
    model = joblib.load('model/eaac.pkl')
    scores = {}
    for kb, frags in dic.items():
        score = np.zeros((len(frags), 2))
        score[:, 0] = frags[:, 0]
        score[:, 1] = model.predict_proba(EAAC(frags[:, 1:]))[:, 1]
        scores[kb] = score
    return scores


def ScoreLSTM(dic):
    scores = {}
    models = [load_model('model/lstm.%d.h5' % i) for i in range(5)]
    for kb, frags in dic.items():
        score = np.zeros((len(frags), 2))
        for model in models:
            score[:, 0] += frags[:, 0]
            score[:, 1] += model.predict_proba(frags[:, 1:])[:, 0]
        scores[kb] = score / 5
    return scores


def Predict(EAACscores, LSTMscores):
    scores = {}
    for kb in LSTMscores:
        EAACscore = EAACscores[kb]
        LSTMscore = LSTMscores[kb]
        score = np.zeros(LSTMscores[kb].shape)
        score[:, 0] = LSTMscore[:, 0]
        score[:, 1] = 1 / (1 + np.exp(-EAACscore[:, 1] * CFG['w_eaac'] - LSTMscore[:, 1] * CFG['w_lstm'] - CFG['bias']))
        scores[kb] = score
    return scores

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvi:o:")
        OPT = dict(opts)
    except getopt.GetoptError:
        print('ERROR: Invalid arguments usage. Please type \'-h\' for help.')
        sys.exit(2)
    if '-h' in OPT:
        print(USAGE + '\n')
    elif '-v' in OPT:
        print(VERSION + '\n')
    else:
        if '-i' not in OPT:
            print('ERROR: Input file is missing. Please type \'-h\' for help.')
            sys.exit(2)
        elif not os.path.exists(OPT['-i']):
            print('ERROR: Input file cannot be found. Please type \'-h\' for help.')
            sys.exit(2)
        # Process train and predict submit
        else:
            dic = Snippet(OPT['-i'])
            LSTMscores = ScoreLSTM(dic)
            EAACscores = ScoreEAAC(dic)
            scores = Predict(LSTMscores=LSTMscores, EAACscores=EAACscores)
            results = 'ID\tSite\tResidue\tScore\tY/N(Sp=90%)\tY/N(Sp=95%)\tY/N(Sp=99%)\n'
            for kb, score in scores.items():
                for i in score:
                    flag90 = 'Y' if i[1] > CFG['cut90'] else 'N'
                    flag95 = 'Y' if i[1] > CFG['cut95'] else 'N'
                    flag99 = 'Y' if i[1] > CFG['cut99'] else 'N'
                    results += '%s\t%d\tK\t%f\t%s\t%s\t%s\n' % (kb, i[0], i[1], flag90, flag95, flag99)
            if '-o' not in OPT:
                print(results)
            else:
                output = open(OPT['-o'], 'w')
                output.write(results)
                output.close()
    print('=== SUCCESS ===')
