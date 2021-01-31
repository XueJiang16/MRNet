

import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from data_preprocess import *
import mit_utils as utils
import time
import matplotlib.pyplot as plt
import tensorflow_addons as tfa


target_class = ['W', 'N1', 'N2', 'N3', 'REM']
target_sig_length = 3072
tic = time.time()
trainX, trainY, TestX, TestY = dataload('channel0.npz')
toc = time.time()

markov_matrix = [[60846, 3612,  183,    9,   83,],
 [ 2042, 16099, 3837,    11, 661],
 [ 1166, 2030,72165, 3136,  970],
 [  163,  102, 2875, 14339,   21],
 [  516,  807,  407,     5, 28945],]

markov_matrix = np.array(markov_matrix)

markov_matrix = np.log2(markov_matrix) ** 3
for i in range(5):
    max = np.max(markov_matrix[i])
    markov_matrix[i] /= max

print('Time for data processing--- '+str(toc-tic)+' seconds---')
model_name = 'myNet.h5'
model = load_model(model_name)
# model.summary()
pred_vt = model.predict(TestX, batch_size=256, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(TestY, axis=1)

def weight_decay(order):
    weights = []
    for i in range(order):
        weights.append(4 ** (-i))
    return weights

order = 6
weight = weight_decay(order)

for i in range(1,len(pred_vt)-order):
    factor = 1
    if pred_v[i-1] != pred_v[i]:
        for j in range(1,order+1):
            if pred_v[i+j] == pred_v[i-1]:
                factor += weight[j-1]*2.1
            elif pred_v[i+j] == pred_v[i]:
                factor -= 0.55 * weight[j-1]
                if factor < 0.1:
                    factor = 0.1
        vector = markov_matrix[pred_v[i - 1]].copy()
        vector[pred_v[i-1]] *= factor
        re_pred = pred_vt[i] * vector
        pred_v[i] = np.argmax(re_pred)

utils.plot_confusion_matrix(true_v, pred_v, np.array(target_class))
utils.print_results(true_v, pred_v, target_class)
plt.savefig('cm.png')
