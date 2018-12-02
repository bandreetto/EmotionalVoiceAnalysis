#!/usr/bin/env python
import numpy
import os
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from utils import *
from tabulate import tabulate

# data_frames = get_features_data_frames('time')

# print get_natural_frequency(data_frames[1][0][0]['Energy'])

# show_graphs('frequency', dimension='abs', actor_numbers=[2], phrase_numbers=[
#             1], emotion_numbers=[1], feature_labels=['Energy'])

# PCA Example

data_frames = get_features_data_frames('time')

unified_data_frame = unify_data_frames(data_frames, feature_reducers)

pca_data_frame = apply_pca(unified_data_frame, n_components=.9)

a = pca_data_frame.values
print len(a[0]) - 1

emotions = []
data = []
for row in range(0, 384):
    if(a[row][len(a[0]) - 1] == 'Happy' or a[row][len(a[0]) - 1] == 'Angry' or a[row][len(a[0]) - 1] == 'Sad' or a[row][len(a[0]) - 1] == 'Fearful'):
        emotions.append([a[row][len(a[0]) - 1]])
        b = a[row][0:len(a[0]) - 1]
        data.append(b)
data = numpy.array(data)


# # accumulated_ratio = 0
# # for index, ratio in enumerate(pca.explained_variance_ratio_):
# #     accumulated_ratio += ratio
# #     print 'Component n {:0>2} Eplained ratio: {:.3f}%         Accumulated ratio: {:.3f}%'.format(
# #         index, ratio*100, accumulated_ratio*100)

# emotions_dict = ["Neutro", "Calma", "Felicidade",
#                  "Tristeza", "Raiva", "Medo", "Nojo", "Surpresa"]

emotions_dict = ["Felicidade",
                 "Tristeza", "Raiva", "Medo"]
iteracoes = 5

a = generate_shuffle_positions(len(data))
# emotions, data = get_categories(1)
emotions = shuffle_array(emotions, a)
data = shuffle_array(data, a)
# emotions = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# data = [["Teste 1"], ["Teste 2"], ["Teste 3"], ["Teste 4"], ["Teste 5"], ["Teste 6"], ["Teste 7"], ["Teste 8"], ["Teste 9"], ["Teste 10"]]
for i in range(0, iteracoes):
        # fnCounter = {
 #        0: 0,
 #        1: 0,
 #        2: 0,
 #        3: 0,
 #        4: 0,
 #        5: 0,
 #        6: 0,
 #        7: 0,
 #    }

 #    fpCounter = {
 #        0: 0,
 #        1: 0,
 #        2: 0,
 #        3: 0,
 #        4: 0,
 #        5: 0,
 #        6: 0,
 #        7: 0,
 #    }

 #    falsePositiveIterationSum = {
 #        0: 0,
 #        1: 0,
 #        2: 0,
 #        3: 0,
 #        4: 0,
 #        5: 0,
 #        6: 0,
 #        7: 0,
 #    }

 #    falseNegativeIterationSum = {
 #        0: 0,
 #        1: 0,
 #        2: 0,
 #        3: 0,
 #        4: 0,
 #        5: 0,
 #        6: 0,
 #        7: 0,
 #    }

 #    hitsIterationSum = {
 #        0: 0,
 #        1: 0,
 #        2: 0,
 #        3: 0,
 #        4: 0,
 #        5: 0,
 #        6: 0,
 #        7: 0,
 #    }
    fnCounter = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    fpCounter = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    falsePositiveIterationSum = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    falseNegativeIterationSum = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    hitsIterationSum = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    statistic_matrix = statistic_matrix_init()
    classifier = naive_bayes_init()
    emotions_splitted = split_list(emotions, 5)
    data_splitted = split_list(data, 5)
    emotion_train, emotion_test = get_train_test_array(emotions_splitted, i)
    data_train, data_test = get_train_test_array(data_splitted, i)
    X = numpy.array(data_train)
    Y = numpy.array(emotion_train)
    classifier.fit(X, Y)
    for j in range(0, len(emotion_test)):
        test = []
        test.append(data_test[j])
        predict = classifier.predict(test)
        statistic_data(emotion_test[j], predict, statistic_matrix)
        # statistic_matrix = numpy.array(statistic_matrix, dtype=numpy.float32)
        # statistic_matrix = statistic_matrix/len(emotion_test)+1

        allResults = [statistic_matrix[0], statistic_matrix[
            1], statistic_matrix[2], statistic_matrix[3]]
        # allResults = [statistic_matrix[0], statistic_matrix[1], statistic_matrix[2], statistic_matrix[3],
        # statistic_matrix[4], statistic_matrix[5], statistic_matrix[6],
        # statistic_matrix[7]]
        allResults = list(map(sum, zip(*allResults)))
        for k in range(4):
            totalTriesForEmotion = sum(statistic_matrix[k])
            hitsAbsNumber = statistic_matrix[k][k]
            if (totalTriesForEmotion > 0):
                hitsPercentage = float(hitsAbsNumber) / \
                    sum(statistic_matrix[k]) * 100
                falseNegativesPercentage = 100 - hitsPercentage
                falseNegativeIterationSum[k] = falseNegativeIterationSum[k] + \
                    falseNegativesPercentage
                hitsIterationSum[k] = hitsIterationSum[k] + hitsPercentage
                fnCounter[k] = fnCounter[k] + 1

                if (sum(allResults) - totalTriesForEmotion > 0):
                    falsePositivesAbsNumber = float(
                        allResults[k] - hitsAbsNumber)
                    falsePositivesPercentage = falsePositivesAbsNumber / \
                        (sum(allResults) - totalTriesForEmotion) * 100
                    falsePositiveIterationSum[k] = falsePositiveIterationSum[k] + \
                        falsePositivesPercentage
                    fpCounter[k] = fpCounter[k] + 1


table = []
for l in range(4):
    if (fnCounter[l] > 0):
        emotion = emotions_dict[l]
        assertivity = '{:.2f}%'.format(hitsIterationSum[l] / fnCounter[l])
        falsePositive = '{:.2f}%'.format(
            falsePositiveIterationSum[l] / fpCounter[l])
        falseNegative = '{: .2f} % '.format(
            falseNegativeIterationSum[l] / fnCounter[l])
        table.append([emotion, assertivity, falsePositive, falseNegative])

print tabulate(table, headers=['Emocao', 'Acuracia',
                               'FN (tipo 1)', 'FP (tipo 2)'], tablefmt='orgtbl')
