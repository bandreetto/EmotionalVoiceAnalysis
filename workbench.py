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

# PCA Example

# data_frames = get_features_data_frames('time')

# unified_data_frame = unify_data_frames(data_frames, feature_reducers)

# pca = apply_pca(unified_data_frame)

# for ratio in pca.explained_variance_ratio_:
#     print '{:.3f}%'.format(ratio*100)

emotions_dict = ["Neutro", "Calma", "Felicidade", "Tristeza", "Raiva", "Medo", "Nojo", "Surpresa"]

iteracoes = 5

a = generate_shuffle_positions(384)
emotions, data = get_categories(1)
emotions = shuffle_array(emotions, a)
data = shuffle_array(data, a)
print len(emotions)
# emotions = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# data = [["Teste 1"], ["Teste 2"], ["Teste 3"], ["Teste 4"], ["Teste 5"], ["Teste 6"], ["Teste 7"], ["Teste 8"], ["Teste 9"], ["Teste 10"]]
for i in range (0, iteracoes):
	resultsCounter = {
		0: 0,
		1: 0,
		2: 0,
		3: 0,
		4: 0,
		5: 0,
		6: 0,
		7: 0,
	}

	falsePositiveIterationSum = {
		0: 0,
		1: 0,
		2: 0,
		3: 0,
		4: 0,
		5: 0,
		6: 0,
		7: 0,
	}

	falseNegativeIterationSum = {
		0: 0,
		1: 0,
		2: 0,
		3: 0,
		4: 0,
		5: 0,
		6: 0,
		7: 0,
	}

	hitsIterationSum = {
		0: 0,
		1: 0,
		2: 0,
		3: 0,
		4: 0,
		5: 0,
		6: 0,
		7: 0,
	}

	statistic_matrix = statistic_matrix_init();
	classifier = naive_bayes_init()
	emotions_splitted = split_list(emotions, 5)
	data_splitted = split_list(data, 5)
	emotion_train, emotion_test = get_train_test_array(emotions_splitted,i)
	data_train, data_test = get_train_test_array(data_splitted,i)
	X = numpy.array(data_train)
	Y = numpy.array(emotion_train)
	classifier.fit(X, Y)
	for j in range (0, len(emotion_test)):
		test = []
		test.append(data_test[j])
		predict = classifier.predict(test)
		statistic_data(emotion_test[j], predict, statistic_matrix)
		# statistic_matrix = numpy.array(statistic_matrix, dtype=numpy.float32)
		# statistic_matrix = statistic_matrix/len(emotion_test)+1

		allResults = [statistic_matrix[0],statistic_matrix[1],statistic_matrix[2],statistic_matrix[3],statistic_matrix[4],statistic_matrix[5],statistic_matrix[6],statistic_matrix[7]]
		allResults = list(map(sum, zip(*allResults)))
		for k in range(8):
			totalTriesForEmotion = sum(statistic_matrix[k])
			hitsAbsNumber = statistic_matrix[k][k]
			if (totalTriesForEmotion > 0):
				hitsPercentage = float(hitsAbsNumber) / sum(statistic_matrix[k]) * 100
				falseNegativesPercentage = 100 - hitsPercentage
				falsePositivesAbsNumber = float(allResults[k] - hitsAbsNumber)
				falsePositivesPercentage =  falsePositivesAbsNumber / sum(allResults) * 100

				falsePositiveIterationSum[k] = falsePositiveIterationSum[k] + falsePositivesPercentage
				falseNegativeIterationSum[k] = falseNegativeIterationSum[k] + falseNegativesPercentage
				hitsIterationSum[k] = hitsIterationSum[k] + hitsPercentage
				resultsCounter[k] = resultsCounter[k] + 1

for l in range(8):
	if (resultsCounter[l] > 0):
		print ('------------------------------------------------------------------------------------')
		print ('Emocao: ' + emotions_dict[l])
		print ('Acertividae: ' + str(hitsIterationSum[l]/resultsCounter[l]) + '%' )
		print ('Falsos positivos (tipo I): ' + str(falsePositiveIterationSum[l]/resultsCounter[l]) + '%')
		print ('Falsos negativos (tipo II): ' + str(falseNegativeIterationSum[l]/resultsCounter[l]) + '%')
