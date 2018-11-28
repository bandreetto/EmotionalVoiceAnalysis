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
		print "---------------------------------------------"
		print "  Iteracao: "+str(i+1) + " Teste: "+str(j+1)
		print "---------------------------------------------"
		test = []
		test.append(data_test[j])
		predict = classifier.predict(test)
		statistic_data(emotion_test[j], predict, statistic_matrix)
		# statistic_matrix = numpy.array(statistic_matrix, dtype=numpy.float32)
		# statistic_matrix = statistic_matrix/len(emotion_test)+1

		allResults = [statistic_matrix[0],statistic_matrix[1],statistic_matrix[2],statistic_matrix[3],statistic_matrix[4],statistic_matrix[5],statistic_matrix[6],statistic_matrix[7]]
		allResults = list(map(sum, zip(*allResults)))
		for k in range(8):
			print "---------------------------------------------"
			print str(statistic_matrix[k]) + " Soma: " +str(sum(statistic_matrix[k]))
			totalTriesForEmotion = sum(statistic_matrix[k])
			hitsAbsNumber = statistic_matrix[k][k]
			if (totalTriesForEmotion > 0):
				hitsPercentage = float(hitsAbsNumber) / sum(statistic_matrix[k]) * 100
				falseNegativesPercentage = 100 - hitsPercentage
				falsePositivesAbsNumber = float(allResults[k] - hitsAbsNumber)
				falsePositivesPercentage =  falsePositivesAbsNumber / sum(allResults) * 100
				print ('Acertos: ' + str(hitsPercentage) + '%')
				print ('Falsos positivos (tipo I): ' + str(falsePositivesPercentage) + '%')
				print ('Falsos negativos (tipo II): ' + str(falseNegativesPercentage) + '%')
				falsePositiveIterationSum[k] = falsePositiveIterationSum[k] + falsePositivesPercentage
				falseNegativeIterationSum[k] = falseNegativeIterationSum[k] + falseNegativesPercentage
				resultsCounter[k] = resultsCounter[k] + 1
		print "---------------------------------------------"
		print "Checksum: "+str(sum(allResults))

for l in range(8):
	if (resultsCounter[l] > 0):
		print ('Emocao: ' + str(l))
		print ('Falsos positivos (tipo I): ' + str(falsePositiveIterationSum[l]/resultsCounter[l]) + '%')
		print ('Falsos negativos (tipo II): ' + str(falseNegativeIterationSum[l]/resultsCounter[l]) + '%')



# for j in range (1, rodada):
# 	print "---------------------------------------------"
# 	print "Rodadas de 1000 em 1000"
# 	print "Rodadas atual: "+str(j)
# 	print "---------------------------------------------"
# 	cont = 0
# 	neutral = 0
# 	calm = 0
# 	happy = 0
# 	sad = 0
# 	angry = 0
# 	fearful = 0
# 	disgust = 0
# 	surprised = 0
# 	neutral_st = 0
# 	calm_st = 0
# 	happy_st = 0
# 	sad_st = 0
# 	angry_st = 0
# 	fearful_st = 0
# 	disgust_st = 0
# 	surprised_st = 0

# 	for i in range (1, iteration):
# 		emotions, data = get_categories(0.1)
# 		# print "Iteracao: "+str(i)
# 		emotion_splitted = split_list(emotions, 5)
# 		data_splitted = split_list(data, 5)

# 		emotion_train, emotion_test get_train_test_array(emotions_splitted,i)
# 		data_train, data_test get_train_test_array(data_splitted,i)
# 		X = numpy.array(data_train)
# 		Y = numpy.array(emotions_train)
# 		classifier.fit(X, Y)
# 		# naive_bayes_train(classifier, data, emotions)
# 		# print "Emocao esperada: "+ str(emotion_test[num])
# 		test = []
# 		test.append(data_test[num])
# 		# print test
# 		predict = classifier.predict(test)
# 		# if(str(emotion_test[num]) == "Neutral"):
# 		# 	neutral_st +=1
# 		# if(str(emotion_test[num]) == "Calm"):
# 		# 	calm_st +=1
# 		if(str(emotion_test[num]) == "Happy"):
# 			happy_st +=1
# 		if(str(emotion_test[num]) == "Sad"):
# 			sad_st +=1
# 		if(str(emotion_test[num]) == "Angry"):
# 			angry_st +=1
# 		if(str(emotion_test[num]) == "Fearful"):
# 			fearful_st +=1
# 		# if(str(emotion_test[num]) == "Disgust"):
# 		# 	disgust_st +=1
# 		# if(str(emotion_test[num]) == "Surprised"):
# 		# 	surprised_st +=1
# 		# predict = naive_bayes_predictions(classifier, data_test[num])
# 		if (str(predict[0]) == str(emotion_test[num])):
# 			cont += 1
# 			# if(str(emotion_test[num]) == "Neutral"):
# 			# 	neutral +=1
# 			# if(str(emotion_test[num]) == "Calm"):
# 			# 	calm +=1
# 			if(str(emotion_test[num]) == "Happy"):
# 				happy +=1
# 			if(str(emotion_test[num]) == "Sad"):
# 				sad +=1
# 			if(str(emotion_test[num]) == "Angry"):
# 				angry +=1
# 			if(str(emotion_test[num]) == "Fearful"):
# 				fearful +=1
# 			# if(str(emotion_test[num]) == "Disgust"):
# 			# 	disgust +=1
# 			# if(str(emotion_test[num]) == "Surprised"):
# 			# 	surprised +=1
# 		# else:
# 		# 	print "Era para ser: "+str(emotion_test[num])+ " ,mas o algoritmo classificou: "+str(predict[0])
# 		# print "Emocao do algoritmo: " + str(predict[0])

# 	print "Acertos Total: "+ str(cont)
# 	print "Acerto(%): "+ str((float(cont)/iteration)*100)
# 	print "---------------------------------------------"
# 	print "Acerto por emocao"
# 	# print "---------------------------------------------"
# 	# print "Acertos Neutral: "+ str(neutral)
# 	# print "Acerto Neutral(%): "+ str((float(neutral)/neutral_st)*100)
# 	# print "---------------------------------------------"
# 	# print "Acertos Calm: "+ str(calm)
# 	# print "Acerto Calm(%): "+ str((float(calm)/calm_st)*100)
# 	print "---------------------------------------------"
# 	print "Acertos Happy: "+ str(happy)
# 	print "Acerto Happy(%): "+ str((float(happy)/happy_st)*100)
# 	print "---------------------------------------------"
# 	print "Acertos Sad: "+ str(sad)
# 	print "Acerto Sad(%): "+ str((float(sad)/sad_st)*100)
# 	print "---------------------------------------------"
# 	print "Acertos Angry: "+ str(angry)
# 	print "Acerto Angry(%): "+ str((float(angry)/angry_st)*100)
# 	print "---------------------------------------------"
# 	print "Acertos Fearful: "+ str(fearful)
# 	print "Acerto Fearful(%): "+ str((float(fearful)/fearful_st)*100)
# 	# print "---------------------------------------------"
# 	# print "Acertos Disgust: "+ str(disgust)
# 	# print "Acerto Disgust(%): "+ str((float(disgust)/disgust_st)*100)
# 	# print "---------------------------------------------"
# 	# print "Acertos Surprised: "+ str(surprised)
# 	# print "Acerto Surprised(%): "+ str((float(surprised)/surprised_st)*100)
# 	print "---------------------------------------------"
# 	print "Checksum : "+ str(neutral_st+calm_st+happy_st+sad_st+angry_st+fearful_st+disgust_st+surprised_st)
# 	print "---------------------------------------------"






# showGraphs('frequency',
#            #    feature_labels=['Energy'],
#            #    emotion_numbers=[3]
#            )


# print open_actor_features_maximums(1)(1)(1)['Energy']

# print get_covered_data_percentile(
# open_actor_features_maximums(1)(1)(1)['Energy'], 0)

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
