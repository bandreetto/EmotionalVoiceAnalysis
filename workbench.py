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
num = 0
iteration = 100


# X = numpy.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = numpy.array(["teste 1", "teste 2", "teste 3", "teste 4", "teste 5", "teste 6"])
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(X, Y)

# print(clf.predict([[-0.8, -1]]))


# data_set = load_iris()

# features, labels = split_features_labels(data_set)

# X_train, y_train, X_test, y_test = split_train_test(features, labels, 0.18)
# train_features, train_labels, test_features, test_labels = split_train_test(features, labels, 0.18)
classifier = naive_bayes_init()


# print(len(train_features), " ", len(test_features))
	
# clf = GaussianNB()
	
# print("Start training...")
# tStart = time()
# clf.fit(train_features, train_labels)
# print("Training time: ", round(time()-tStart, 3), "s")
	
# print("Accuracy: ", accuracy_score(clf.predict(test_features), test_labels))

# X_train, y_train, X_test, y_test = split_train_test(data, emotions, 0.18)


# print "X Train: "+str(X_train)

# print "y train: "+str(y_train)

# print "X Test: "+str(X_test)

# print "y test: "+str(y_test)
cont = 0
neutral = 0
calm = 0
happy = 0
sad = 0
angry = 0
fearful = 0
disgust = 0
surprised = 0
neutral_st = 0
calm_st = 0
happy_st = 0
sad_st = 0
angry_st = 0
fearful_st = 0
disgust_st = 0
surprised_st = 0
for i in range (1, iteration):
	emotions, data = get_categories(0.2)
	# print "Iteracao: "+str(i)
	emotion_test, data_test = split_array(emotions, data)
	X = numpy.array(data)
	Y = numpy.array(emotions)
	classifier.fit(X, Y)
	# naive_bayes_train(classifier, data, emotions)
	# print "Emocao esperada: "+ str(emotion_test[num])
	test = []
	test.append(data_test[num])
	# print test
	predict = classifier.predict(test)
	if(str(emotion_test[num]) == "Neutral"):
		neutral_st +=1
	if(str(emotion_test[num]) == "Calm"):
		calm_st +=1
	if(str(emotion_test[num]) == "Happy"):
		happy_st +=1
	if(str(emotion_test[num]) == "Sad"):
		sad_st +=1
	if(str(emotion_test[num]) == "Angry"):
		angry_st +=1
	if(str(emotion_test[num]) == "Fearful"):
		fearful_st +=1
	if(str(emotion_test[num]) == "Disgust"):
		disgust_st +=1
	if(str(emotion_test[num]) == "Surprised"):
		surprised_st +=1
	# predict = naive_bayes_predictions(classifier, data_test[num])
	if (str(predict[0]) == str(emotion_test[num])):
		cont += 1
		if(str(emotion_test[num]) == "Neutral"):
			neutral +=1
		if(str(emotion_test[num]) == "Calm"):
			calm +=1
		if(str(emotion_test[num]) == "Happy"):
			happy +=1
		if(str(emotion_test[num]) == "Sad"):
			sad +=1
		if(str(emotion_test[num]) == "Angry"):
			angry +=1
		if(str(emotion_test[num]) == "Fearful"):
			fearful +=1
		if(str(emotion_test[num]) == "Disgust"):
			disgust +=1
		if(str(emotion_test[num]) == "Surprised"):
			surprised +=1

	# print "Emocao do algoritmo: " + str(predict[0])

print "Acertos Total: "+ str(cont)
print "Acerto(%): "+ str((float(cont)/iteration)*100)
print "---------------------------------------------"
print "Acerto por emocao"
print "---------------------------------------------"
print "Acertos Neutral: "+ str(neutral)
print "Acerto Neutral(%): "+ str((float(neutral)/neutral_st)*100)
print "---------------------------------------------"
print "Acertos Calm: "+ str(calm)
print "Acerto Calm(%): "+ str((float(calm)/calm_st)*100)
print "---------------------------------------------"
print "Acertos Happy: "+ str(happy)
print "Acerto Happy(%): "+ str((float(happy)/happy_st)*100)
print "---------------------------------------------"
print "Acertos Sad: "+ str(sad)
print "Acerto Sad(%): "+ str((float(sad)/sad_st)*100)
print "---------------------------------------------"
print "Acertos Angry: "+ str(angry)
print "Acerto Angry(%): "+ str((float(angry)/angry_st)*100)
print "---------------------------------------------"
print "Acertos Fearful: "+ str(fearful)
print "Acerto Fearful(%): "+ str((float(fearful)/fearful_st)*100)
print "---------------------------------------------"
print "Acertos Disgust: "+ str(disgust)
print "Acerto Disgust(%): "+ str((float(disgust)/disgust_st)*100)
print "---------------------------------------------"
print "Acertos Surprised: "+ str(surprised)
print "Acerto Surprised(%): "+ str((float(surprised)/surprised_st)*100)
print "---------------------------------------------"
print "Checksum : "+ str(neutral_st+calm_st+happy_st+sad_st+angry_st+fearful_st+disgust_st+surprised_st)
print "---------------------------------------------"

	




# showGraphs('frequency',
#            #    feature_labels=['Energy'],
#            #    emotion_numbers=[3]
#            )


# print open_actor_features_maximums(1)(1)(1)['Energy']

# print get_covered_data_percentile(
# open_actor_features_maximums(1)(1)(1)['Energy'], 0)

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
