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

# data_set = load_iris()

# features, labels = split_features_labels(data_set)

# X_train, y_train, X_test, y_test = split_train_test(features, labels, 0.18)
# train_features, train_labels, test_features, test_labels = split_train_test(features, labels, 0.18)
# classifier = naive_bayes_init()


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

# for i in range (1, 50):
# 	emotions, data = get_categories(0.2)
# 	print "Iteracao: "+str(i)
# 	emotion_test, data_test = split_array(emotions, data)
# 	naive_bayes_train(classifier, data, emotions)
# 	print "Emocao esperada: "+ str(emotion_test[num])
# 	predict = naive_bayes_predictions(classifier, data_test[num])

# 	print "Emocao do algoritmo: " + str(predict)

# 	print("Number of mislabeled points out of a total {} "
#       .format(
#           emotion_test.shape[0])
# 	)




# showGraphs('frequency',
#            #    feature_labels=['Energy'],
#            #    emotion_numbers=[3]
#            )


# print open_actor_features_maximums(1)(1)(1)['Energy']

# print get_covered_data_percentile(
# open_actor_features_maximums(1)(1)(1)['Energy'], 0)

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
