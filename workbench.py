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
from pyAudioAnalysis import audioBasicIO
import base64

n_testing_slices = 5
categories_deviation = 0.001
explained_variance = .8

# data_frames = get_features_data_frames('time')

# print get_natural_frequency(data_frames[1][0][0]['Energy'])

# show_graphs('frequency', dimension='abs', actor_numbers=[2], phrase_numbers=[
#             1], emotion_numbers=[1], feature_labels=['Energy'])

# PCA Example
# x = []
# y_angry = []
# y_fearful = []
# y_happy = []
# y_sad = []
# while categories_deviation < .5:

data_frames = get_features_data_frames(
    'time', emotion_numbers=[3, 4, 5, 6])

unified_data_frame = unify_data_frames(data_frames, feature_reducers, {
    1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fearful'})

pca_data_frame, _pca_model, _scaler_model = apply_pca(
    unified_data_frame, n_components=explained_variance)

components = list(set(pca_data_frame) - set(['Emotion']))
for component in components:
    pca_data_frame[component] = categorize_data(
        categories_deviation, pca_data_frame[component])
data = pca_data_frame.loc[:, components].values
emotions = pca_data_frame.loc[:, ['Emotion']].values

# emotions_dict = ["Neutro", "Calma", "Felicidade",
#                  "Tristeza", "Raiva", "Medo", "Nojo", "Surpresa"]

positions_to_shuffle = generate_shuffle_positions(len(data))
emotions = shuffle_array(emotions, positions_to_shuffle)
data = shuffle_array(data, positions_to_shuffle)
# emotions = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# data = [["Teste 1"], ["Teste 2"], ["Teste 3"], ["Teste 4"], ["Teste 5"], ["Teste 6"], ["Teste 7"], ["Teste 8"], ["Teste 9"], ["Teste 10"]]

emotions_splitted = split_list(emotions, n_testing_slices)
data_splitted = split_list(data, n_testing_slices)

absolute_statistics_data_frames = []
relative_statistics_data_frames = []
for iteration in range(0, n_testing_slices):
    emotion_train, emotion_test = get_train_test_array(
        emotions_splitted, iteration)
    data_train, data_test = get_train_test_array(data_splitted, iteration)
    absolute_statistics, relative_statistics = test_model(
        (emotion_train, emotion_test), (data_train, data_test))

    absolute_statistics_data_frames.append(pd.DataFrame(absolute_statistics).apply(
        lambda p: p * 100).fillna(0))
    relative_statistics_data_frames.append(pd.DataFrame(relative_statistics).apply(
        lambda p: p * 100).fillna(0))

absolute_concat_dfs = pd.concat(absolute_statistics_data_frames)
relative_concat_dfs = pd.concat(relative_statistics_data_frames)
print('\n\n\nABSOLUTE\n\n')
absolute_mean_statistics = absolute_concat_dfs.groupby(
    absolute_concat_dfs.index).mean()
print pd.DataFrame(absolute_mean_statistics)
print('\n\n\nRELATIVE\n\n')
relative_mean_statistics = relative_concat_dfs.groupby(
    relative_concat_dfs.index).mean()
print pd.DataFrame(relative_mean_statistics)

#     a = pd.DataFrame.values
#     x.append(categories_deviation)
#     y_angry.append(mean_statistics['Angry']['Angry'])
#     y_fearful.append(mean_statistics['Fearful']['Fearful'])
#     y_happy.append(mean_statistics['Happy']['Happy'])
#     y_sad.append(mean_statistics['Sad']['Sad'])
#     categories_deviation += .01

# plt.plot(x, y_angry, 'r', label="Raiva")
# plt.plot(x, y_fearful, 'g', label="Medo")
# plt.plot(x, y_happy, 'y', label="Felicidade")
# plt.plot(x, y_sad, 'b', label="Trizteza")
# plt.legend(loc='best')
# plt.title('Desvio padrao')
# plt.show()
