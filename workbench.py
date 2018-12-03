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

n_testing_slices = 1
categories_deviation = .01
explained_variance = .99

# data_frames = get_features_data_frames('time')

# print get_natural_frequency(data_frames[1][0][0]['Energy'])

# show_graphs('frequency', dimension='abs', actor_numbers=[2], phrase_numbers=[
#             1], emotion_numbers=[1], feature_labels=['Energy'])

# PCA Example

data_frames = get_features_data_frames('time')
# , emotion_numbers=[3, 4, 5, 6])

unified_data_frame = unify_data_frames(data_frames, feature_reducers)
# {
# 1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fearful'})

pca_data_frame = apply_pca(unified_data_frame, n_components=explained_variance)

components = list(set(pca_data_frame) - set(['Emotion']))
for component in components:
    pca_data_frame[component] = categorize_data(
        categories_deviation, pca_data_frame[component])
data = pca_data_frame.loc[:, components].values
emotions = pca_data_frame.loc[:, ['Emotion']].values


# emotions_dict = ["Neutro", "Calma", "Felicidade",
#                  "Tristeza", "Raiva", "Medo", "Nojo", "Surpresa"]

emotions_dict = ["Felicidade",
                 "Tristeza", "Raiva", "Medo"]

positions_to_shuffle = generate_shuffle_positions(len(data))
emotions = shuffle_array(emotions, positions_to_shuffle)
data = shuffle_array(data, positions_to_shuffle)
# emotions = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# data = [["Teste 1"], ["Teste 2"], ["Teste 3"], ["Teste 4"], ["Teste 5"], ["Teste 6"], ["Teste 7"], ["Teste 8"], ["Teste 9"], ["Teste 10"]]

emotions_splitted = split_list(emotions, n_testing_slices)
data_splitted = split_list(data, n_testing_slices)

for iteration in range(0, n_testing_slices):
    emotion_train, emotion_test = get_train_test_array(
        emotions_splitted, iteration)
    data_train, data_test = get_train_test_array(data_splitted, iteration)
    statistics = test_model(
        (emotion_train, emotion_test), (data_train, data_test))

    print pd.DataFrame(statistics).apply(
        lambda p: p*100).fillna(0)
