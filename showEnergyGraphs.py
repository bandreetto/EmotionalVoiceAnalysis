#!/usr/bin/env python
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from utils import *

feature_labels = [
    'Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5', 'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10', 'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

# print open_actor_features(6)(1)(1)


feature_data_frames = unwind_features(
    curry(open_actor_features_FFT, dimension='angle'), range(1, 25), range(1, 3), range(1, 9))

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
color = 'r'
colors = ['r', 'b', 'g']
i = 0
for actor_features in feature_data_frames:
    i += 1
    color = colors[i % 3]
    for frase_features in actor_features:
        for emotion_features in frase_features:
            plt.plot(
                range(0, len(emotion_features['Energy'])), emotion_features['Energy'], color)
plt.show()
