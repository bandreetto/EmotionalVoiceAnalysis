#!/usr/bin/env python
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from utils import *


# plt.plot(range(0, len(open_actor_features(5)(1)(1)[
#          'Energy'])), open_actor_features(5)(1)(1)['Energy'])

showGraphs('frequency')

# feature_data_frames = unwind_features(
#     curry(open_actor_features_FFT, dimension='angle'), range(1, 25), range(1, 3), range(1, 9))

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# color = 'r'
# colors = ['r', 'b', 'g']
# i = 0
# for actor_features in feature_data_frames:
#     i += 1
#     color = colors[i % 3]
#     for frase_features in actor_features:
#         for emotion_features in frase_features:
#             plt.plot(
#                 range(0, len(emotion_features['Energy'])), emotion_features['Energy'], color)
