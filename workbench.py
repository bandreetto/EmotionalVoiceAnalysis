#!/usr/bin/env python
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from utils import *

emotions, data = get_categories(0.2)
print emotions, data


classifier = naive_bayes_init()

naive_bayes_train(classifier, data, emotions)




# showGraphs('frequency',
#            #    feature_labels=['Energy'],
#            #    emotion_numbers=[3]
#            )


# print open_actor_features_maximums(1)(1)(1)['Energy']

# print get_covered_data_percentile(
# open_actor_features_maximums(1)(1)(1)['Energy'], 0)

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
