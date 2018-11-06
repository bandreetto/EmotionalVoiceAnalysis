#!/usr/bin/env python
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt

feature_labels = [
    'Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5', 'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10', 'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']


def unwind_features(root, *args):
    if len(args) == 0:
        return root
    else:
        head, tail = args[0], args[1:]
        new_roots = map(root, head)
        return map(lambda new_root: unwind_features(new_root, *tail), new_roots)


def open_actor_features(actor_number):
    def open_phrase_features(phrase_number):
        def open_emotion_features(emotion_number):
            path = './data/Actor_{0:0>2}/Frase_{1}/Actor_{0:0>2}_0{2}_st.csv'.format(
                actor_number, phrase_number, emotion_number)
            return pd.read_csv(path, names=feature_labels)
        return open_emotion_features
    return open_phrase_features


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# print open_actor_features(6)(1)(1)

feature_data_frames = unwind_features(
    open_actor_features, range(1, 25), range(1, 3), range(1, 9))

# nervosinho1 = feature_data_frames[0][0][4]
# nervosinho2 = feature_data_frames[0][0][4]
# tristinho2 = feature_data_frames[0][0][3]

# # plt.plot(range(0, len(nervosinho1['Energy'])), nervosinho1['Energy'], 'b')
# plt.plot(range(0, len(nervosinho2['Energy'])), nervosinho2['Energy'], 'r')
# plt.plot(range(0, len(tristinho2['Energy'])), tristinho2['Energy'], 'b')
# plt.show()
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
color = 'r'
colors = ['r', 'b', 'g']
array=[]
i = 0
j =0 
k =0
l=0
for actor_features in feature_data_frames:
    color = colors[i % 3]
    i += 1
    for frase_features in actor_features:
        j+=1
        for emotion_features in frase_features:
            # if(color == 'g'):
            #     plt.suptitle('teste', fontsize=20)
            #     plt.plot(
            #     range(0, len(emotion_features['Energy'][18:70])), emotion_features['Energy'][18:70], color)
            #     plt.show()
            # plt.plot(
            #     range(0, len(emotion_features['Energy'])), emotion_features['Energy'], color)
            k+=1
            for label in feature_labels:
                l+=1
                array.append(emotion_features[label][18:70]) 
            l=0     

            path = './dataCropped/Actor_{0:0>2}/Frase_{1}/Emotion_0{2}_st.csv'.format(
                i, j, k) 
            createFolder('./dataCropped/Actor_{0:0>2}/Frase_{1}/'.format(
                i, j))
            array = numpy.transpose(array)
            numpy.savetxt(path, array, fmt='%f', delimiter=',')
            # numpy.savetxt(path, array, fmt='%f', delimiter=',', header="Zero Crossing Rate, Energy, Entropy of Energy, Spectral Centroid, Spectral Spread, Spectral Entropy, Spectral Flux, Spectral Rolloff, MFCC1, MFCC2, MFCC3, MFCC4, MFCC5, MFCC6, MFCC7, MFCC8, MFCC9, MFCC10, MFCC11, MFCC12, MFCC13, Chroma Vector 1, Chroma Vector 2, Chroma Vector 3, Chroma Vector 4, Chroma Vector 5, Chroma Vector 6, Chroma Vector 7, Chroma Vector 8, Chroma Vector 9, Chroma Vector 10, Chroma Vector 11, Chroma Vector 12, Chroma Deviation")
            array = []
        k =0
    j=0    
            # k+=1
# plt.show()
