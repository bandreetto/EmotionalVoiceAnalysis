from numpy import genfromtxt
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.signal import argrelextrema
from sklearn.naive_bayes import GaussianNB
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import heapq

feature_labels = [
    'Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5', 'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10', 'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']

feature_labels_FFT = [
    'domain', 'Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5', 'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10', 'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

# dimension is 'abs', or 'angle'


def extractFFT(actorFeatures, dimension):

    fftFeaturesValues = []
    fftFramesFeatures = []

    frequencies = fftfreq(len(actorFeatures[0]), 0.05)
    fftFeaturesValues.append(frequencies)
    for frameFeatures in actorFeatures:
        fftFramesFeatures.append(fft(frameFeatures))

    for fftFrameFeaturesValues in fftFramesFeatures:
        frameFeatureValues = []
        for fftValue in fftFrameFeaturesValues:
            if dimension == 'angle':
                frameFeatureValues.append(numpy.angle(fftValue))
            else:
                frameFeatureValues.append(numpy.abs(fftValue))
        fftFeaturesValues.append(frameFeatureValues)

    fftFeaturesValuesNumpyArray = numpy.asarray(
        numpy.transpose(fftFeaturesValues))

    return fftFeaturesValuesNumpyArray


def naive_bayes_init():
    # Cria o classifier
    return GaussianNB()


def naive_bayes_train(clf, feature_data_array, feature_labels_array):
        # Treina o classifier
    clf.fit(feature_data_array, feature_labels_array)


def naive_bayes_predictions(clf, featute_data_test_array):
    # Testa o classifier
    predicted = clf.predict(featute_data_test_array)


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
            path = './data/Actor_{0:0>2}/Frase_{1}/Emotion_0{2}_st.csv'.format(
                actor_number, phrase_number, emotion_number)
            return pd.read_csv(path, names=feature_labels)
        return open_emotion_features
    return open_phrase_features


# dimension is either 'abs' or 'angle'
def open_actor_features_FFT(actor_number, **kwargs):
    def open_phrase_features(phrase_number):
        def open_emotion_features(emotion_number):
            dimension = kwargs.get('dimension')
            if dimension == 'angle':
                path = './FFT/Actor_{0:0>2}/Frase_{1}/FFT_0{2}_ANGLE.csv'.format(
                    actor_number, phrase_number, emotion_number)
            else:
                path = './FFT/Actor_{0:0>2}/Frase_{1}/FFT_0{2}_ABS.csv'.format(
                    actor_number, phrase_number, emotion_number)
            return pd.read_csv(path, names=feature_labels_FFT)
        return open_emotion_features
    return open_phrase_features


def open_actor_features_maximums(actor_number):
    def open_phrase_features(phrase_number):
        def open_emotion_features(emotion_number):
            path = './FFT/Actor_{0:0>2}/Frase_{1}/FFT_{2:0>2}_MAXIMUMS.csv'.format(
                actor_number, phrase_number, emotion_number)
            return pd.read_csv(path, names=feature_labels)
        return open_emotion_features
    return open_phrase_features

# Naive Bayes


def getFFTMaximums(fftData, output):
    transposedData = numpy.transpose(fftData)
    frequencies = fftfreq(len(transposedData[0]), 0.05)
    i = 0
    maximumsFrequencies = []
    for label in feature_labels_FFT:
        if (i != 0):
            localMaximunsIndexes = argrelextrema(
                transposedData[i], numpy.greater)
            localMaximunsValues = []
            for index in localMaximunsIndexes:
                localMaximunsValues.append(transposedData[i][index])

            localMaximunsValues = localMaximunsValues[0]
            maxLocalMaximumIndexAtMaxArray = numpy.argmax(
                localMaximunsValues)
            localMaximunsIndexes = localMaximunsIndexes[0]
            maxLocalMaximumIndexAtFeatureColumn = localMaximunsIndexes[
                maxLocalMaximumIndexAtMaxArray]
            maximumsFrequencies.append(
                transposedData[0][maxLocalMaximumIndexAtFeatureColumn])
        i += 1
    maxFrequenciesColumnsArray = [maximumsFrequencies]
    numpy.savetxt(output, maxFrequenciesColumnsArray, delimiter=",")


def extractAndSaveAllDataFFT():
    feature_data_frames = unwind_features(
        open_actor_features, range(1, 25), range(1, 3), range(1, 9))

    actorCounter = 1
    for actor_features in feature_data_frames:
        # actor_features = todas as frases de um ator
        phraseCounter = 1
        for frase_features in actor_features:
            # frase_features = todas as emocoes de uma frase de um ator
            emotionCounter = 1
            for emotion_features in frase_features:
                # emotion_features = todas as features de uma emocao de uma frase de um ator

                # transposedArray tem 34 arrays contendo os valores de cada feature por posicao
                transposedArray = numpy.transpose(emotion_features.values)

                # fft eh o fft de todas as features de uma emocao de uma frase por ator
                fft_emotion_features_abs = extractFFT(transposedArray, 'abs')
                fft_emotion_features_angle = extractFFT(
                    transposedArray, 'angle')

                fft_emotion_features_abs_half = fft_emotion_features_abs[:len(
                    fft_emotion_features_abs)/2 - 1]
                fft_emotion_features_angle_half = fft_emotion_features_angle[:len(
                    fft_emotion_features_angle)/2 - 1]
                if not os.path.exists('FFT/Actor_{0:0>2}/Frase_{1}'.format(actorCounter, phraseCounter)):
                    os.makedirs(
                        'FFT/Actor_{0:0>2}/Frase_{1}'.format(actorCounter, phraseCounter))

                getFFTMaximums(fft_emotion_features_abs_half, 'FFT/Actor_{0:0>2}/Frase_{1}/FFT_{2:0>2}_MAXIMUMS.csv'.format(
                    actorCounter, phraseCounter, emotionCounter))

                numpy.savetxt('FFT/Actor_{0:0>2}/Frase_{1}/FFT_{2:0>2}_ABS.csv'.format(
                    actorCounter, phraseCounter, emotionCounter), fft_emotion_features_abs_half, delimiter=",")
                numpy.savetxt('FFT/Actor_{0:0>2}/Frase_{1}/FFT_{2:0>2}_ANGLE.csv'.format(
                    actorCounter, phraseCounter, emotionCounter), fft_emotion_features_angle_half, delimiter=",")
                emotionCounter += 1
            phraseCounter += 1
        actorCounter += 1


extractAndSaveAllDataFFT()


def get_covered_data_percentile(data_array, deviation):
    if deviation < 0:
        raise ValueError('deviation must be a positive value')
    median_value = numpy.median(data_array)
    values_in_range = filter(lambda data: data <= median_value +
                             deviation or data >= median_value - deviation, data_array)

    return len(values_in_range) / len(data_array)

# domain is either 'time' or 'frequency'
# dimension is either 'abs' or 'angle'


def getFeaturesDataFrames(domain, **kwargs):
    actors_range = kwargs.get('actor_numbers') or range(1, 25)
    phrases_range = kwargs.get('phrase_numbers') or range(1, 3)
    emotions_range = kwargs.get('emotion_numbers') or range(1, 9)
    dimension = kwargs.get('dimension')

    openers = {
        'time': open_actor_features,
        'frequency': curry(open_actor_features_FFT,  dimension=dimension),
        'maximums': open_actor_features_maximums
    }

    return unwind_features(
        openers[domain],
        actors_range,
        phrases_range,
        emotions_range
    )


def showGraphs(domain, **kwargs):
    actors_range = kwargs.get('actor_numbers') or range(1, 25)
    phrases_range = kwargs.get('phrase_numbers') or range(1, 3)
    emotions_range = kwargs.get('emotion_numbers') or range(1, 9)
    features_range = kwargs.get('feature_labels') or (
        feature_labels if domain == 'time' else feature_labels_FFT)

    dimension = kwargs.get('dimension')
    formatted_domain = domain if domain == 'time' else 'FTT({})'.format(
        dimension if dimension == 'angle' else 'abs')

    features_data_frames = getFeaturesDataFrames(domain, **kwargs)

    print 'Generating {} graphs for\n      features: {}\n      emotions: {}\n      phrases: {}\n      actors: {}\n'.format(
        formatted_domain, features_range, emotions_range, phrases_range, actors_range)
    colors = ['r', 'g', 'b', 'y']
    i = 0
    for actor_features in features_data_frames:
        for phrase_features in actor_features:
            for emotion_features in phrase_features:
                for feature in features_range:
                    color = colors[i % 4]
                    i += 1
                    plt.plot(
                        range(0, len(emotion_features[feature])), emotion_features[feature], color)
        print '{} graphs loaded - {}%'.format(i, 100*i/(len(features_data_frames)*len(
            actor_features)*len(actor_features[0])*len(feature_labels)))

    plt.show()
