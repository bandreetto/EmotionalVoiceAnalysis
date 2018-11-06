from numpy import genfromtxt
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt


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
def open_actor_features_FFT(actor_number, dimension):
    def open_phrase_features(phrase_number):
        def open_emotion_features(emotion_number):
            path
            if dimension == 'angle':
                path = './FFT/Actor_{0:0>2}/Frase_{1}/FFT_0{2}_ANGLE.csv'.format(
                    actor_number, phrase_number, emotion_number)
            else:
                path = './FFT/Actor_{0:0>2}/Frase_{1}/FFT_0{2}_ABS.csv'.format(
                    actor_number, phrase_number, emotion_number)
            return pd.read_csv(path, names=feature_labels_FFT)
        return open_emotion_features
    return open_phrase_features


def getFFTMaximums(fftData, output):
    transposedData = numpy.transpose(fftData)
    i = 0
    maximums = []
    for label in feature_labels_FFT:
        if (i != 0):
            maximums.append(max(transposedData[i]))
        i += 1
    numpy.savetxt(output, maximums, delimiter=",")


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
                if not os.path.exists('FFT/Actor_{0:0>2}/Frase_{1}'.format(actorCounter, phraseCounter)):
                    os.makedirs(
                        'FFT/Actor_{0:0>2}/Frase_{1}'.format(actorCounter, phraseCounter))

                getFFTMaximums(fft_emotion_features_abs, 'FFT/Actor_{0:0>2}/Frase_{1}/FFT_{2:0>2}_MAXIMUMS.csv'.format(
                    actorCounter, phraseCounter, emotionCounter))

                numpy.savetxt('FFT/Actor_{0:0>2}/Frase_{1}/FFT_{2:0>2}_ABS.csv'.format(
                    actorCounter, phraseCounter, emotionCounter), fft_emotion_features_abs, delimiter=",")
                numpy.savetxt('FFT/Actor_{0:0>2}/Frase_{1}/FFT_{2:0>2}_ANGLE.csv'.format(
                    actorCounter, phraseCounter, emotionCounter), fft_emotion_features_angle, delimiter=",")
                emotionCounter += 1
            phraseCounter += 1
        actorCounter += 1

extractAndSaveAllDataFFT()
