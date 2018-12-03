from numpy import genfromtxt
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.signal import argrelextrema
from sklearn.naive_bayes import GaussianNB
import numpy
import copy
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import heapq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_fft(dimension, data_array):
    ftt_data = fft(data_array)

    if dimension == 'angle':
        return numpy.angle(ftt_data)
    else:
        return numpy.abs(ftt_data)


def get_first_natural_frequency(data_array):
    ftt_data = apply_fft('abs', data_array)
    indexes_of_local_max, = argrelextrema(ftt_data, numpy.greater)
    values_of_local_max = [ftt_data[index] for index in indexes_of_local_max]
    greatest_local_max = max(values_of_local_max)
    return list(ftt_data).index(greatest_local_max)


def get_second_natural_frequency(data_array):
    ftt_data = apply_fft('abs', data_array)
    indexes_of_local_max, = argrelextrema(ftt_data, numpy.greater)
    values_of_local_max = [ftt_data[index] for index in indexes_of_local_max]
    values_of_local_max.sort(reverse=True)
    greatest_local_max = values_of_local_max[1]
    return list(ftt_data).index(greatest_local_max)


def get_third_natural_frequency(data_array):
    ftt_data = apply_fft('abs', data_array)
    indexes_of_local_max, = argrelextrema(ftt_data, numpy.greater)
    values_of_local_max = [ftt_data[index] for index in indexes_of_local_max]
    values_of_local_max.sort(reverse=True)
    greatest_local_max = values_of_local_max[2]
    return list(ftt_data).index(greatest_local_max)
# feature_reducers = {
#     'Zero Crossing Rate':  [numpy.std],
#     'Energy': [numpy.std],
#     'Entropy of Energy': [numpy.std],
#     'Spectral Centroid': [numpy.std],
#     'Spectral Spread': [numpy.std],
#     'Spectral Entropy': [numpy.std],
#     'Spectral Flux': [numpy.std],
#     'Spectral Rolloff': [numpy.std],
#     'MFCC1': [numpy.std],
#     'MFCC2': [numpy.std],
#     'MFCC3': [numpy.std],
#     'MFCC4': [numpy.std],
#     'MFCC5': [numpy.std],
#     'MFCC6': [numpy.std],
#     'MFCC7': [numpy.std],
#     'MFCC8': [numpy.std],
#     'MFCC9': [numpy.std],
#     'MFCC10': [numpy.std],
#     'MFCC11': [numpy.std],
#     'MFCC12': [numpy.std],
#     'MFCC13': [numpy.std],
#     'Chroma Vector 1': [numpy.std],
#     'Chroma Vector 2': [numpy.std],
#     'Chroma Vector 3': [numpy.std],
#     'Chroma Vector 4': [numpy.std],
#     'Chroma Vector 5': [numpy.std],
#     'Chroma Vector 6': [numpy.std],
#     'Chroma Vector 7': [numpy.std],
#     'Chroma Vector 8': [numpy.std],
#     'Chroma Vector 9': [numpy.std],
#     'Chroma Vector 10': [numpy.std],
#     'Chroma Vector 11': [numpy.std],
#     'Chroma Vector 12': [numpy.std],
#     'Chroma Deviation': [numpy.std],
# }

# feature_reducers = {
#     'Zero Crossing Rate':  [numpy.mean],
#     'Energy': [numpy.mean],
#     'Entropy of Energy': [numpy.mean],
#     'Spectral Centroid': [numpy.mean],
#     'Spectral Spread': [numpy.mean],
#     'Spectral Entropy': [numpy.mean],
#     'Spectral Flux': [numpy.mean],
#     'Spectral Rolloff': [numpy.mean],
#     'MFCC1': [numpy.mean],
#     'MFCC2': [numpy.mean],
#     'MFCC3': [numpy.mean],
#     'MFCC4': [numpy.mean],
#     'MFCC5': [numpy.mean],
#     'MFCC6': [numpy.mean],
#     'MFCC7': [numpy.mean],
#     'MFCC8': [numpy.mean],
#     'MFCC9': [numpy.mean],
#     'MFCC10': [numpy.mean],
#     'MFCC11': [numpy.mean],
#     'MFCC12': [numpy.mean],
#     'MFCC13': [numpy.mean],
#     'Chroma Vector 1': [numpy.mean],
#     'Chroma Vector 2': [numpy.mean],
#     'Chroma Vector 3': [numpy.mean],
#     'Chroma Vector 4': [numpy.mean],
#     'Chroma Vector 5': [numpy.mean],
#     'Chroma Vector 6': [numpy.mean],
#     'Chroma Vector 7': [numpy.mean],
#     'Chroma Vector 8': [numpy.mean],
#     'Chroma Vector 9': [numpy.mean],
#     'Chroma Vector 10': [numpy.mean],
#     'Chroma Vector 11': [numpy.mean],
#     'Chroma Vector 12': [numpy.mean],
#     'Chroma Deviation': [numpy.mean],
# }

# feature_reducers = {
#     'Zero Crossing Rate':  [numpy.median],
#     'Energy': [numpy.median],
#     'Entropy of Energy': [numpy.median],
#     'Spectral Centroid': [numpy.median],
#     'Spectral Spread': [numpy.median],
#     'Spectral Entropy': [numpy.median],
#     'Spectral Flux': [numpy.median],
#     'Spectral Rolloff': [numpy.median],
#     'MFCC1': [numpy.median],
#     'MFCC2': [numpy.median],
#     'MFCC3': [numpy.median],
#     'MFCC4': [numpy.median],
#     'MFCC5': [numpy.median],
#     'MFCC6': [numpy.median],
#     'MFCC7': [numpy.median],
#     'MFCC8': [numpy.median],
#     'MFCC9': [numpy.median],
#     'MFCC10': [numpy.median],
#     'MFCC11': [numpy.median],
#     'MFCC12': [numpy.median],
#     'MFCC13': [numpy.median],
#     'Chroma Vector 1': [numpy.median],
#     'Chroma Vector 2': [numpy.median],
#     'Chroma Vector 3': [numpy.median],
#     'Chroma Vector 4': [numpy.median],
#     'Chroma Vector 5': [numpy.median],
#     'Chroma Vector 6': [numpy.median],
#     'Chroma Vector 7': [numpy.median],
#     'Chroma Vector 8': [numpy.median],
#     'Chroma Vector 9': [numpy.median],
#     'Chroma Vector 10': [numpy.median],
#     'Chroma Vector 11': [numpy.median],
#     'Chroma Vector 12': [numpy.median],
#     'Chroma Deviation': [numpy.median],
# }


# feature_reducers = {
#     'Zero Crossing Rate':  [numpy.sum],
#     'Energy': [numpy.sum],
#     'Entropy of Energy': [numpy.sum],
#     'Spectral Centroid': [numpy.sum],
#     'Spectral Spread': [numpy.sum],
#     'Spectral Entropy': [numpy.sum],
#     'Spectral Flux': [numpy.sum],
#     'Spectral Rolloff': [numpy.sum],
#     'MFCC1': [numpy.sum],
#     'MFCC2': [numpy.sum],
#     'MFCC3': [numpy.sum],
#     'MFCC4': [numpy.sum],
#     'MFCC5': [numpy.sum],
#     'MFCC6': [numpy.sum],
#     'MFCC7': [numpy.sum],
#     'MFCC8': [numpy.sum],
#     'MFCC9': [numpy.sum],
#     'MFCC10': [numpy.sum],
#     'MFCC11': [numpy.sum],
#     'MFCC12': [numpy.sum],
#     'MFCC13': [numpy.sum],
#     'Chroma Vector 1': [numpy.sum],
#     'Chroma Vector 2': [numpy.sum],
#     'Chroma Vector 3': [numpy.sum],
#     'Chroma Vector 4': [numpy.sum],
#     'Chroma Vector 5': [numpy.sum],
#     'Chroma Vector 6': [numpy.sum],
#     'Chroma Vector 7': [numpy.sum],
#     'Chroma Vector 8': [numpy.sum],
#     'Chroma Vector 9': [numpy.sum],
#     'Chroma Vector 10': [numpy.sum],
#     'Chroma Vector 11': [numpy.sum],
#     'Chroma Vector 12': [numpy.sum],
#     'Chroma Deviation': [numpy.sum],
# }


# feature_reducers = {
#     'Zero Crossing Rate':  [ numpy.std, get_natural_frequency],
#     'Energy': [numpy.sum,  numpy.std, get_natural_frequency],
#     'Entropy of Energy': [numpy.sum,  numpy.std, get_natural_frequency],
#     'Spectral Centroid': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Spectral Spread': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Spectral Entropy': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Spectral Flux': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Spectral Rolloff': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC1': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC2': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC3': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC4': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC5': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC6': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC7': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC8': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC9': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC10': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC11': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC12': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'MFCC13': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 1': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 2': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 3': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 4': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 5': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 6': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 7': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 8': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 9': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 10': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 11': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Vector 12': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
#     'Chroma Deviation': [numpy.mean, numpy.median, numpy.std, get_natural_frequency],
# }

feature_reducers = {
    'Zero Crossing Rate':  [get_second_natural_frequency],
    'Energy': [get_second_natural_frequency],
    'Entropy of Energy': [get_second_natural_frequency],
    'Spectral Centroid': [get_second_natural_frequency],
    'Spectral Spread': [get_second_natural_frequency],
    'Spectral Entropy': [get_second_natural_frequency],
    'Spectral Flux': [get_second_natural_frequency],
    'Spectral Rolloff': [get_second_natural_frequency],
    'MFCC1': [get_second_natural_frequency],
    'MFCC2': [get_second_natural_frequency],
    'MFCC3': [get_second_natural_frequency],
    'MFCC4': [get_second_natural_frequency],
    'MFCC5': [get_second_natural_frequency],
    'MFCC6': [get_second_natural_frequency],
    'MFCC7': [get_second_natural_frequency],
    'MFCC8': [get_second_natural_frequency],
    'MFCC9': [get_second_natural_frequency],
    'MFCC10': [get_second_natural_frequency],
    'MFCC11': [get_second_natural_frequency],
    'MFCC12': [get_second_natural_frequency],
    'MFCC13': [get_second_natural_frequency],
    'Chroma Vector 1': [get_second_natural_frequency],
    'Chroma Vector 2': [get_second_natural_frequency],
    'Chroma Vector 3': [get_second_natural_frequency],
    'Chroma Vector 4': [get_second_natural_frequency],
    'Chroma Vector 5': [get_second_natural_frequency],
    'Chroma Vector 6': [get_second_natural_frequency],
    'Chroma Vector 7': [get_second_natural_frequency],
    'Chroma Vector 8': [get_second_natural_frequency],
    'Chroma Vector 9': [get_second_natural_frequency],
    'Chroma Vector 10': [get_second_natural_frequency],
    'Chroma Vector 11': [get_second_natural_frequency],
    'Chroma Vector 12': [get_second_natural_frequency],
    'Chroma Deviation': [get_second_natural_frequency],
}

emotions_dict = {
    1: "Neutral",
    2: "Calm",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgust",
    8: "Surprised",
}

# emotions_dictionary = {
#     "Neutral":  0,
#     "Calm": 1,
#     "Happy": 2,
#     "Sad": 3,
#     "Angry": 4,
#     "Fearful": 5,
#     "Disgust": 6,
#     "Surprised": 7,
# }

emotions_dictionary = {
    "Happy": 0,
    "Sad": 1,
    "Angry": 2,
    "Fearful": 3,
}

feature_labels = [
    'Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5', 'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10', 'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']

feature_labels_FFT = [
    'domain', 'Zero Crossing Rate', 'Energy', 'Entropy of Energy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Chroma Vector 1', 'Chroma Vector 2', 'Chroma Vector 3', 'Chroma Vector 4', 'Chroma Vector 5', 'Chroma Vector 6', 'Chroma Vector 7', 'Chroma Vector 8', 'Chroma Vector 9', 'Chroma Vector 10', 'Chroma Vector 11', 'Chroma Vector 12', 'Chroma Deviation']

# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 =
# fearful, 07 = disgust, 08 = surprised).


# n_components is the number of components to use o PCA,
# if n_components is < 1, then the number of components
# is automaticly choose to fit the percentage passed in
# n_components, e.g. with n_components assigned to .9
# the function will choose k components so that 90% of
# the variance is explained
def apply_pca(data_frame, n_components):
    features = list(set(data_frame) - set(['Emotion']))
    data = data_frame.loc[:, features].values
    emotions = data_frame.loc[:, ['Emotion']].values

    data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=n_components).fit(data)
    accumulated_ratio = 0
    explained_variance = 0
    for index, ratio in enumerate(pca.explained_variance_ratio_):
        accumulated_ratio += ratio
        if (index < pca.n_components_):
            explained_variance += ratio
        print 'Component n {:>2} Eplained ratio: {:.3f}%         Accumulated ratio: {:.3f}%'.format(
            index, ratio * 100, accumulated_ratio * 100)
    print '\nChoosed components: {}\nExplained Variance: {:.3f}%\n'.format(
        pca.n_components_, explained_variance * 100)
    pca_data_frame = pd.DataFrame(data=pca.transform(data), columns=[
        'Component {}'.format(i) for i in range(0, pca.n_components_)])
    pca_data_frame['Emotion'] = emotions

    return pca_data_frame


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


def numeros_menores(array, num):
    cont = 0
    for numero in array:
        if numero < num:
            cont += 1
    return cont

# def split_features_labels(data_set):
#     features = data_set.data
#     labels = data_set.target
#     return features, labels


def split_train_test(features, labels, test_size):
    total_test_size = int(len(features) * test_size)
    print total_test_size
    numpy.random.seed(2)
    indices = numpy.random.permutation(len(features))
    train_features = features[indices[:-total_test_size]]
    train_labels = labels[indices[:-total_test_size]]
    test_features = features[indices[-total_test_size:]]
    test_labels = labels[indices[-total_test_size:]]
    return train_features, train_labels, test_features, test_labels


# 5 partes
def split_list(input_array, wanted_parts=1):
    backup_array = numpy.array(input_array)
    splitted_array = numpy.array_split(backup_array, wanted_parts)
    return splitted_array


def generate_shuffle_positions(positions):
    a = list(range(positions))
    a = numpy.array(a)
    numpy.random.shuffle(a)
    return a


def shuffle_array(array, shuffle_positions):
    backup_array = numpy.array(array)
    shuffled_array = []
    for i in range(0, len(shuffle_positions)):
        shuffled_array.append(backup_array[shuffle_positions[i]])
    return shuffled_array


def statistic_matrix_init():
    return [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    # return [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0,
    # 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0,
    # 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]


def test_model(emotions, data):
    emotions_train, emotions_test = emotions
    data_train, data_test = data
    X = numpy.array(data_train)
    Y = numpy.array(emotions_train)
    classifier = naive_bayes_init()
    classifier.fit(X, Y)
    statistics = {}
    for i, emotion in enumerate(emotions_test.flatten()):
        [predict] = classifier.predict([data_test[i]])
        if emotion not in statistics:
            statistics[emotion] = {predict: 0}
        if predict not in statistics[emotion]:
            statistics[emotion][predict] = 1
        else:
            statistics[emotion][predict] += 1

    emotions, counts = numpy.unique(
        emotions_test.flatten(), return_counts=True)
    occourences = dict(zip(emotions, counts))
    for emotion in statistics:
        for predicted_emotion in statistics[emotion]:
            statistics[emotion][predicted_emotion] = statistics[emotion][predicted_emotion] / \
                float(occourences[emotion])
    return statistics


def statistic_data(emotion, predict, stat_matrix):
    stat_matrix[emotions_dictionary[emotion[0]]][
        emotions_dictionary[predict[0]]] += 1


def get_train_test_array(splitted_array, iteration):
    array = copy.deepcopy(splitted_array)
    test_array = array[iteration]
    if len(array) > 1:
        del array[iteration]
    train_array = numpy.concatenate(array)
    return numpy.array(train_array), numpy.array(test_array)


def split_array(labels_array, data_array):
    labels_test = []
    data_test = []
    data_backup = copy.deepcopy(data_array)
    labels_backup = copy.deepcopy(labels_array)
    numeros = []
    cont = 0
    num_sorteados = []
    # 4 emocoes 192
    # 2 emocoes 96
    # todas emocoes 384
    num_sorteados = sorteia_numeros(.2, 192)
    for num in num_sorteados:
        cont = numeros_menores(numeros, num)
        numeros.append(num)
        # print "Numero sorteado: "+str(num)
        labels_test.append(labels_backup[num])
        data_test.append(data_backup[num])
        del data_array[num - cont]
        del labels_array[num - cont]
        # print "Tamanho Backup: " + str(len(labels_backup))
        # print "Tamanho Original: " + str(len(labels_array))
    return labels_test, data_test


def sorteia_numeros(porcentagem, quantidade_total):
    n = 0
    num_sorteados = []
    num = int(quantidade_total * porcentagem)
    while (n < num):
        sorteado = random.randint(0, quantidade_total - 1)
        if sorteado not in num_sorteados:
            num_sorteados.append(sorteado)
            n += 1
    return num_sorteados


def naive_bayes_init():
    # Cria o classifier
    return GaussianNB()


def naive_bayes_train(clf, feature_data_array, feature_labels_array):
        # Treina o classifier
    X = numpy.array(feature_data_array)
    Y = numpy.array(feature_labels_array)
    clf.fit(X, Y)


def naive_bayes_predictions(clf, feature_data_test_array):
    # Testa o classifier
    test = []
    test.append(feature_data_test_array)
    predicted = clf.predict(test)


def unwind_features(root, *args):
    if len(args) == 0:
        return root
    else:
        head, tail = args[0], args[1:]
        new_roots = [root(number) for number in head]
        return [unwind_features(new_root, *tail) for new_root in new_roots]


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


def create_category(deviation, feature_value):
    return int(feature_value / (deviation * 2))


def categorize_data(deviation, feature_array):
    return [create_category(deviation, value)
            for value in feature_array]


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
                # emotion_features = todas as features de uma emocao de uma
                # frase de um ator

                # transposedArray tem 34 arrays contendo os valores de cada
                # feature por posicao
                transposedArray = numpy.transpose(emotion_features.values)

                # fft eh o fft de todas as features de uma emocao de uma frase
                # por ator
                fft_emotion_features_abs = extractFFT(transposedArray, 'abs')
                fft_emotion_features_angle = extractFFT(
                    transposedArray, 'angle')

                fft_emotion_features_abs_half = fft_emotion_features_abs[:len(
                    fft_emotion_features_abs) / 2 - 1]
                fft_emotion_features_angle_half = fft_emotion_features_angle[:len(
                    fft_emotion_features_angle) / 2 - 1]
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


def get_covered_data_percentile(data_array, deviation):
    if deviation < 0:
        raise ValueError('deviation must be a positive value')
    median_value = numpy.median(data_array)
    values_in_range = filter(lambda data: data <= median_value +
                             deviation or data >= median_value - deviation, data_array)

    return len(values_in_range) / len(data_array)

# domain is either 'time' or 'frequency'
# dimension is either 'abs' or 'angle'


def get_features_data_frames(domain, **kwargs):
    actors_range = kwargs.get('actor_numbers') or range(1, 25)
    phrases_range = kwargs.get('phrase_numbers') or range(1, 5)
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


def unify_data_frames(data_frames, reducers, emotions_dict=emotions_dict):
    unified_data_frame = pd.DataFrame()

    for actor in data_frames:
        for phrase in actor:
            emotion_number = 1
            for emotion in phrase:
                columns_dict = {}
                for label in feature_labels:
                    new_columns_data = [
                        {
                            '{}_{}'.format(label, index): reducer(emotion[label])
                        }
                        for index, reducer in enumerate(reducers[label])
                    ]
                    columns_dict = reduce(
                        merge_dicts, new_columns_data, columns_dict)
                columns_dict['Emotion'] = emotions_dict[emotion_number]
                unified_data_frame = unified_data_frame.append(
                    columns_dict, ignore_index=True)
                emotion_number += 1

    return unified_data_frame


def show_graphs(domain, **kwargs):
    actors_range = kwargs.get('actor_numbers') or range(1, 25)
    phrases_range = kwargs.get('phrase_numbers') or range(1, 3)
    emotions_range = kwargs.get('emotion_numbers') or range(1, 9)
    features_range = kwargs.get('feature_labels') or (
        feature_labels if domain == 'time' else feature_labels_FFT)

    dimension = kwargs.get('dimension')
    formatted_domain = domain if domain == 'time' else 'FTT({})'.format(
        dimension if dimension == 'angle' else 'abs')

    features_data_frames = get_features_data_frames(domain, **kwargs)

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
        print '{} graphs loaded - {}%'.format(i, 100 * i / (len(features_data_frames) * len(
            actor_features) * len(actor_features[0]) * len(feature_labels)))

    plt.show()


def create_emotion_maximums_files():
    featureMaximums = get_features_data_frames('maximums')

    emotion_maximums_array = map(lambda n: [], range(1, 9))

    for actor in featureMaximums:
        for phrase in actor:
            emotion_number = 1
            for emotion in phrase:
                maximums_array = []
                i = 0
                for label in feature_labels:
                    maximums_array.append(emotion[label][0])
                    i += 1
                emotion_maximums_array[emotion_number -
                                       1].append(maximums_array)
                emotion_number += 1

    for i in range(1, 9):
        df = pd.DataFrame(data=numpy.matrix(
            emotion_maximums_array[i - 1]).astype(float))
        df.to_csv('Maximums/Emotion_{:0>2}.csv'.format(i),
                  header=False, index=False)
