#!/usr/bin/env python

from __future__ import print_function
from flask import Flask, request, jsonify
import sys
import base64
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO
import mimetypes
import uuid
from functions import *

CATEGORIES_DEVIATION = 0.001
EXPLAINED_VARIANCE = .8

app = Flask(__name__)
app.config["DEBUG"] = False

data_frames = get_features_data_frames('time')
unified_data_frame = unify_data_frames(data_frames, feature_reducers)
pca_data_frame, pca_model, scaler_model = apply_pca(
    unified_data_frame, n_components=EXPLAINED_VARIANCE)
components = list(set(pca_data_frame) - set(['Emotion']))
for component in components:
    pca_data_frame[component] = categorize_data(
        CATEGORIES_DEVIATION, pca_data_frame[component])
data = pca_data_frame.loc[:, components].values
emotions = pca_data_frame.loc[:, ['Emotion']].values

X = numpy.array(data)
Y = numpy.array(emotions)
classifier = naive_bayes_init()
classifier.fit(X, Y)


@app.route('/api/v1/audio/classify-emotion', methods=['POST'])
def home():
    req_data = request.get_json()
    audio_path = 'audios/{}'.format(req_data['audio_path'])
    print(audio_path)
    fs, audio_signal = audioBasicIO.readAudioFile(audio_path)
    audio_features, _feature_labels = audioFeatureExtraction.stFeatureExtraction(
        audio_signal, fs, fs * .05, fs * .05)
    reduced_features = []
    for feature_index, feature in enumerate(feature_labels):
        for reducer in feature_reducers[feature]:
            reduced_features.append(reducer(audio_features[feature_index]))
    normalized_features = scaler_model.transform(
        [numpy.transpose(reduced_features)])
    pca_features = pca_model.transform(normalized_features)
    [prediction] = classifier.predict(pca_features)
    print(audio_features, _feature_labels)
    return jsonify({'Emotion': prediction})


app.run()
