from __future__ import print_function
from flask import Flask, request
import sys
import magic
import base64
import pyAudioAnalysis
import mimetypes
import uuid

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api/v1/audio/upload', methods=['POST'])
def home():
    req_data = request.get_json()
    audio_base64 = req_data['audio']
    audio_decoded = base64.b64decode(str(audio_base64))
    return file

app.run()
