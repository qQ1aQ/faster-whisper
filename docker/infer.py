import os
import tempfile

from flask import Flask, jsonify, request

from faster_whisper import WhisperModel

app = Flask(__name__)
