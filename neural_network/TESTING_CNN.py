import numpy as np
import pandas as pd
from joblib import dump, load
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.simplefilter("ignore")



def get(file_name):


        def get_decoded(pred_y):
                classes = ["AI Generated","Human Voice"]
                if pred_y<0:
                        pred_y=0
                if pred_y>1:
                        pred_y=1
                return classes[pred_y]

        y, sr = librosa.load(file_name, mono=True, duration=30)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        XX = [np.mean(chroma_stft), np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(zcr)]
        model = load('NN.H5')

        for e in mfcc:
                XX.append(np.mean(e))


        pred_y = get_decoded(round(model.predict([XX])[0]))

        print(pred_y)
        return pred_y





