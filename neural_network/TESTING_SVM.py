
# import numpy as np
# import pandas as pd
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import OrdinalEncoder
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor

# from joblib import dump, load

# import warnings
# warnings.simplefilter("ignore")

# from sklearn.preprocessing import LabelEncoder, StandardScaler

# import numpy as np
# import pandas as pd
# from joblib import dump, load
# import librosa
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# import os
# from PIL import Image
# import pathlib
# import csv

# # Preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler


# data=pd.read_csv("Features_final.csv")

# genre_list = data.iloc[:, -1]
# encoder = LabelEncoder()
# y = encoder.fit_transform(genre_list)

# print("Y : ",len(y))

# scaler = StandardScaler()

# X = np.array(data.iloc[:, :-1], dtype = float)
# print("X : ",len(X))


# SVM_regressor = SVR()
# SVM_regressor.fit(X,y)
# dump(SVM_regressor, 'SVM_CLASSIFIER.sav')

# print("Done")


# def get(file_name):

#         def get_decoded(pred_y):
#                 classes = ["AI Generated","Human Voice"]
#                 if pred_y<0:
#                         pred_y=0
#                 if pred_y>1:
#                         pred_y=1
#                 return classes[pred_y]

#         y, sr = librosa.load(file_name, mono=True, duration=30)

#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)

#         XX = [np.mean(chroma_stft), np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(zcr)]

#         for e in mfcc:
#                 XX.append(np.mean(e))

#         SVM_regressor = load('SVM_CLASSIFIER.sav')


#         pred_y = get_decoded(round(SVM_regressor.predict([XX])[0]))

#         print(pred_y)
#         return pred_y


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
import librosa.display  # Import librosa.display to avoid AttributeError
import os
from PIL import Image
import pathlib
import csv

warnings.simplefilter("ignore")

# Load and preprocess data
data = pd.read_csv("Features_final.csv")
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print("Y : ", len(y))

scaler = StandardScaler()
X = np.array(data.iloc[:, :-1], dtype=float)
print("X : ", len(X))

# Train SVM regressor
SVM_regressor = SVR()
SVM_regressor.fit(X, y)
dump(SVM_regressor, 'SVM_CLASSIFIER.sav')
print("Done")

def get(file_name):
    def get_decoded(pred_y):
        classes = ["AI Generated", "Human Voice"]
        if pred_y < 0:
            pred_y = 0
        if pred_y > 1:
            pred_y = 1
        return classes[pred_y]

    y, sr = librosa.load(file_name, mono=True, duration=30)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    XX = [np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
    for e in mfcc:
        XX.append(np.mean(e))

    SVM_regressor = load('SVM_CLASSIFIER.sav')
    pred_y = get_decoded(round(SVM_regressor.predict([XX])[0]))

    print(pred_y)

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.title('Chroma STFT')
    librosa.display.specshow(chroma_stft, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.title('MFCC')
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return pred_y

# Example usage
file_name = 'audio_files\Human_Voice\sample-000004.mp3'
get(file_name)
