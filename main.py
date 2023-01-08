import os
import pandas as pd
import librosa
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def extract_features(files):
    file_name = os.path.join(os.path.abspath('files') + '/' + str(files.file))
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)

    return mfcc, chroma, mel, contrast, tonnetz


def preprocess_files(file_list):
    df = pd.DataFrame(file_list)
    df.columns = ['file']
    features_label = df.apply(extract_features, axis=1)
    print(features_label)
    features = []
    for i in range(0, len(features_label)):
        features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                        features_label[i][2], features_label[i][3],
                                        features_label[i][4]), axis=0))
    df1 = pd.DataFrame(features)
    speaker = []
    for i in range(0, len(df)):
        speaker.append(df['file'][i].split('-')[0])
    df1['speaker'] = speaker
    print (df1)
    return df1

def get_speaker(speaker):
    speaker = str(speaker)
    if speaker == "[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca1"
    elif speaker == "[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca2"
    elif speaker == "[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca3"
    elif speaker == "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca4"
    elif speaker == "[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca5"
    elif speaker == "[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca6"
    elif speaker == "[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca7"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]":
        return "Mowca8"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]":
        return "Mowca9"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]":
        return "Mowca10"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]":
        return "Mowca11"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]":
        return "Mowca12"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]":
        return "Mowca13"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]":
        return "Mowca14"
    elif speaker == "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]":
        return "Mowca15"
    else:
        return "Nieznany_mowca"


def model_output(X_data, y_data):
    print('\n Predykcja mówców')
    for i in range(len(y_data)):
        prediction = get_speaker(np.round(model.predict(X_data[i:i + 1])[0]))
        speaker = get_speaker(y_data[i])
        print("mowca={0:15s}- predykcja={1:15s}- dopasowanie?={2}".format(speaker, prediction, speaker==prediction))

    plt.figure(figsize=(12, 8))

    plt.plot(train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
    plt.title('Wykres celności na danych treningowych i walidacyjnych', fontsize=25)
    plt.xlabel('Epoch (liczba powtorzen) ', fontsize=18)
    plt.ylabel('Accuracy (celnosc) ', fontsize=18)
    plt.xticks(range(0, 100, 5), range(0, 100, 5))

    plt.legend(fontsize=18)
    plt.show()

if __name__ == "__main__":
    df1 = preprocess_files(os.listdir('files'))

    X = np.array(df1.iloc[:, :-1], dtype = float)
    y = df1.iloc[:, -1]

    lb = LabelEncoder()
    y = to_categorical(lb.fit_transform(y))

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)



    model = Sequential()
    model.add(Dense(193, input_shape=(193,), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    history = model.fit(X_train, y_train, batch_size=128, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[early_stop], shuffle=True)

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    print(model.evaluate(X_test, y_test))

    model_output(X_train[0:15], y_train[0:15])


