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
from sklearn.model_selection import ShuffleSplit
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
    df = df.rename(columns={0: 'file'})
    df = df.sample(frac=1).reset_index(drop=True)

    features_label = df.apply(extract_features, axis=1)
    print(features_label)

    features = []
    for i in range(0, len(features_label)):
        features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                        features_label[i][2], features_label[i][3],
                                        features_label[i][4]), axis=0))

    speaker = []
    for i in range(0, len(df)):
        speaker.append(df['file'][i].split('-')[0])
    df['speaker'] = speaker
    print(df)
    X = np.array(features)
    y = np.array(speaker)
    return df, X, y


def creating_model(X, y):
    lb = LabelEncoder()
    y = to_categorical(lb.fit_transform(y))

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_valid = ss.transform(X_valid)
    X_test = ss.transform(X_test)

    rs = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    rs.get_n_splits(X_train)
    rs.get_n_splits(X_test)
    rs.get_n_splits(X_valid)
    rs.get_n_splits(y_train)
    rs.get_n_splits(y_valid)
    rs.get_n_splits(y_test)

    model = Sequential()
    model.add(Dense(193, input_shape=(193,), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    history = model.fit(X_train, y_train, batch_size=128, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[early_stop], shuffle=True)

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    print(model.evaluate(X_test, y_test))

    return train_accuracy, val_accuracy


if __name__ == "__main__":
    df, X, y = preprocess_files(os.listdir('files'))
