import librosa
import numpy as np
import pandas as pd
import csv
import os


data_file_name = 'simple_28features_30sec'

WORKING_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_ROOT_DIR = os.path.join(WORKING_DIR, 'data')
DATA_MUSIC_DIR = os.path.join(DATA_ROOT_DIR, 'genres_original')

headers = ['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
           'rolloff', 'zero_crossing_rate', 'tempo'] + [f' mfcc{i}' for i in range(1, 21)] + ['label']

genres = [g for g in os.listdir(f'{DATA_MUSIC_DIR}') if not g.startswith(".")]

with open(f'{DATA_ROOT_DIR}/{data_file_name}.csv', 'w', newline='') as file:
    writer = csv.writer(file)  # objeto writer para escribir dentro del archivo
    writer.writerow(headers)  # escribimos el encabezado

    for genre in genres:
        for filename in os.listdir(f'{DATA_MUSIC_DIR}/{genre}'):
            songname = f'{DATA_MUSIC_DIR}/{genre}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y=y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            to_append = [filename, np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent),
                         np.mean(spec_bw), np.mean(rolloff), np.mean(zcr), tempo] + [np.mean(e) for e in mfcc] + [genre]

            writer.writerow(to_append)
