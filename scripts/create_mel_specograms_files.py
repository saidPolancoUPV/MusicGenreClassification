import matplotlib.pyplot as plt
import librosa
import librosa.display
import pathlib
import numpy as np
import os

WORKING_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_ROOT_DIR = os.path.join(WORKING_DIR, 'data')
DATA_MUSIC_DIR = os.path.join(DATA_ROOT_DIR, 'genres_original')

# Extracting the spectogram for every audio
cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10, 10))
genres = os.listdir(f'{DATA_MUSIC_DIR}')

hop_length = 512  # number audio of frames between STFT columns (looks like a good default)

for genre in genres:
    pathlib.Path(f'{DATA_ROOT_DIR}/img_data/mel_specgrams/{genre}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'{DATA_MUSIC_DIR}/{genre}'):
        absolute_song_path = f'{DATA_MUSIC_DIR}/{genre}/{filename}'
        y, sr = librosa.load(absolute_song_path, mono=True, duration=5)
        y, _ = librosa.effects.trim(y)
        S = librosa.feature.melspectrogram(y, sr=sr)
        S_DB = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log',
                                 cmap=cmap)
        plt.savefig(f'{DATA_ROOT_DIR}/img_data/mel_specgrams/{genre}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
