import matplotlib.pyplot as plt
import librosa
import pathlib
import os

WORKING_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_ROOT_DIR = os.path.join(WORKING_DIR, 'data')
DATA_MUSIC_DIR = os.path.join(DATA_ROOT_DIR, 'genres_original')

# Extracting the spectogram for every audio
cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10, 10))
genres = os.listdir(f'{DATA_MUSIC_DIR}')

for genre in genres:
    pathlib.Path(f'{DATA_ROOT_DIR}/img_data/specgrams/{genre}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'{DATA_MUSIC_DIR}/{genre}'):
        absolute_song_path = f'{DATA_MUSIC_DIR}/{genre}/{filename}'
        y, sr = librosa.load(absolute_song_path, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128,
                     cmap=cmap, sides='default', mode='default', scale='dB')
        plt.axis('off')
        plt.savefig(f'{DATA_ROOT_DIR}/img_data/specgrams/{genre}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
