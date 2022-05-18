import os
import numpy as np
import librosa
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.simplefilter('ignore')

datasets_path = '../../data'
datasets = ['carnatic', 'hindustani']


def extract_mel(y, sr=22050):
    mel = librosa.feature.melspectrogram(y, sr)
    h, p =  librosa.decompose.hpss(mel)
    return h


def process_song(song):
    sr = 22050
    slices = [song[int(i * sr): int((i + 3) * sr)] for i in np.arange(0, int(len(song) / sr), 5)[:-1]]

    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_mel, slices)
    return result


def process_genre(i, folder):
    dataset = {
        'genre': [],
        'hpss': [],
        'label': []
    }

    with open(folder, 'rb') as f:
        data = pickle.load(f)

    for f in tqdm(data['data']):
        chroma = process_song(f)
        if chroma:
            for c in chroma:
                dataset['genre'].append(folder.split('/')[-1].split('.')[0])
                dataset['hpss'].append(c)
                dataset['label'].append(i)

    dataset['genre'] = np.array(dataset['genre']).astype('S')
    dataset['hpss'] = np.array(dataset['hpss'])
    dataset['label'] = np.array(dataset['label'])

    return dataset


def process_genres(dataset):
    dataset_path = os.path.join(datasets_path, dataset)
    folders = sorted(os.listdir(dataset_path))
    folders = [folder for folder in folders if folder.split('.')[-1] == 'pickle']
    folders = [os.path.join(dataset_path, folder) for folder in folders]
    features_path = '../../features/hpss'
    for i, folder in enumerate(tqdm(folders)):
        data = process_genre(i, folder)
        dst_path = folder.split('/')[-2]
        folder = folder.split('/')[-1].split('.')[0]
        dst_path = os.path.join(features_path, dst_path, folder + '.pickle')
        with open(dst_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for dataset in datasets:
        print(dataset.upper(), ':')
        process_genres(dataset)
