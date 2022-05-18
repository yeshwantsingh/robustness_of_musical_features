import os
import numpy as np
import librosa
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.simplefilter('ignore')

datasets_path = '../../data'
datasets = ['carnatic', 'homburg', 'hindustani', 'ismir04', 'extendedballroom']

def extract_tonnetz(x, sr=22050):
    return librosa.feature.tonnetz(y=x, sr=sr)


def process_song(song):
    sr = 22050
    slices = [song[int(i * sr): int((i + 1) * sr)] for i in np.arange(0, int(len(song) / sr), 0.5)[:-1]]

    with ThreadPoolExecutor(max_workers=61) as ex:
        result = ex.map(extract_tonnetz, slices)
    return result


def process_genre(i, folder):
    dataset = {
        'genre': [],
        'tonnetz': [],
        'label': []
    }

    with open(folder, 'rb') as f:
        data = pickle.load(f)

    for f in tqdm(data['data']):
        chroma = process_song(f)
        if chroma:
            for c in chroma:
                dataset['genre'].append(folder)
                dataset['tonnetz'].append(c)
                dataset['label'].append(i)

    dataset['genre'] = np.array(dataset['genre']).astype('S')
    dataset['tonnetz'] = np.array(dataset['tonnetz'])
    dataset['label'] = np.array(dataset['label'])

    return dataset


def process_genres(dataset):
    dataset_path = os.path.join(datasets_path, dataset)
    folders = sorted(os.listdir(dataset_path))
    folders = [folder for folder in folders if folder.split('.')[-1] == 'pickle']
    folders = [os.path.join(dataset_path, folder) for folder in folders]
    features_path = '../../features/tonnetz'
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
