import os
import numpy as np
import librosa
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.simplefilter('ignore')

datasets_path = '../../data'
datasets = ['gtzan']


def extract_mel(x, sr=22050):
    return librosa.power_to_db((librosa.feature.melspectrogram(y=x, sr=sr)), ref=np.max)


def process_song(song, frame, slide):
    sr = 22050
    slices = [song[int(i * sr): int((i + frame) * sr)] for i in np.arange(0, int(len(song) // sr), slide)[:-5]]

    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_mel, slices)
    return result


def process_genre(i, folder, frame, slide):
    dataset = {
        'genre': [],
        'mel_spec': [],
        'label': []
    }

    with open(folder, 'rb') as f:
        data = pickle.load(f)

    for f in tqdm(data['data']):
        chroma = process_song(f, frame, slide)
        if chroma:
            for c in chroma:
                dataset['genre'].append(folder.split('/')[-1].split('.')[0])
                dataset['mel_spec'].append(c)
                dataset['label'].append(i)

    dataset['genre'] = np.array(dataset['genre']).astype('S')
    dataset['mel_spec'] = np.array(dataset['mel_spec'])
    dataset['label'] = np.array(dataset['label'])

    return dataset


def process_genres(dataset, frame, slide):
    dataset_path = os.path.join(datasets_path, dataset)
    folders = sorted(os.listdir(dataset_path))
    folders = [folder for folder in folders if folder.split('.')[-1] == 'pickle']
    folders = [os.path.join(dataset_path, folder) for folder in folders]
    features_path = '../../features/mel_spec'
    for i, folder in enumerate(tqdm(folders)):
        data = process_genre(i, folder, frame, slide)
        dst_path = folder.split('/')[-2]
        folder = folder.split('/')[-1].split('.')[0]
        dst_path = os.path.join(features_path, dst_path, '_'+str(frame)+'_'+str(slide), folder + '.pickle')
        with open(dst_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for frame in np.arange(2.5, 3.1, 0.5):
        for slide in np.arange(.5, 3.1, 0.5):
            for dataset in datasets:
                print(dataset.upper(),': ', frame, slide)
                process_genres(dataset, frame, slide)
