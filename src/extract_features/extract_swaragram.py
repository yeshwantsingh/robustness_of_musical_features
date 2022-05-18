import os
import numpy as np
import pickle
from swaragram import core, utils
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

warnings.simplefilter('ignore')

datasets_path = '../../data'
datasets = ['carnatic', 'hindustani', 'gtzan', 'extendedballroom', 'homburg', 'ismir04']


def process_song(song):
    sr = 22050
    slices = [song[int(i * sr): int((i + 1) * sr)] for i in np.arange(0, int(len(song) / sr), 0.5)[:-1]]
    tonic = utils.getTonic(song)

    def extract_swara(x, sr=22050):
            stft = utils.computeSTFT(x)
            refPitch, refFreq = utils.computeRefPitch(tonic)

            ret = core.computeSwaragram(stft, refPitch=refPitch, refFreq=refFreq)

            return (23 * np.log10(2.220446049250313e-16 + ret))



    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_swara, slices)
    return result


def process_genre(i, folder):
    dataset = {
        'genre': [],
        'swaragram': [],
        'label': []
    }

    with open(folder, 'rb') as f:
        data = pickle.load(f)

    for f in tqdm(data['data']):
        chroma = process_song(f)
        if chroma:
            for c in chroma:
                dataset['genre'].append(folder)
                dataset['swaragram'].append(c)
                dataset['label'].append(i)

    dataset['genre'] = np.array(dataset['genre']).astype('S')
    dataset['swaragram'] = np.array(dataset['swaragram'])
    dataset['label'] = np.array(dataset['label'])

    return dataset


def process_genres(dataset):
    dataset_path = os.path.join(datasets_path, dataset)
    folders = sorted(os.listdir(dataset_path))
    folders = [folder for folder in folders if folder.split('.')[-1] == 'pickle']
    folders = [os.path.join(dataset_path, folder) for folder in folders]
    features_path = '../../features/swaragram'
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
