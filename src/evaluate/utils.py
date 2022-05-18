import os
import pickle
import numpy as np

def load_data(feature=None, dataset=None):
    print(os.getcwd())
    print(feature, dataset)
    features_dir = os.path.join('../', '../', '../', '../', 'features')
    feature_path = os.path.join(features_dir, feature, dataset)
    print(feature_path)
    data_files = os.listdir(feature_path)
    feature = 'chroma' if feature == 'chromagram' else feature
    feature_data = {'genre': [],
                    'label': [],
                    feature: [],}
    for f in sorted(data_files):
        with open(os.path.join(feature_path, f), 'rb') as f:
            data = pickle.load(f)
        feature_data['genre'] += data['genre'].tolist()
        feature_data['label'] += data['label'].tolist()
        feature_data[feature] += data[feature].tolist()

    feature_data['genre'] = np.array(feature_data['genre'])
    feature_data['label'] = np.array(feature_data['label'])
    feature_data[feature] = np.array(feature_data[feature])

    return feature_data

