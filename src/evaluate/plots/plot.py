import json

import numpy as np
from matplotlib import pyplot as plt

plt.style.use('classic')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff6600']


def plot(ax, history, dataset, metric):
    if metric == 'Accuracy':
        key = 'acc'
    elif metric == 'Loss':
        key = 'loss'
    elif metric == 'Validation Accuracy':
        key = 'val_acc'
    else:
        key = 'val_loss'

    epochs = np.arange(1, 101)
    for feature in range(0, 9):
        for model in range(0, 8):

            hist = np.array(history[feature][model][key][:100]) * 100
            if feature == 8 and dataset == 'Carnatic':
                ax[feature].plot(epochs, hist, label='M' + str(model + 1), color=colors[model])
            else:
                ax[feature].plot(epochs, hist, color=colors[model])
            ax[feature].grid(True)
            ax[feature].set_yticks([0, 25, 50, 75, 100])
            ax[feature].set_xticks([0, 25, 50, 75, 100])
            # ax[feature].set_aspect(1.6)
            ax[feature].tick_params(color='k', labelcolor='k', labelsize=14)


def main(metric):
    datasets = ['carnatic', 'gtzan', 'hindustani', 'homburg']

    features = ['chromagram', 'mfcc', 'spectral_contrast', 'swaragram', 'tonnetz', 'stft', 'cqt', 'mel_spec', 'hpss']
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    ax = fig.subplots(4, 9, sharey=True, sharex=True)
    history = []
    for dataset in datasets:
        hist = []
        for feature in features:
            h = []
            for i in [1, 2, 5, 6, 7, 8, 9, 10]:
                with open('../' + dataset + '/' + feature + '/history' + str(i) + '.json') as f:
                    h.append(json.load(f))
            hist.append(h)
        history.append(hist)

    plot(ax[0], history[0], 'Carnatic', metric)
    plot(ax[1], history[1], 'GTZAN', metric)
    plot(ax[2], history[2], 'Hindustani', metric)
    plot(ax[3], history[3], 'Homburg', metric)

    ylabel = ''
    if metric == 'Accuracy' or metric == 'Validation Accuracy':
        ylabel = 'Accuracy'
    else:
        ylabel = 'Loss'

    title = 'Training'
    if metric.split(' ')[0] == 'Validation':
        title = 'Validation'

    ax[0][8].yaxis.set_label_position('right')
    ax[0][8].set_ylabel('Carnatic', fontsize=18, color='k')

    ax[1][8].yaxis.set_label_position('right')
    ax[1][8].set_ylabel('GTZAN', fontsize=18, color='k')

    ax[2][8].yaxis.set_label_position('right')
    ax[2][8].set_ylabel('Hindustani', fontsize=18, color='k')

    ax[3][8].yaxis.set_label_position('right')
    ax[3][8].set_ylabel('Homburg', fontsize=18, color='k')

    ax[0][0].xaxis.set_label_position('top')
    ax[0][0].set_xlabel('Chroma', fontsize=18, color='k')

    ax[0][1].xaxis.set_label_position('top')
    ax[0][1].set_xlabel('MFCC', fontsize=18, color='k')

    ax[0][2].xaxis.set_label_position('top')
    ax[0][2].set_xlabel('SC', fontsize=18, color='k')

    ax[0][3].xaxis.set_label_position('top')
    ax[0][3].set_xlabel('SGram', fontsize=18, color='k')

    ax[0][4].xaxis.set_label_position('top')
    ax[0][4].set_xlabel('Tonnetz', fontsize=18, color='k')

    ax[0][5].xaxis.set_label_position('top')
    ax[0][5].set_xlabel('STFT', fontsize=18, color='k')

    ax[0][6].xaxis.set_label_position('top')
    ax[0][6].set_xlabel('CQT', fontsize=18, color='k')

    ax[0][7].xaxis.set_label_position('top')
    ax[0][7].set_xlabel('MSpec', fontsize=18, color='k')

    ax[0][8].xaxis.set_label_position('top')
    ax[0][8].set_xlabel('HMS', fontsize=18, color='k')

    fig.supxlabel('Epochs', fontsize=18, color='k', weight='normal')
    fig.supylabel(metric, fontsize=18, color='k', weight='normal')
    fig.legend(loc=8, ncol=8, prop={'size': 20, 'weight': 'normal'})
    plt.savefig(metric + '.eps', dpi=100, bbox_inches='tight', pad_inches=.7)


if __name__ == '__main__':
    main('Accuracy')
    # main('Loss')
    main('Validation Accuracy')
    # main('Validation Loss')
