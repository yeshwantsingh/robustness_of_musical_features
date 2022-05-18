import numpy as np
import json
from matplotlib import pyplot as plt
plt.style.use('bmh')


def plot(ax, history, dataset, metric):
    if metric == 'Accuracy':
        keys = ['acc', 'val_acc']
    elif metric == 'Loss':
        keys = ['loss', 'val_loss']

    epochs = np.arange(1, 101)
    for feature in range(0, 4):
        for key in keys:
            ax[feature].plot(epochs, history[feature][0][key][:100], label=key)
            if feature == 3:
                ax[feature].legend()
            ax[feature].grid(True)



def main(metric):

    datasets = ['carnatic', 'hindustani']

    features = ['chromagram', 'spectral_contrast', 'swaragram', 'tonnetz']
    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    ax = fig.subplots(2, 4, sharey=True, sharex=True)
    history = []
    for dataset in datasets:
        hist = []
        for feature in features:
            h = []
            for i in [6]:
                with open(dataset + '/' + feature + '/history' + str(i) + '.json') as f:
                    h.append(json.load(f))
            hist.append(h)
        history.append(hist)

    plot(ax[0], history[0], 'Carnatic', metric)
    plot(ax[1], history[1], 'Hindustani', metric)

    ylabel = ''
    if metric == 'Accuracy' or metric == 'Validation Accuracy':
        ylabel = 'Accuracy'
    else:
        ylabel = 'Loss'


    title = 'Training'
    if metric.split(' ')[0] == 'Validation':
        title = 'Validation'

    #ax[0][0].set_ylabel(ylabel)
    ax[0][3].yaxis.set_label_position('right')
    ax[0][3].set_ylabel('Carnatic')

    #ax[2][0].set_ylabel(ylabel)
    ax[1][3].yaxis.set_label_position('right')
    ax[1][3].set_ylabel('Hindustani')

    ax[0][0].xaxis.set_label_position('top')
    ax[0][0].set_xlabel('Chromagram')

    ax[0][1].xaxis.set_label_position('top')
    ax[0][1].set_xlabel('Spectral Contrast')

    ax[0][2].xaxis.set_label_position('top')
    ax[0][2].set_xlabel('Swaragram')

    ax[0][3].xaxis.set_label_position('top')
    ax[0][3].set_xlabel('Tonnetz')

    #ax[3][0].set_xlabel('Epochs')
    #ax[3][1].set_xlabel('Epochs')
    #ax[3][2].set_xlabel('Epochs')
    #ax[3][3].set_xlabel('Epochs')
    #ax[3][4].set_xlabel('Epochs')
    ax[1][0].set_xlabel('.', color=(0,0,0,0))
    ax[1][0].set_ylabel('.', color=(0,0,0,0))

    #plt.suptitle(title)
    #plt.xlabel('Epochs')
    #plt.ylabel(metric)
    fig.text(0.5, 0.01, 'Epochs', ha='center', size='x-large')
    fig.text(0.001, 0.5, metric, va='center', rotation='vertical', size='x-large')

    plt.savefig(metric + '.svg')


if __name__ == '__main__':
    main('Accuracy')
    main('Loss')


