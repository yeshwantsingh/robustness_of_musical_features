import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

plt.style.use('ggplot')


def get_accuracy(model, frame, slide):
    with open('history' + str(model) + '_' + str(frame) + '_' + str(slide) + '.json') as f:
        data = json.load(f)
    return data['test_acc'] * 100


def main():
    fig = plt.figure(figsize=(5, 7), constrained_layout=True)
    ax = fig.subplots(4, 2, sharex=True, sharey=True)
    count = 0
    for model in [1, 2, 5, 6, 7, 8, 9, 10]:
        frame_vs_slide = []
        for frame in np.arange(0.5, 3.5, 0.5):
            per_frame = []
            for slide in np.arange(0.5, 3.5, 0.5):
                acc = get_accuracy(model, frame, slide)
                per_frame.append(acc)
            frame_vs_slide.append(per_frame)
        ax.flat[count].tick_params(direction='out', color='k', labelcolor='k', labelsize=8)
        sns.heatmap(frame_vs_slide[::-1], cbar=False, cmap='GnBu', annot=True,
                    fmt='0.2f', ax=ax.flat[count], square=False, annot_kws={"fontsize": 6},
                    xticklabels=['0.5', '1.0', '1.5', '2.0', '2.5', '3.0'],
                    yticklabels=['3.0', '2.5', '2.0', '1.5', '1.0', '0.5']
                    )
        if model > 2:
            model -= 2
        ax.flat[count].set_title('Model ' + str(model), fontdict={'fontsize': 10})
        count += 1

    fig.supylabel('Frame window length (sec)', size='large')
    fig.supxlabel('Sliding window length (sec)', size='large')
    plt.savefig('frame_vs_slide.eps', dpi=100, bbox_inches='tight')


if __name__ == '__main__':
    main()
