import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('classic')

def main():
    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
    carnatic = [
        [ 127,  305,  458,  607, 1355,  575,  195,  503],
        [  25,   67,   70,   79,  249,   88,   36,   84],
        [  24,   56,   64,   74,  154,   77,   25,   76],
        [  35,   58,   77,  104,  198,  115,   33,  126],
        [ 118,  288,  428,  568, 1395,  597,  188,  463],
        [ 117,  256,  432,  563, 1447,  576,  178,  526],
        [  52,   75,   88,  120,  287,  149,   53,  160],
        [ 118,  267,  455,  572, 1458,  581,  194,  513],
        [ 115,  238,  468,  592, 1434,  617,  183,  515]
    ]

    gtzan = [
        [  8,  15,  29,  38,  91,  37,  13,  33],
        [  1,   3,   4,   4,  14,   5,   1,   4],
        [  2,   2,   5,   7,  10,   8,   1,  12],
        [  1,   3,   4,   5,  11,   5,   2,   6],
        [  9,  20,  31,  41, 133,  45,  14,  35],
        [  8,  18,  31,  42, 110,  45,  14,  38],
        [  8,  13,  29,  34,  48,  42,   6,  68],
        [  6,  14,  29,  40,  86,  44,   9,  34],
        [  8,  17,  33,  42, 110,  43,  14,  38]
    ]

    hindustani = [
        [ 219,  484,  793, 1125, 2763, 1083,  352, 1031],
        [  43,  129,  144,  181,  476,  198,   64,  167],
        [  49,  107,  119,  159,  336,  170,   49,  169],
        [  45,   99,  128,  156,  333,  167,   53,  189],
        [ 227,  564,  809, 1092, 2711, 1165,  367, 1013],
        [ 233,  427,  858, 1122, 2847, 1167,  362, 1091],
        [  85,   93,  131,  117,  155,  340,  249,  290],
        [ 219,  523,  775, 1035, 2662, 1099,  350, 1002],
        [ 199,  429,  755, 1053, 2571, 1049,  333,  948]
    ]

    homburg = [
        [ 11,  25,  39,  50, 118,  53,  14,  42],
        [  2,   7,  11,  14,  21,  18,   4,  13],
        [  5,   9,   9,  12,  39,  15,   4,  12],
        [  5,   9,   9,  12,  38,  15,   4,  12],
        [ 14,  32,  46,  56, 127,  68,  18,  51],
        [ 13,  28,  46,  56, 128,  69,  18,  51],
        [  9,  12,  14,  19,  42,  21,   9,  22],
        [ 13,  29,  45,  57, 158,  66,  19,  49],
        [ 13,  28,  35,  45, 116,  45,  17,  54]
    ]


    x = np.arange(len(labels))  # the label locations
    y = np.arange(0, 2750, 250)  # the label locations
    width = 0.08  # the width of the bars

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    ax = fig.subplots(4, 1, sharex=True)

    # Carnatic

    rects1 = ax[0].bar(x - 4 * width, carnatic[0], width)
    rects2 = ax[0].bar(x - 3 * width, carnatic[1], width)
    rects3 = ax[0].bar(x - 2 * width, carnatic[2], width)
    rects4 = ax[0].bar(x - width, carnatic[3], width)
    rects5 = ax[0].bar(x, carnatic[4], width)
    rects6 = ax[0].bar(x + width, carnatic[5], width)
    rects7 = ax[0].bar(x + 2 * width, carnatic[6], width)
    rects8 = ax[0].bar(x + 3 * width, carnatic[7], width)
    rects9 = ax[0].bar(x + 4 * width, carnatic[8], width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].yaxis.set_label_position('right')
    ax[0].set_ylabel('Carnatic', fontsize=18, color='k')
    ax[0].set_xticks(x)
    ax[0].set_yticks(np.arange(0, 1750, 250))
    ax[0].grid(True)
    ax[0].set_xticklabels(labels)
    ax[0].tick_params(color='k', labelcolor='k', labelsize=12)

    # Gtzan
    rects1 = ax[1].bar(x - 4 * width, gtzan[0], width)
    rects2 = ax[1].bar(x - 3 * width, gtzan[1], width)
    rects3 = ax[1].bar(x - 2 * width, gtzan[2], width)
    rects4 = ax[1].bar(x - width, gtzan[3], width)
    rects5 = ax[1].bar(x, gtzan[4], width)
    rects6 = ax[1].bar(x + width, gtzan[5], width)
    rects7 = ax[1].bar(x + 2 * width, gtzan[6], width)
    rects8 = ax[1].bar(x + 3 * width, gtzan[7], width)
    rects9 = ax[1].bar(x + 4 * width, gtzan[8], width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1].yaxis.set_label_position('right')
    ax[1].set_ylabel('Gtzan', fontsize=18, color='k')
    ax[1].set_xticks(x)
    ax[1].set_yticks(np.arange(0, 165, 15))
    ax[1].grid(True)
    ax[1].set_xticklabels(labels)
    ax[1].tick_params(color='k', labelcolor='k', labelsize=12)

    # Hindustani
    rects1 = ax[2].bar(x - 4 * width, hindustani[0], width)
    rects2 = ax[2].bar(x - 3 * width, hindustani[1], width)
    rects3 = ax[2].bar(x - 2 * width, hindustani[2], width)
    rects4 = ax[2].bar(x - width, hindustani[3], width)
    rects5 = ax[2].bar(x, hindustani[4], width)
    rects6 = ax[2].bar(x + width, hindustani[5], width)
    rects7 = ax[2].bar(x + 2 * width, hindustani[6], width)
    rects8 = ax[2].bar(x + 3 * width, hindustani[7], width)
    rects9 = ax[2].bar(x + 4 * width, hindustani[8], width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[2].yaxis.set_label_position('right')
    ax[2].set_ylabel('Hindustani', fontsize=18, color='k')
    ax[2].set_xticks(x)
    ax[2].set_yticks(np.arange(0, 3250, 250))
    ax[2].grid(True)
    ax[2].set_xticklabels(labels)
    ax[2].tick_params(color='k', labelcolor='k', labelsize=12)

    # Homburg
    rects1 = ax[3].bar(x - 4 * width, homburg[0], width, label='Chroma')
    rects2 = ax[3].bar(x - 3 * width, homburg[1], width, label='CQT')
    rects3 = ax[3].bar(x - 2 * width, homburg[2], width, label='HMS')
    rects4 = ax[3].bar(x - width, homburg[3], width, label='MSpec')
    rects5 = ax[3].bar(x, homburg[4], width, label='MFCC')
    rects6 = ax[3].bar(x + width, homburg[5], width, label='SC')
    rects7 = ax[3].bar(x + 2 * width, homburg[6], width, label='STFT')
    rects8 = ax[3].bar(x + 3 * width, homburg[7], width, label='SGram')
    rects9 = ax[3].bar(x + 4 * width, homburg[8], width, label='Tonnetz')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[3].yaxis.set_label_position('right')
    ax[3].set_ylabel('Homburg', fontsize=21, color='k', weight='normal')
    ax[3].set_yticks(np.arange(0, 180, 15))
    ax[3].set_xticks(x)
    ax[3].grid(True)
    ax[3].set_xticklabels(labels, fontsize=18, color='k', weight='normal')
    ax[3].tick_params(color='k', labelcolor='k', labelsize=12)

    fig.supxlabel('Models', fontsize=18, color='k', weight='normal')
    fig.supylabel('Time (sec)', fontsize=18, color='k', weight='normal')
    fig.legend(loc=8, ncol=9, prop={'size': 12, 'weight': 'normal'})
    plt.savefig('execution_time_plot.eps', dpi=100, bbox_inches='tight', pad_inches=.5)


if __name__ == '__main__':
    main()
