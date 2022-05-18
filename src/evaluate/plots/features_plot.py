import os
import sys
import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import seaborn as sns

# sns.set(font_scale=1.1)
plt.style.use('ggplot')


sys.path.append(os.path.join('../', '../'))

from extract_features.swaragram import utils, core


def extract_swara(x, sr=22050):
    tonic = utils.getTonic(x)
    stft = utils.computeSTFT(x, sr)
    refPitch, refFreq = utils.computeRefPitch(tonic)

    ret = core.computeSwaragram(stft, refPitch=refPitch, refFreq=refFreq)

    return (23 * np.log10(2.220446049250313e-16 + ret))

def main():
    y, sr = librosa.load('Raga_Bhopali.mp3', sr=22050, duration=30)

    # Plots
    fig = plt.figure(figsize=(9, 13), constrained_layout=True)
    ax = fig.subplots(5, 2)

    # Chromagram
    chroma = librosa.feature.chroma_stft(y, sr)
    img1 = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[0][0])
    fig.colorbar(img1, ax=ax[0][0], pad=-0.07)
    ax[0][0].set(title='Chromagram')

    # MFCC
    mfcc = librosa.feature.mfcc(y, sr)
    img2 = librosa.display.specshow(mfcc, x_axis='time', ax=ax[0][1])
    fig.colorbar(img2, ax=ax[0][1])
    ax[0][1].set(title='MFCC', ylabel='Mel bands')

    # SC
    sc = librosa.feature.spectral_contrast(y, sr)
    img3 = librosa.display.specshow(sc, x_axis='time', ax=ax[1][0])
    fig.colorbar(img3, ax=ax[1][0], pad=-0.07)
    ax[1][0].set(title='Spectral Contrast', ylabel='Frequency bands')

    # Swaragram
    swara = extract_swara(y, sr)
    stft = np.abs(librosa.stft(y, sr))
    t = librosa.frames_to_time(np.arange(stft.shape[1]), n_fft=2048)

    left = min(t)
    right = max(t)
    lower = 0
    upper = 12

    chroma_label = ['S', 'r', 'R', 'g', 'G', 'M', "M'", 'P', 'd', 'D', 'n', 'N']
    img4 = ax[1][1].imshow(swara, origin='lower', aspect='auto', extent=[0, 30, 0, 12], cmap='inferno')
    img4.set_clim([0, 100])
    ax[1][1].set_xlabel('Time')
    ax[1][1].set(title='Swaragram', ylabel='Swara')
    ax[1][1].grid(False)
    fig.colorbar(img4, ax=ax[1][1])
    ax[1][1].set_yticks(np.arange(12)+0.5)
    ax[1][1].set_yticklabels(chroma_label)


    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y, sr)
    img5 = librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time', ax=ax[2][0])
    fig.colorbar(img5, ax= ax[2][0], pad=-0.07)
    ax[2][0].set(title='Tonnetz', ylabel='Harmonics')

    # STFT
    img6 = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax[2][1])
    ax[2][1].set(title='STFT', ylabel='Frequency (Hz)')
    fig.colorbar(img6, ax=ax[2][1], format="%+2.0f dB")

    # CQT
    cqt = np.abs(librosa.cqt(y, sr))
    img7 = librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max),
                                   sr=sr, x_axis='time', y_axis='log', ax=ax[3][0])
    ax[3][0].set(title='CQT', ylabel='Frequency (Hz)')
    fig.colorbar(img7, ax=ax[3][0], format="%+2.0fdB", pad=-0.07)

    # MSpec
    mspec = librosa.feature.melspectrogram(y, sr)
    S_dB = librosa.power_to_db(mspec, ref=np.max)
    img8 = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax[3][1])
    fig.colorbar(img8, ax=ax[3][1], format='%+2.0f dB')
    ax[3][1].set(title='Mel Spectrogram', ylabel='Frequency (Hz)')

    # HMS
    gs = ax[4][0].get_gridspec()
    ax[4][0].remove()
    ax[4][1].remove()
    baxis = fig.add_subplot(gs[4, :])
    D = librosa.stft(y, sr)
    HMS, _ = librosa.decompose.hpss(mspec)
    img9 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(HMS),
                                                     ref=np.max(np.abs(D))),
                             y_axis='log', x_axis='time', ax=baxis)

    fig.colorbar(img9, ax=baxis, format='%+2.0f dB', pad=0.02)
    baxis.set(title='Harmonic separated Mel spectrogram', ylabel='Frequency (Hz)')

    #ax[4][1].axis('off')
    
    plt.savefig('features.eps', dpi=100, bbox_inches='tight', pad_inches=.01)


if __name__ == '__main__':
    main()
