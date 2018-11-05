from typing import Tuple
from scipy.io import wavfile
from skimage import util
from scipy import signal

import pyfftw
import numpy as np
import matplotlib.pyplot as plt
import cupy
import time
import re
import os


def get_audio_and_rate(filepath: str) -> Tuple[int, np.ndarray]:
    rate, audio = wavfile.read(filepath)
    return rate, np.mean(audio, axis=1)


def get_audio_slices(audio: np.ndarray, window_size: int, step: int) -> np.ndarray:
    slices = util.view_as_windows(audio, window_shape=(window_size,), step=step)
    hanning_window = np.hanning(window_size + 1)
    win = hanning_window[:-1]
    slices = (slices * win).T
    return slices


def get_normalized_spectrum(spectrum: np.ndarray, window_size: int) -> np.ndarray:
    cut_freq = spectrum[:window_size // 2 + 1:-1]
    abs_spectrum = np.abs(cut_freq)
    return 20 * np.log10(abs_spectrum / np.max(abs_spectrum))


def get_fftw_spectrum(audio: np.ndarray, window_size: int, step: int) -> np.ndarray:
    fft_start = time.time()
    slices = get_audio_slices(audio, window_size, step)
    slices_fftw = pyfftw.empty_aligned(slices.shape, dtype='float64')
    slices_fftw[:] = slices
    fft = pyfftw.interfaces.numpy_fft.fft(slices_fftw, axis=0)
    normalized_spectrum = get_normalized_spectrum(fft, window_size)
    print(f'get_fftw_spectrum - calculation time: {time.time()-fft_start:.5f} s')
    return normalized_spectrum


def get_cuda_spectrum(audio: np.ndarray, window_size: int, step: int) -> np.ndarray:
    fft_start = time.time()
    slices = get_audio_slices(audio, window_size, step)
    cuda_spectrum = cupy.asnumpy(cupy.fft.fft(cupy.array(slices, dtype='float64'), axis=0))
    cupy.cuda.Device().synchronize()
    normalized_spectrum = get_normalized_spectrum(cuda_spectrum, window_size)
    print(f'get_cuda_spectrum - calculation time: {time.time()-fft_start:.5f} s')
    return normalized_spectrum


def plot_spectrum(spectrum: np.ndarray, audio_length: float, rate: int, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        spectrum,
        origin='lower',
        extent=(0, audio_length, 0, rate / 2 / 1000)
    )
    ax.axis('tight')
    ax.set_ylabel('Frequency, kHz')
    ax.set_xlabel('Time, s')
    plt.title(title)
    filename = re.sub(r'[/.,]', ' ', title).replace(" ", "_")
    fig.savefig(f'plots/{filename}')
    plt.show()


def etalon_plot(audio: np.ndarray, rate: int, title: str, window_size: int = 512, step: int = 100):
    start = time.time()
    freqs, times, Sx = signal.spectrogram(audio, fs=rate, window='hanning',
                                          nperseg=window_size, noverlap=window_size - step,
                                          detrend=False, scaling='spectrum', mode='magnitude')
    print(f'signal.spectrogram - calculation time: {time.time()-start:.5f} s')
    f, ax = plt.subplots(figsize=(10, 5))
    ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx))
    ax.set_ylabel('Frequency, kHz')
    ax.set_xlabel('Time, s')
    plt.title(title)
    plt.show()


def calculate_plots(filepath: str, window_size: int = 512, step: int = 100):
    print(f'-----{filepath}, calculation started..')
    rate, audio = get_audio_and_rate(filepath)
    audio_duration = audio.shape[0] / rate
    print(f'Audio - duration: {audio_duration:.2f} seconds, rate: {rate} Hz')
    fftw_spectrum, cuda_spectrum = get_fftw_spectrum(audio, window_size, step), get_cuda_spectrum(audio, window_size,
                                                                                                  step)
    np.testing.assert_allclose(fftw_spectrum, cuda_spectrum)
    plot_spectrum(fftw_spectrum, audio_duration, rate, f'{filepath} FFTW Spectrogram, {window_size} window Size')
    plot_spectrum(cuda_spectrum, audio_duration, rate, f'{filepath} CUFFT Spectrogram, {window_size} window Size')
    etalon_plot(audio, rate, f'{filepath} \'scipy.signal\' Spectrogram, {window_size} window Size', window_size, step)
    print(f'-----{filepath}, calculation finished!')


def get_wavs(folder: str = './data'):
    wavs = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                wavs.append(file)
    return wavs


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    wavs_folder = './data'
    for wav in get_wavs(wavs_folder):
        calculate_plots(f'{wavs_folder}/{wav}', 512)
    plt.rcParams = plt.rcParamsDefault
