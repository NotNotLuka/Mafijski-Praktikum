import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft

plt.rcParams.update({"font.size": 16})


def autocorrelate(signal):
    mean = np.mean(signal)
    signal = signal - mean

    fft_signal = fft(signal)
    magnitude = np.power(np.abs(fft_signal), 2)
    ifft_signal = ifft(magnitude)

    N = signal.size
    var = np.var(signal)
    atc = ifft_signal.real / (var * N)

    ind = len(atc) // 2
    atc = np.concatenate((atc[ind:], atc[:ind]))
    return atc


def correlate(g, h):
    g_mean, h_mean = np.mean(g), np.mean(h)
    g, h = g - g_mean, h - h_mean

    fft_g, fft_h = fft(g), fft(h)
    magnitude = fft_g * np.conj(fft_h)
    ifft_signal = ifft(magnitude)

    N = h.size

    atc = ifft_signal.real / N

    ind = len(atc) // 2
    atc = np.concatenate((atc[ind:], atc[:ind]))
    return atc / len(atc)


def examples():
    sampling = 10000
    t = np.linspace(-10, 10, sampling)
    signal = np.cos(2 * np.pi * t)
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = signal + noise

    autocorr = autocorrelate(noisy_signal)

    plt.xticks([])
    plt.plot(t, noisy_signal, label=r"$\cos{2\pi x}$")
    plt.plot(t, autocorr, label="autokoreliran signal")
    plt.grid()
    plt.legend()
    plt.savefig("images/cosine.pdf")
    plt.close()

    sampling = 10000
    t = np.linspace(-5, 5, sampling)
    signal = np.sin(2 * np.pi * t)
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = signal + noise

    autocorr = autocorrelate(noisy_signal)

    plt.xticks([])
    plt.plot(t, noisy_signal, label=r"$\sin{2\pi x}$")
    plt.plot(t, autocorr, label="autokoreliran signal")
    plt.grid()
    plt.legend()
    plt.savefig("images/sine.pdf")
    plt.close()


def get_autocorrelations():
    filenames = [
        "bubomono",
        "bubo2mono",
        "mix",
        "mix1",
        "mix2",
        "mix22",
    ]
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    for i in range(len(filenames)):
        fn = filenames[i]
        data = np.loadtxt(f"data/{fn}.txt")
        atc = autocorrelate(data)
        ax[i // 2][i % 2].plot(atc)
        title = "Borut" if i == 0 else "Paul" if i == 1 else fn
        ax[i // 2][i % 2].set_title(title)
        ax[i // 2][i % 2].grid()
        ax[i // 2][i % 2].set_xticks([])
    plt.savefig("images/autocor1.pdf")
    plt.close()

    rest = ["mix2", "mix22"]

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for i in range(len(rest)):
        fn = rest[i]
        data = np.loadtxt(f"data/{fn}.txt")
        atc = data
        for _ in range(2):
            atc = autocorrelate(atc)
        ax[i].plot(atc)
        ax[i].set_title(fn)
        ax[i].grid()
        ax[i].set_xticks([])
    plt.savefig("images/autocor2.pdf")
    plt.close()


def compare_last_one():
    owl1 = np.loadtxt("data/bubomono.txt")
    owl2 = np.loadtxt("data/bubo2mono.txt")
    unknown = np.loadtxt("data/mix22.txt")

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    plt.xticks([])
    data = correlate(unknown, owl1)
    ax[0].plot(data)
    ax[0].grid()
    ax[0].set_title("Korelacija s sovo Borut")
    data = correlate(unknown[1:], owl2)
    ax[1].plot(data)
    ax[1].grid()
    ax[1].set_title("Korelacija s sovo Paul")
    plt.savefig("images/autocorlast.pdf")


def noaa():
    rate, data = wavfile.read("data/satellite2.wav")
    atc = autocorrelate(data)
    out = np.float32(atc)
    wavfile.write("data/output.wav", rate, out)


examples()
get_autocorrelations()
compare_last_one()
noaa()
