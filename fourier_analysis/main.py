import numpy as np
import time
import cupy as cp
import matplotlib.pyplot as plt
from dft_example import DFT_slow, DFT_simplest


def mirror(signal, mask):
    signal = np.concatenate((signal[~mask], signal[mask]))
    return signal


def gaussian_ft():
    fig, ax = plt.subplots(1)
    sample_rate = 6000
    t = np.linspace(-5, 5, sample_rate, endpoint=False)
    for a in [1000, 2000, 3000]:
        signal = np.exp(-a * np.power(t, 2))
        mask = t < 0

        r = np.fft.fft(signal)
        r = mirror(r, mask)
        y = np.abs(r)
        x = np.linspace(-sample_rate / 2, sample_rate / 2, sample_rate, endpoint=False)
        ax.scatter(x, y, label=rf"a={a}")

    plt.legend()
    plt.xlabel(r"$\nu$")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig("images/gauss.pdf")
    plt.clf()


def sines_ft():
    sample_rate = 2000
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = np.zeros(t.shape)
    for f in range(0, 1000, 100):
        signal += f / 100 * np.sin(2 * np.pi * f * t)

    plt.plot(t[:200], signal[:200], color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig("images/sinefunc.pdf")
    plt.clf()

    r = DFT_simplest(signal)

    x = np.linspace(0, sample_rate / 2, sample_rate // 2, endpoint=False)
    y = 2 * np.abs(r[: sample_rate // 2]) / sample_rate
    plt.plot(x, y, color="blue")

    plt.xlabel(r"$\nu$")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig("images/sines.pdf")
    plt.clf()

    y = np.fft.ifft(r)
    plt.plot(t[:200], y[:200], color="green")
    plt.grid("True")
    plt.show()
    plt.clf()

    print(sum(signal - y))


def delta_func():
    sample_rate = 200
    delta = np.zeros(sample_rate)
    delta[10] = 1
    ft = DFT_slow(delta)

    plt.plot(ft, color="blue")
    plt.grid(True)
    plt.xticks([])
    plt.savefig("images/deltasingle.pdf")
    plt.clf()

    delta[20] = 2
    delta[15] = 1
    ft = DFT_slow(delta)
    plt.plot(ft, color="blue")
    plt.xticks([])
    plt.grid(True)
    plt.savefig("images/delta.pdf")
    plt.clf()


def nyquist():
    signal_fun = lambda t: np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 100 * t)

    def helper(sample_rate, ax, color):
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        signal = signal_fun(t)

        lbl = "high" if color == "green" else "low"
        ax[0].scatter(t, signal, color=color, label=f"{lbl} sampling")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("Amplitude")

        freq = np.fft.fft(signal)
        x = np.linspace(0, sample_rate / 2, sample_rate // 2)

        ax[1].plot(
            x,
            2 * np.abs(freq[: sample_rate // 2]) / sample_rate,
            label=f"{lbl} sampling",
            color=color,
        )
        ax[1].set_xlabel(r"$\nu$")
        ax[1].set_ylabel("Amplitude")

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    smoother = np.linspace(0, 0.5, 1000)
    ax[0].plot(smoother, signal_fun(smoother), label="original function", color="blue")

    helper(71, ax, "red")
    helper(800, ax, "green")

    ax[0].set_xlim(0, 0.25)
    ax[0].grid(True)
    ax[0].legend()
    ax[1].set_xlim(0, 120)
    ax[1].grid(True)
    ax[1].legend()

    plt.savefig("images/nyquist.pdf")
    plt.clf()


def time_analysis():
    signal_fun = lambda t: np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 100 * t)
    ts = [[], [], [], []]
    funs = [np.fft.fft, cp.fft.fft, DFT_simplest, DFT_slow]
    names = ["numpy FFT", "cupy FFT", "DFT simplest", "DFT slow"]
    x = []
    for sample_rate in range(100, 1000, 100):
        x.append(sample_rate)
        print(sample_rate)
        t = np.linspace(0, 0.5, sample_rate)
        signal = signal_fun(t)
        for i in range(len(funs)):
            if i == 1:
                signal = cp.array(signal)
            tmp = 0
            n = 0
            for j in range(10 if i < 2 else 1):
                n += 1
                t0 = time.time()
                funs[i](signal)
                tmp += time.time() - t0
            ts[i].append(tmp / n)
            if i == 1:
                signal = signal.get()
    for i in range(len(funs)):
        plt.plot(x, ts[i], label=names[i])
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel("N samples")
    plt.ylabel("t[s]")
    plt.legend()
    plt.savefig("images/dtf.pdf")
    plt.clf()


def bach():
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    samplings = [882, 1378, 2756, 5512, 11025, 44100]
    for i in range(len(samplings)):
        sample_rate = samplings[i]
        signal = np.loadtxt(f"recordings/Bach.{sample_rate}.txt")
        T = signal.size / sample_rate
        y = np.fft.fft(signal)
        x = np.fft.fftfreq(len(signal), d=1 / sample_rate)
        ax[i // 2][i % 2].plot(x, 2 * np.abs(y) / (T * sample_rate), color="blue")
        ax[i // 2][i % 2].set_title(fr'$\nu={sample_rate}Hz$')
    ax[2][0].set_xlim(-4000, 4000)
    ax[2][1].set_xlim(-4000, 4000)
    plt.savefig("images/bach.pdf")
    plt.clf()



# delta_func()
# gaussian_ft()
# sines_ft()
# nyquist()
# time_analysis()
bach()
