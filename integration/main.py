import matplotlib.pyplot as plt
import time
import numpy as np
from diffeq_tsint import euler, heun, rk2a, rk2b, rk45, rku4, pc4


plt.rcParams.update({"font.size": 14})


def euler_graph():
    T0s = [21, -15]
    steps = [0.1, 1, 3, 6, 12, 16]

    T_zun = -5
    k = 0.1

    for T_ind in range(len(T0s)):
        T_0 = T0s[T_ind]
        for s_ind in range(len(steps)):
            step = steps[s_ind]
            f = lambda T, t: -k * (T - T_zun)

            t = np.arange(0, 100, step)
            Ts = euler(f, T_0, t)
            plt.plot(t, Ts, label=f"Step length {steps[s_ind]}")
        plt.legend()
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("T")
        plt.savefig(rf"images/euler{T_0}.pdf")
        plt.close()


def different_methods():
    methods = [euler, heun, rk2a, rk2b, rk45, rku4, pc4]
    T_zun = np.float64(-5)
    k = np.float64(0.1)
    T_0 = np.float64(21)
    f = lambda T, t: -k * (T - T_zun)

    theory = lambda t: T_zun + np.exp(-k * t) * (T_0 - T_zun)

    for method in methods:
        steps = []
        sws = []
        diffs = []
        for step in range(1, 10_000, 50):
            step = step * 1e-3
            steps.append(step)
            t = np.arange(0, 500, step, dtype=np.float64)

            theoretical = theory(t)
            T = method(f, T_0, t)
            diffs.append(np.max(np.abs(theoretical - T)))

        plt.plot(steps, diffs, label=method.__name__)

    plt.legend()
    plt.yscale("log")
    plt.xlabel("step")
    plt.grid()
    plt.savefig("images/precision.pdf")
    plt.close()

    for method in methods:
        steps = []
        sws = []
        diffs = []
        for step in range(1, 1000, 50):
            step = step * 1e-3
            steps.append(step)
            N = 5
            sw = 0
            t = np.arange(0, 500, step, dtype=np.float64)
            for i in range(N):
                t0 = time.time()
                T = method(f, T_0, t)
                sw += time.time() - t0
            sws.append(sw / N)
        plt.plot(steps, sws, label=method.__name__)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.savefig("images/time.pdf")
    plt.close()


def we_love_families(T_0, extra=0):
    T_zun = np.float64(-5)
    T_0 = np.float64(T_0)

    data = []
    kk = []
    t = np.arange(0, 100, 0.1, dtype=np.float64)
    for val in range(1, 30_000, 50):
        if extra == 0:
            k = np.float64(val * 1e-5)
            f = lambda T, t: -k * (T - T_zun)
        else:
            if extra == 1:
                k = np.float64(val * 1e-5)
                A = 1
            if extra == 2:
                A = np.float64(val * 1e-4)
                k = 0.1
            delta = 10
            f = lambda T, t: -k * (T - T_zun) + A * np.sin(2 * np.pi / 24 * (t - delta))
        T = rku4(f, T_0, t)
        data.append(T)
        if extra == 1 or extra == 0:
            kk.append(k)
        if extra == 2:
            kk.append(A)

    data_array = np.array(data)[::-1]

    vmin = -25 if extra == 2 else -15
    plt.imshow(
        data_array,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=21,
        aspect="auto",
        extent=[min(t), max(t), min(kk), max(kk)],
    )
    plt.colorbar(label="T")
    plt.xlabel("t")
    if extra == 1 or extra == 0:
        plt.ylabel("k")
    if extra == 2:
        plt.ylabel("A")
    plt.savefig(f"images/family{T_0}{extra}.pdf")
    plt.close()


euler_graph()
different_methods()
for opt in range(3):
    we_love_families(21, opt)
    we_love_families(-15, opt)
