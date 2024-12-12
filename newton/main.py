import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk
from diffeq_tsint import euler, heun, rk2a, rk2b, rk45, rku4, pc4, leapfrog
import random, time


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 24,
    "text.latex.preamble": "\n".join([
        r"\usepackage{siunitx}"
    ])
})
METHODS = [euler, heun, rk45, rku4, pc4]


def random_colour():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def start_graph():
    dxdt = lambda x, _: np.array([x[1], - np.sin(x[0])])
    E = lambda x, v: 1 - np.cos(x) + np.power(v, 2) / 2
    x0 = 1
    v0 = 0
    E0 = E(x0, v0)

    t = np.arange(0, 35, 0.1)
    integrated = euler(dxdt, np.array([x0, v0]), t)

    x = integrated[:, 0]
    v = integrated[:, 1]

    energy_euler = E(x, v)

    plt.rcParams["font.size"] = 24
    plt.plot(t, x, label="pozicija")
    plt.plot(t, v, label="hitrost")
    plt.legend()
    plt.grid()
    plt.savefig("images/euler.pdf")
    plt.close()

    integrated = rk45(dxdt, np.array([x0, v0]), t)
    x = integrated[:, 0]
    v = integrated[:, 1]
    energy_rk45 = E(x, v)

    plt.plot(t, x, label="pozicija")
    plt.plot(t, v, label="hitrost")
    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/rk45.pdf")
    plt.close()

    plt.rcParams["font.size"] = 17
    mask = t < 25
    plt.plot(t[mask], energy_euler[mask], label="Euler")
    plt.plot(t[mask], energy_rk45[mask], label="RK45")
    plt.axhline(y=E0, color="black", linestyle="--", label="$E_0$")

    plt.grid()
    plt.legend()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/euler-rk45.pdf")
    plt.close()


def method_comparison():
    plt.rcParams["font.size"] = 24
    methods = METHODS

    E = lambda x, v: 1 - np.cos(x) + np.power(v, 2) / 2
    dxdt = lambda x, _: np.array([x[1], - np.sin(x[0])])
    x0 = 1
    v0 = 0

    E0 = E(x0, v0)
    t = np.arange(0, 35, 0.1)
    times = []
    for method in methods:
        t0 = time.time()
        integrated = method(dxdt, np.array([x0, v0]), t)
        times.append(time.time() - t0)
        x = integrated[:, 0]
        v = integrated[:, 1]
        E_diff = E(x, v) - E0
        plt.plot(t, E_diff, label=f'{method.__name__}')

    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$t[\si{\second}]$")
    plt.ylabel(r"$E - E_0$")
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/diffenergy.pdf")
    plt.close()

    plt.bar([x.__name__ for x in methods], times, width=0.5, color="teal", edgecolor="black")
    plt.yscale("log")
    plt.ylabel(r"$t[\si{\second}]$")
    plt.grid()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/difftime.pdf")
    plt.close()


def leapfrog_show():
    plt.rcParams["font.size"] = 17

    E = lambda x, v: 1 - np.cos(x) + np.power(v, 2) / 2
    dxdt = lambda x, _: np.array([x[1], - np.sin(x[0])])
    x0 = 1
    v0 = 0

    E0 = E(x0, v0)
    t = np.arange(0, 35, 0.1)
    t0 = time.time()
    integrated = leapfrog(dxdt, np.array([x0, v0]), t)
    print("Leapfrog time:", time.time() - t0)
    x = integrated[:, 0]
    v = integrated[:, 1]
    E_diff = E(x, v) - E0

    plt.plot(t, E_diff)
    plt.grid()
    plt.xlabel(r"$t[\si{\second}]$")
    plt.ylabel(r"$E - E_0$")
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/leapfrog.pdf")
    plt.close()


def phase_space():
    plt.rcParams["font.size"] = 17

    dxdt = lambda x, _: np.array([x[1], - np.sin(x[0])])
    for v in range(0, int(np.pi * 3 * 100), 1):
        x0 = v * 1e-2
        v0 = 0

        t0 = 4 * ellipk(np.power(np.sin(x0 / 2), 2))

        t = np.arange(0, t0, t0 / 1000)
        integrated = rk45(dxdt, np.array([x0, v0]), t)
        x = integrated[:, 0]
        v = integrated[:, 1]
        plt.plot(x, v, linewidth=3)
    plt.grid()
    plt.xlabel(r"$\varphi$")
    plt.ylabel(r"$v$")
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/phase.pdf")
    plt.close()


start_graph()
method_comparison()
leapfrog_show()
phase_space()
