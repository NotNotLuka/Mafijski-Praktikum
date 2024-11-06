import time
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from diag import get_eigenvalues
import hamiltonian as hm
import scipy
import random

hm.cp = cp
hm.define_factorial_div()
plt.rcParams.update({"font.size": 14})


def random_colour():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def convergence():
    N = 1000
    harmonic = hm.harmonic_hamiltonian(N)

    lamdas = []
    diff = []
    for i in range(1, 10):
        anharmonic = hm.anharmonic_hamiltonian(N, i * 0.000001, 4)
        difference = cp.sum(harmonic - anharmonic)
        lamdas.append(i * 0.000001)
        diff.append(difference.get())

    plt.plot(lamdas, diff)
    plt.show()


def energies_relations():
    N = 1000
    lamda = []
    energies = []
    for i in range(0, 100):
        anharmonic = hm.anharmonic_hamiltonian(N, i * 0.01)
        eigenvalues, _ = cp.linalg.eigh(anharmonic)
        energies.append(eigenvalues)
        lamda.append(i * 0.01)

    energies_from_lambda = cp.array(energies).T.get()
    for i in range(10):
        energy = energies_from_lambda[i]
        plt.plot(
            lamda,
            energy,
            label=rf"$\lambda_{{{i}}}$",
        )
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$E/\hbar \omega$")
    plt.legend()
    plt.savefig("images/energyfromlambda.pdf")
    plt.clf()

    energies = []
    Ns = []
    anharmonic = hm.anharmonic_hamiltonian(N, 1)
    for n in range(10, 100, 1):
        eigenvalues, _ = cp.linalg.eigh(anharmonic[:n, :n])
        energies.append(eigenvalues[:10])
        Ns.append(n)
    energies = cp.array(energies).T.get()
    for i in range(10):
        plt.plot(Ns, energies[i], label=rf"$\lambda_{{{i}}}$")
    plt.xlabel("N")
    plt.ylabel(r"$E/(\hbar \omega)$")
    plt.legend()
    plt.savefig("images/energyfromN.pdf")
    plt.clf()


def time_analysis(opt=1):
    ts_householder = []
    ts_cupy = []
    ts_cupy_batch = []
    ts_numpy = []
    ts_numpy_batch = []
    Ns = []
    if opt == 1:
        N = 50
        diff = 10
    elif opt == 2:
        N = 1000
        diff = 50
    else:
        N = 2500
        diff = 100
    print(N, diff, opt)
    for i in range(1, N, diff):
        print(i)
        average_cupy = 0
        average_numpy = 0
        average_householder = 0
        lambds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        matrix_batch = []
        for lambd in lambds:
            hamiltonian = hm.anharmonic_hamiltonian(i, lambd)
            matrix_batch.append(hamiltonian)

        for matrix in matrix_batch:
            t = time.time()
            cp.linalg.eigh(matrix)
            average_cupy += time.time() - t
            matrix = matrix
            t = time.time()
            np.linalg.eigh(matrix)
            average_numpy += time.time() - t
            if opt == 1:
                t = time.time()
                get_eigenvalues(matrix)
                average_householder += time.time() - t
        ts_numpy.append(average_numpy / len(lambds))
        ts_cupy.append(average_cupy / len(lambds))
        ts_householder.append(average_householder / len(lambds))

        matrix_batch = cp.array(matrix_batch)
        if opt <= 2:
            time.sleep(0.05)
            t = time.time()
            cp.linalg.eigh(matrix_batch)
            tim = time.time() - t
            ts_cupy_batch.append(tim / len(lambds))

        matrix_batch = matrix_batch.get()
        t = time.time()
        np.linalg.eigh(matrix_batch)
        tim = time.time() - t
        ts_numpy_batch.append(tim / len(lambds))

        Ns.append(i)
    plt.plot(Ns, ts_numpy, label="numpy")
    plt.plot(Ns, ts_numpy_batch, label="numpy batch")
    plt.plot(Ns, ts_cupy, label="cupy")
    if opt <= 2:
        plt.plot(Ns, ts_cupy_batch, label="cupy batch")
    if opt == 1:
        plt.plot(Ns, ts_householder, label="householder")

    plt.legend()
    plt.xlabel("N")
    plt.ylabel("t[s]")
    if opt == 1:
        plt.savefig("images/householder.pdf")
    if opt == 2:
        plt.savefig("images/batch.pdf")
    else:
        plt.savefig("images/cuda.pdf")
    plt.clf()


def q_speed_analysis(device):
    if device == "cpu":
        hm.cp = np
    if device == "gpu":
        hm.cp = cp
    hm.define_factorial_div()

    Ns = []
    pwrs = [1, 2, 4]
    times = [[], [], []]
    for n in range(1, 500):
        Ns.append(n)
        for j in range(3):
            pwr = pwrs[j]
            t = time.time()
            hm.anharmonic_hamiltonian(n, 0.5, qpower=pwr)
            times[j].append(time.time() - t)

    for i in range(3):
        plt.plot(Ns, times[i], label=rf"$<q^{{{pwrs[i]}}}>$")
    plt.legend()
    plt.savefig(f"images/{device}.pdf")
    plt.clf()


def draw_specific(matrix, potential, N_values, wave, q, ax, split=None):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    print(eigenvalues[:10])
    eigenvectors = eigenvectors[:N_values, :]
    eigenvectors = eigenvectors.T
    wavefunc = np.matmul(eigenvectors, wave)

    ax.plot(q, potential(q), "--")
    max_y = 0
    min_y = np.inf
    if split is None:
        rng = range(9)
    elif split == 1:
        rng = range(1, 9, 2)
    elif split == 2:
        rng = range(0, 9, 2)
    for i in rng:
        colour = random_colour()
        ax.fill_between(
            q, eigenvalues[i], (eigenvalues[i] + wavefunc[i]), color=colour
        )
        ax.axhline(y=eigenvalues[i], color="black", linestyle="--")
        y = wavefunc[i] + eigenvalues[i]
        if max_y < np.max(y):
            max_y = np.max(y)
        if np.min(y) < min_y:
            min_y = np.min(y)
        ax.plot(q, y, color=colour, label=f'$E_{i}$')
    ax.set_ylim(min_y - min_y * 0.05, max_y + max_y * 0.05)


def draw_eigenvectors():
    hm.cp = np
    hm.define_factorial_div()
    N_values = 50

    def get_wavefunctions(q):
        values = []
        for n in range(N_values):
            n_factorial = scipy.special.gamma(n + 1)
            Hn = scipy.special.hermite(n)
            wave = lambda q: (
                np.power((np.power(2, n) * n_factorial * np.sqrt(np.pi)), -1 / 2)
                * np.exp(-np.power(q, 2) / 2)
            ) * Hn(q)
            values.append(wave(q))
        return np.array(values)

    q = np.linspace(-5, 5, 500)
    wave = get_wavefunctions(q)
    lambds = [0, 0.25, 0.4, 0.6, 0.75, 1]
    fig, ax = plt.subplots(3, 2, figsize=(10, 25))
    for j in range(len(lambds)):
        lambd = lambds[j]
        potential = lambda x: lambd * np.power(x, 4) + np.power(x, 2) * 1 / 2
        anharmonic = hm.anharmonic_hamiltonian(1000, lambd, qpower=4)
        draw_specific(anharmonic, potential, N_values, wave, q, ax[j // 2][j % 2])

        ax[j // 2][j % 2].set_title(rf"$\lambda={lambd}$")
    plt.savefig("images/eigenfunctions.pdf")
    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    q = np.linspace(-6, 6, 500)
    wave = get_wavefunctions(q)
    potential = lambda x: - 2 * np.power(x, 2) + 1 / 10 * np.power(x, 4)
    extra = hm.extra_hamiltonian(1000)
    draw_specific(extra, potential, N_values, wave, q, ax[0], split=1)
    draw_specific(extra, potential, N_values, wave, q, ax[1], split=2)
    ax[0].set_ylim(-10, 1)
    ax[1].set_ylim(-10, 1)
    ax[0].set_xlim(-6, 6)
    ax[1].set_xlim(-6, 6)
    ax[0].legend()
    ax[1].legend()
    plt.savefig("images/extra.pdf")
    plt.clf()


print("q speed analysis")
q_speed_analysis("cpu")
q_speed_analysis("gpu")
print("time analysis")
time_analysis(1)
time_analysis(2)
time_analysis(3)
print("energies")
energies_relations()
print("draw eigen vectors")
draw_eigenvectors()
