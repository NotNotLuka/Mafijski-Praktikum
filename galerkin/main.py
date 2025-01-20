import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 24,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def draw(result, x, ts, name=""):
    plot = plt.imshow(
        result.T[::1, ::1],
        aspect="auto",
        cmap=cm.coolwarm,
        extent=[min(ts), max(ts), min(x), max(x)],
    )
    plt.xlabel("t")
    plt.ylabel("x")

    plt.colorbar(plot)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    if name != "":
        plt.savefig(f"images/{name}2d.pdf")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    T, X = np.meshgrid(ts, x)
    surface = ax.plot_surface(
        T, X, result.T[::-1, ::-1], cmap=cm.coolwarm, shade=True, antialiased=True
    )
    ax.set_xlabel("t")
    ax.set_ylabel("X")
    ax.set_zlabel("T")

    fig.colorbar(surface)

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    if name != "":
        plt.savefig(f"images/{name}3d.pdf")
    plt.show()
    plt.close()


def construct(N, M):
    n_og = np.arange(1, N + 1, 1)
    n = n_og[:, np.newaxis]
    n_ = n_og[np.newaxis, :]

    A = np.zeros((M * N, M * N))

    for m in range(M):
        low = m * N
        high = (m + 1) * N
        A[low:high, low:high] = (
            -np.pi
            / 2
            * n
            * n_
            * (3 + 4 * m)
            / (2 + 4 * m + n + n_)
            * scipy.special.beta(n + n_ - 1, 3 + 4 * m)
        )

    b = np.zeros(M * N)
    n = n_og
    for m in range(M):
        b[(m * N) : ((m + 1) * N)] = (
            -2 / (2 * m + 1) * scipy.special.beta(2 * m + 3, n + 1)
        )

    return A, b


def solve_C(N, M):
    A, b = construct(N, M)
    a = np.linalg.solve(A, b)
    C = -32 / np.pi * np.dot(b, a)
    return C


def draw_C():
    m_values = range(0, 150, 10)
    n_values = range(1, 151, 10)

    C = solve_C(150, 150)
    print(C)

    Cs = []
    n = 150
    for m in m_values:
        Cs.append(solve_C(n, m))
    Cs = np.abs(C - np.array(Cs))
    plt.plot(m_values, Cs, "-o")
    plt.xlabel("m")
    plt.ylabel(r"$|\Psi_{150, 150} - \Psi_{m, 150}|$")
    plt.yscale("log")
    plt.grid()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/mprecision.pdf")
    plt.show()

    Cs = []
    m = 150
    for n in n_values:
        Cs.append(solve_C(n, m))
    Cs = np.abs(C - np.array(Cs))
    plt.plot(n_values, Cs, "-o")
    plt.xlabel("n")
    plt.yscale("log")
    plt.ylabel(r"$|\Psi_{150, 150} - \Psi_{150, n}|$")
    plt.grid()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/nprecision.pdf")
    plt.show()


def draw_funcs():
    ksi = np.linspace(0, 1, 200)
    phi = np.linspace(0, np.pi, 200)
    psi = (
        lambda ksi, phi, m, n: np.power(ksi, 2 * m + 1)
        * np.power(1 - ksi, n)
        * np.sin((2 * m + 1) * phi)
    )

    M = 100
    N = 100
    A, b = construct(N, M)
    a = np.linalg.solve(A, b)
    x = ksi[:, np.newaxis] * np.cos(phi[np.newaxis, :])
    y = ksi[:, np.newaxis] * np.sin(phi[np.newaxis, :])
    z = np.zeros((ksi.size, phi.size))

    #for m in range(M):
    #    for n in range(1, N + 1):
    #        val = psi(ksi[:, np.newaxis], phi[np.newaxis, :], m, n) * a[m * N + n - 1]
    #        z += val
    m = 1
    n = 4
    z = psi(ksi[:, np.newaxis], phi[np.newaxis, :], m, n)

    plt.figure(figsize=(8, 6))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    theta = np.linspace(0, np.pi, 500)
    plt.plot(np.cos(theta), np.sin(theta), color="black", linewidth=2)
    plt.axhline(0, color="black", linewidth=2)
    plt.pcolormesh(x, y, z, shading="auto", cmap="magma")
    plt.colorbar()
    plt.savefig(f'images/pretok{m}-{n}.png')
    plt.show()


def analytical_wave(ksi, t):
    N = 50
    psi = lambda x, k: 1 / np.sqrt(2 * np.pi) * np.exp(1j * k * x)
    ks = np.arange(-N // 2, N // 2 + 1, 1)
    funcs = psi(ksi, ks[:, np.newaxis])
    a_ks = []
    for i, k in enumerate(ks):
        if k >= 0:
            bessel = scipy.special.jv(k, np.pi)
        else:
            bessel = (
                np.cos(k * np.pi) * scipy.special.jv(-k, np.pi) +
                np.sin(k * np.pi) * scipy.special.yv(-k, np.pi)
            )
        analytical = (
            np.sin(k * np.pi / 2) * scipy.special.jv(k, np.pi) * np.exp(1j * k * t)
        )
        a_ks.append(analytical)

    a_ks = np.array(a_ks)
    vals = np.sum(a_ks[:, :, np.newaxis] * funcs[:, np.newaxis, :], axis=0)
    return vals


def galerkin_wave(ksi, t):
    N = 50
    psi = lambda x, k: 1 / np.sqrt(2 * np.pi) * np.exp(1j * k * x)
    derivative = lambda t, a_k, k: [-k * a_k[1], k * a_k[0]]
    ks = np.arange(-N // 2, N // 2 + 1, 1)
    funcs = psi(ksi, ks[:, np.newaxis])

    a_ks = []
    for i, k in enumerate(ks):
        ak0 = scipy.integrate.trapz(
            np.sin(np.pi * np.cos(ksi)) * np.conj(funcs[i, :]), ksi
        )

        sol = scipy.integrate.solve_ivp(
            lambda t, y: derivative(t, y, k),
            [np.min(t.real), np.max(t.real)],
            [ak0.real, ak0.imag],
            method="RK45",
            t_eval=t.real,
        )
        a_ks.append(sol.y[0, :] + 1j * sol.y[1, :])

    a_ks = np.array(a_ks)
    vals = np.sum(a_ks[:, :, np.newaxis] * funcs[:, np.newaxis, :], axis=0)
    return vals


def differential_wave(ksi, t):
    dt = t[1] - t[0]
    dksi = ksi[1] - ksi[0]
    initial = np.sin(np.pi * np.cos(ksi))

    u = initial.copy()
    vals = [initial]
    for i in range(len(t) - 1):
        for j in range(1, len(ksi) - 1):
            u[j] = u[j] + dt / dksi * (u[j + 1] - u[j])
        u[0] = u[-2]
        u[-1] = u[1]
        vals.append(u.copy())
    vals = np.array(vals)
    return vals


def solve_wave():
    t = np.linspace(0, 10, 1000, dtype=np.complex64)
    ksi = np.linspace(0, 2 * np.pi, 500, dtype=np.complex64)
    anal = analytical_wave(ksi, t)
    galerkin = galerkin_wave(ksi, t)
    diff = differential_wave(ksi, t)
    draw(np.abs(diff / 3 - anal), ksi.real, t.real, name="wavediff")
    draw(np.abs(galerkin / 3 - anal), ksi.real, t.real, name="wavegalerkin")
    draw(np.abs(anal), ksi.real, t.real, name="waveanal")


solve_wave()
# print(solve_C(150, 150))
# draw_C()
# draw_funcs()
