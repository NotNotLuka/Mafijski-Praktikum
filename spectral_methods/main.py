import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 24,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)

a = 1
sigma = 0.25
D = 1


def draw(result, x, ts, name="", force_norm=False):
    if force_norm:
        norm = Normalize(vmax=0.0030, vmin=0)
    else:
        norm = Normalize(vmax=np.max(result), vmin=np.min(result))
    plt.imshow(
        result.T,
        aspect="auto",
        cmap=cm.coolwarm,
        norm=norm,
        extent=[min(ts), max(ts), min(x), max(x)],
    )
    plt.xlabel("t")
    plt.ylabel("x")

    plt.colorbar()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    if name != "":
        plt.savefig(f"images/{name}2d.pdf")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    T, X = np.meshgrid(ts, x)
    surface = ax.plot_surface(
        T, X, result.T, cmap=cm.coolwarm, shade=True, antialiased=True
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


def fourier_solve(x, ts, initial, a):
    fft = np.fft.rfft(initial)

    k = np.arange(0, fft.size, 1)
    exp_factor = np.exp(-4 * np.power(np.pi * k / a, 2) * D * ts[:, np.newaxis])

    fft_timewise = fft * exp_factor
    result = np.fft.irfft(fft_timewise, axis=1).real
    return result


def fourier_draw(homogoneous=False):
    T0 = lambda x: np.exp(-((x - a / 2) ** 2) / sigma**2)

    x = np.linspace(0, 1, 1000)
    ts = np.linspace(0, 0.3, 1000)
    gauss_initial = T0(x)
    domain = a

    if homogoneous:
        gauss_initial = np.concatenate((-gauss_initial, gauss_initial))
        N = len(gauss_initial)
        domain = 2 * a
    result = fourier_solve(x, ts, gauss_initial, domain)

    if homogoneous:
        result = result[:, N // 2:]

    result = np.array(result)

    draw(result, x, ts, "hom_fourier" if homogoneous else "fourier")


def bspline(x, k, dx):
    x_k_2 = dx * (k - 2)
    x_k_1 = dx * (k - 1)
    x_k = dx * k
    x_k1 = dx * (k + 1)
    x_k2 = dx * (k + 2)
    if x <= x_k_2:
        return 0
    elif x <= x_k_1:
        return (1 / dx**3) * (x - x_k_2) ** 3
    elif x <= x_k:
        return (1 / dx**3) * ((x - x_k_2) ** 3 - 4 * (x - x_k_1) ** 3)
    elif x <= x_k1:
        return (1 / dx**3) * ((x_k2 - x) ** 3 - 4 * (x_k1 - x) ** 3)
    elif x <= x_k2:
        return (1 / dx**3) * (x_k2 - x) ** 3
    else:
        return 0


def get_spline_sum(x, c, dx):
    sol = 0
    N = len(c) + 1
    i = int(x / dx)
    for k in range(i - 1, i + 2):
        if k == -1:
            c_k = -c[0]
        elif k == 0 or k == N:
            c_k = 0
        elif k == N + 1:
            c_k = -c[-1]
        else:
            c_k = c[k - 1]
        spline = bspline(x, k, dx)
        sol += c_k * spline
    return sol


def get_solution(xs, c, dx):
    solution = []
    for x in xs:
        solution.append(get_spline_sum(x, c, dx))
    return solution


def splines_solve(x, ts, initial):
    N = x.size
    dx = x[1] - x[0]
    diagonals_A = [4 * np.ones(N), np.ones(N - 1), np.ones(N - 1)]
    A = (
        np.diag(diagonals_A[0], k=0)
        + np.diag(diagonals_A[1], k=1)
        + np.diag(diagonals_A[2], k=-1)
    )

    diagonals_B = [-2 * np.ones(N), np.ones(N - 1), np.ones(N - 1)]
    B = (
        np.diag(diagonals_B[0], k=0)
        + np.diag(diagonals_B[1], k=1)
        + np.diag(diagonals_B[2], k=-1)
    )
    B = 6 * D / dx**2 * B

    dt = ts[1] - ts[0]
    right = A + dt / 2 * B
    left = np.linalg.inv(A - dt / 2 * B)
    term = right @ left

    results = []
    c0 = np.linalg.solve(A, initial)
    cn = c0
    for _ in ts:
        sol = get_solution(x, cn, dx)
        results.append(sol)
        cn = term @ cn

    results = np.array(results)
    return results


def splines_draw():
    T0 = lambda x: np.exp(-((x - a / 2) ** 2) / sigma**2)

    x = np.linspace(0, 1, 1000)
    ts = np.linspace(0, 0.1, 1000)
    gauss_initial = T0(x)
    result = splines_solve(x, ts, gauss_initial)

    draw(result, x, ts, "splines")


def compare(clean=False):
    T0 = lambda x: np.exp(-((x - a / 2) ** 2) / sigma**2)

    x = np.linspace(0, 1, 1000)
    ts = np.linspace(0, 0.1, 1000)
    gauss_initial = T0(x)
    gauss_initial[0] = 0
    gauss_initial[-1] = 0
    splines = splines_solve(x, ts, gauss_initial)
    # draw(splines, x, ts)
    gauss_initial1 = np.concatenate((-gauss_initial, gauss_initial))
    N = len(gauss_initial1)
    fourier = fourier_solve(x, ts, gauss_initial1, 2 * a)[:, N // 2:]
    # draw(fourier, x, ts)
    gauss_initial2 = np.concatenate((-gauss_initial, gauss_initial, -gauss_initial))
    N = len(gauss_initial2)
    fourier2 = fourier_solve(x, ts, gauss_initial2, 3 * a)[:, N // 3:2 * N // 3]
    print(N)
    # draw(fourier2, x, ts)
    diff = np.abs(fourier - splines)
    if clean:
        diff = diff.T
        row_averages = np.mean(diff, axis=1)
        threshold = np.percentile(row_averages, 99)
        new_diff = diff[row_averages < threshold]
        x = x[row_averages < threshold]
        diff = new_diff.T

    draw(diff, x, ts, "diffclean", True)


fourier_draw(True)
fourier_draw()
splines_draw()
compare(True)
