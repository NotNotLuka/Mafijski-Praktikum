import numpy as np
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots()
    plot = ax.imshow(
        result.T[::-1, :],
        aspect="auto",
        cmap=cm.coolwarm,
        extent=[min(ts), max(ts), min(x), max(x)],
    )
    ax.set_xlabel("t")
    ax.set_ylabel("x")

    fig.colorbar(plot)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    if name != "":
        plt.savefig(f"images/{name}.pdf")


def get_matrix(degree, dx, dt, V):
    coefficients_dict = {
        1: [-2, 1],
        2: [-5 / 2, 4 / 3, -1 / 12],
        3: [-49 / 18, 3 / 2, -3 / 20, 1 / 90],
        4: [-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
        5: [-5269 / 1800, 5 / 3, -5 / 21, 5 / 126, -5 / 1008, 1 / 3150],
        6: [-5369 / 1800, 12 / 7, -15 / 56, 10 / 189, -1 / 112, 2 / 1925, -1 / 16632],
        7: [
            -266681 / 88200,
            7 / 4,
            -7 / 24,
            7 / 108,
            -7 / 528,
            7 / 3300,
            -7 / 30888,
            1 / 84084,
        ],
    }
    coefficients = coefficients_dict[degree]
    a = - 1j * dt / (4 * dx**2)
    d = lambda c, V: 1 + c + 1j * dt / 2 * V
    d = d(a * coefficients[0], V)
    A = np.diag(d, 0).copy()
    for i in range(1, len(coefficients)):
        line = [a * coefficients[i]] * (len(V) - i)
        A += np.diag(line, i) + np.diag(line, -i)

    return A


def solve(xs, ts, initial, V, degree=1):
    dx = xs[1] - xs[0]
    dt = ts[1] - ts[0]

    A = get_matrix(degree, dx, dt, V)
    Ac = np.conj(A)
    A_inv = np.linalg.inv(A)

    sol = []
    state = initial
    for t in ts:
        add = np.abs(state)
        sol.append(add)
        # state = np.linalg.solve(A, Ac @ state)
        state = A_inv @ Ac @ state

    sol = np.array(sol, dtype=np.float64)
    return sol


def analytical_first(x, t, k=0.04, lambd=10):
    omega = np.sqrt(k)
    alpha = np.sqrt(omega)
    xi = alpha * x
    xi_lambda = alpha * lambd

    real_part = -0.5 * np.power((xi - xi_lambda * np.cos(omega * t[:, np.newaxis])), 2)
    imag_part = -(
        0.5 * omega * t[:, np.newaxis]
        + xi * xi_lambda * np.sin(omega * t[:, np.newaxis])
        - 0.25 * xi_lambda * xi_lambda * np.sin(2 * omega * t[:, np.newaxis])
    )

    psi_val = np.sqrt(alpha / np.sqrt(np.pi)) * np.exp(real_part + 1j * imag_part)
    return psi_val


def initial_first(xs, k=0.04, lambd=10):
    omega = np.sqrt(k)
    alpha = np.sqrt(omega)
    initial = np.sqrt(alpha / np.sqrt(np.pi)) * np.exp(
        -np.power(alpha * (xs - lambd), 2) / 2
    )
    return initial


def difference():
    xs = np.linspace(-40, 40, 300, dtype=np.complex128)
    ts = np.linspace(0, np.pi * 10 * 10, 5000, dtype=np.complex128)
    k = 0.04
    lambd = 10
    initial = initial_first(xs, k, lambd)
    V = 0.5 * k * np.power(xs, 2)
    sol_1 = solve(xs, ts, initial, V, degree=1)
    sol_2 = solve(xs, ts, initial, V, degree=2)
    sol_6 = solve(xs, ts, initial, V, degree=6)
    sol_7 = solve(xs, ts, initial, V, degree=7)
    anal = analytical_first(xs, ts, k=k, lambd=lambd)

    plt.rcParams["font.size"] = 16
    draw(np.abs(anal), xs.real, ts.real, name="anal")
    plt.rcParams["font.size"] = 24
    draw(np.abs(sol_1 - anal), xs.real, ts.real, name="red1")
    draw(np.abs(sol_2 - anal), xs.real, ts.real, name="red2")
    draw(np.abs(sol_7 - anal), xs.real, ts.real, name="red7")
    draw(np.abs(sol_2 - sol_7), xs.real, ts.real, name="red27")
    draw(np.abs(sol_6 - sol_7), xs.real, ts.real, name="red67")
    plt.show()


def analytical_gauss(x, t, sigma=1 / 20, k=50 * np.pi, lambda_=0.25):
    up = (
        -np.power((x - lambda_) / (2 * sigma), 2)
        + 1j * k * (x - lambda_)
        - 1j * k * k * t[:, np.newaxis] / 2
    )
    below = 1 + 1j * t[:, np.newaxis] / (2 * sigma**2)
    exponent = up / below

    up = np.power(2 * np.pi * sigma ** 2, -1 / 4)
    below = np.sqrt(1 + 1j * t[:, np.newaxis] / (2 * sigma ** 2))
    A = up / below

    return A * np.exp(exponent)


def difference_gauss():
    xs = np.linspace(-0.5, 1.5, 300, dtype=np.complex128)
    dx = xs[1] - xs[0]
    dt = int((3e-3 / (2 * dx ** 2)).real)
    ts = np.linspace(0, 3e-3, dt, dtype=np.complex128)
    k = 50 * np.pi
    sigma = 1 / 20
    lambda_ = 0.25
    initial = (
        np.power(2 * np.pi * sigma**2, -1 / 4)
        * np.exp(1j * k * (xs - lambda_))
        * np.exp(-np.power((xs - lambda_) / (2 * sigma), 2))
    )
    V = np.array([0] * len(xs))
    sol_1 = solve(xs, ts, initial, V, degree=1)
    sol_7 = solve(xs, ts, initial, V, degree=7)
    anal = analytical_gauss(xs, ts, sigma=sigma, k=k, lambda_=lambda_)

    plt.rcParams["font.size"] = 16
    draw(np.abs(anal), xs.real, ts.real, name="analgauss")
    plt.rcParams["font.size"] = 24
    draw(np.abs(anal - sol_1), xs.real, ts.real, name="diff1")
    draw(np.abs(sol_1 - sol_7), xs.real, ts.real, name="diff7")

    ts = np.linspace(0, 1, 2200, dtype=np.complex128)
    nonstable = solve(xs, ts, initial, V, degree=1)
    plt.rcParams["font.size"] = 16
    draw(np.abs(nonstable), xs.real, ts.real, name="nonstable")
    plt.show()


# difference()
difference_gauss()
