import numpy as np
from bvp import shoot, fd
from diffeq import rku4 as rk4
import matplotlib.pyplot as plt
import scipy
import random, sys, time


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 24,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def random_colour():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def schrodinger(psi, x, E):
    V = 0 if x <= 1.0 else np.inf
    dpsi1_dx = psi[1]
    dpsi2_dx = (V - E) * psi[0]
    return np.array([dpsi1_dx, dpsi2_dx])


def solve_shoot(x, f):
    psi_derivative1 = 4.9
    psi_derivative2 = 5
    psi = shoot(f, 0.0, 0.0, psi_derivative1, psi_derivative2, x, tol=1e-2)

    M = np.sqrt(np.trapz(psi**2, x))
    psi = psi / M

    return x, psi


def draw_numerical_solutions():
    boundary_diff = []
    Es = []
    eigen = []
    prev = 0
    x = np.linspace(0, 1, 100)
    for E in np.linspace(0, 500, 1000, dtype=np.float64):
        infinite = lambda psi, x: schrodinger(psi, x, E)
        t, psi_solution = solve_shoot(x, infinite)
        M = np.sqrt(np.trapz(psi_solution**2, t))
        psi = psi_solution / M
        Es.append(E)
        boundary_diff.append(psi[len(psi) - 1])
        if psi[len(psi) - 1] * prev < 0:
            eigen.append((E, psi))
        prev = psi[len(psi) - 1]

    plt.rcParams["font.size"] = 17
    for e in eigen:
        plt.axvline(x=e[0], linestyle="--", color="red")

    plt.plot(Es, boundary_diff)

    plt.xlabel("E")
    plt.ylabel(r"$\psi(x=1)$")
    plt.grid()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/diff.pdf")
    plt.close()

    x = np.linspace(0, 1, 100)
    plt.rcParams["font.size"] = 17
    for i in range(0, 6):
        e = eigen[i]
        colour = random_colour()
        y = e[0] + 10 * e[1]
        plt.plot(x, y, color=colour, label=rf"$E_{i}$")
        plt.axhline(y=e[0], color="black")
        plt.fill_between(x, y, e[0], color=colour, alpha=0.5)
    plt.axvspan(1, 1.4, color="black", alpha=0.5)
    plt.axvspan(0, -0.4, color="black", alpha=0.5)
    plt.xlim(-0.2, 1.2)
    plt.legend()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/eigen.pdf")
    plt.close()


def solve_differential(x, V):
    n = x.shape[0]
    dx = x[1] - x[0]
    diag = np.ones(n) * 2 / (dx**2) + V
    side_diag = np.ones(n - 1) * (-1 / (dx**2))

    V0 = np.min(V)
    if V0 == 0:
        vals, vecs = scipy.linalg.eigh_tridiagonal(diag, side_diag)
    else:
        vals, vecs = scipy.linalg.eigh_tridiagonal(
            diag, side_diag, select_range=[V0, 0], select="v"
        )
    psi = vecs.T
    for i in range(len(vals)):
        psi[i] = psi[i] / np.sqrt(np.trapz(psi[i] ** 2, x))
    return vals, psi


def bisection(f, start, end, x, n=0):
    mid = (start + end) / 2

    infinite = lambda psi, x: schrodinger(psi, x, mid)
    x, psi = f(x, infinite)
    mid_val = np.abs(psi[psi.size - 1])
    if mid_val < 1e-2 or n == 10:
        return (mid, mid_val)

    (val1, precision1) = bisection(f, start, mid, x, n + 1)
    (val2, precision2) = bisection(f, start, mid, x, n + 1)

    if precision1 < precision2:
        return (val1, precision1)
    else:
        return (val2, precision2)


def compare():
    theoretical = lambda x, n: np.sqrt(2) * np.sin(n * np.pi * x)
    eigen_e = [np.power(n * np.pi, 2) for n in range(1, 20)]
    err_shoot = []
    err_diff = []
    x = np.linspace(0, 1, 1000)
    vals, vecs = solve_differential(x, np.zeros_like(x))
    for n in range(1, len(eigen_e) + 1):
        energy = eigen_e[n - 1]
        eigen_num, _ = bisection(solve_shoot, energy - 2, energy + 2, x)
        infinite = lambda psi, x: schrodinger(psi, x, eigen_num)
        x, psi = solve_shoot(x, infinite)
        theory = theoretical(x, n)
        err_shoot.append(np.mean(np.abs(theory - psi)))
        err_diff.append(np.mean(np.abs(theory - vecs[n - 1])))
    plt.plot(range(1, len(eigen_e) + 1), err_shoot, label="Strelska metoda")
    plt.plot(range(1, len(eigen_e) + 1), err_diff, label="Diferenčna metoda")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.xlabel("N-to stanje")
    plt.ylabel(r"$\overline{|\psi_{theory} - \psi}|$")
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/errstates.pdf")
    plt.close()

    err_shoot = []
    err_diff = []
    rang = range(100, 7500, 250)
    for n in rang:
        print(n, 7500)
        x = np.linspace(0, 1, n)
        vals, vecs = solve_differential(np.linspace(0, 1, n), np.zeros_like(x))
        energy = eigen_e[0]
        eigen_num, _ = bisection(solve_shoot, energy - 2, energy + 2, x)
        infinite = lambda psi, x: schrodinger(psi, x, eigen_num)
        x, psi = solve_shoot(x, infinite)
        theory = theoretical(x, 1)
        err_shoot.append(np.mean(np.abs(theory - psi)))
        err_diff.append(np.mean(np.abs(theory - vecs[0])))

    plt.plot(rang, err_shoot, label="Strelska metoda")
    plt.plot(rang, err_diff, label="Diferenčna metoda")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("N korakov")
    plt.ylabel(r"$\overline{|\psi_{theory} - \psi}|$")
    plt.legend()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/errstep.pdf")
    plt.close()


def finite_well():
    def get_V(x):
        V = np.zeros_like(x)
        V[0.5 > np.abs(x)] = -100
        return V

    x = np.linspace(-3, 3, 1000)
    V = get_V(x)
    vals, vecs = solve_differential(x, V)
    V0 = np.min(V)
    vals += -V0
    print(vals)

    plt.rcParams["font.size"] = 17
    plt.fill_betweenx(np.linspace(0, -V0, 100), 0.5, 3, color="black", alpha=0.5)
    plt.fill_betweenx(np.linspace(0, -V0, 100), -3, -0.5, color="black", alpha=0.5)
    for i in range(len(vecs)):
        colour = random_colour()
        y = vals[i] + 5 * vecs[i]

        plt.axhline(y=vals[i], color="black", linestyle="--", alpha=0.5)
        plt.plot(x, y, color=colour, label=rf"$E_{i}$")
        plt.fill_between(x, y, vals[i], color=colour, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("E")
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 110)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/finiteeigen.pdf")
    # plt.show()
    plt.close()


def get_analytical(V0, x0):
    u0 = np.sqrt(V0) / 2
    fun = lambda u: np.tan(u) - np.sqrt(np.power(u0 / u, 2) - 1)
    result = scipy.optimize.root_scalar(fun, bracket=(1, 1.5))
    x = result.root
    kappa = 2 * x * (1 / np.tan(x))
    k = 2 * x

    x1 = np.linspace(-x0, x0, 500)
    x2 = np.linspace(x0, 300, 1000)
    E = np.trapz(np.power(np.cos(k * x1), 2))
    C = 2 * np.trapz(np.exp(-2 * kappa * x2))
    G = np.exp(-kappa * x0) / np.cos(k * x0)
    D = np.sqrt(1 / (np.power(G, 2) * E + C))
    B = D * G

    def eigen(x):
        out = np.zeros_like(x)
        xbelow = x[np.abs(x) <= x0]
        xabove = x[x0 < np.abs(x)]
        out[np.abs(x) <= x0] = B * np.cos(k * xbelow)
        out[x0 < np.abs(x)] = D * np.exp(-kappa * np.abs(xabove))
        return out

    return eigen


def schrodinger1(x, psi, E, x0):
    V = np.zeros_like(x)
    V[np.abs(x) < x0] = -100
    dpsi1_dx = psi[1]
    dpsi2_dx = (V - E) * psi[0]
    return [dpsi1_dx, dpsi2_dx]


def solve_shoot1(x, f):
    sol = scipy.integrate.solve_ivp(
        f, [x[0], x[-1]], [1e-10, 1e-10], t_eval=x, method="RK23"
    )
    psi = sol.y[0]
    M = np.sqrt(np.trapz(psi**2, x))
    psi /= M
    return psi


def shoot_finite():
    x = np.linspace(-10, 0, 1000)
    x0 = 0.5
    diff = []
    Es = np.linspace(-100, 0, 500, dtype=np.float64)

    prev = 1
    eigen = []
    for E in Es:
        finite = lambda t, psi: schrodinger1(t, psi, E, x0)
        psi = solve_shoot1(x, finite)
        dx = x[1] - x[0]
        dy = (psi[-1] - psi[-2]) / dx
        if prev * dy < 0:
            eigen.append(E)
        prev = dy
        diff.append(dy)

    print(np.array(eigen) + 100)
    plt.plot(Es, diff)
    plt.xlabel("E")
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("images/10oddaljenost.pdf")


# draw_numerical_solutions()
# compare()
# finite_well()
shoot_finite()
