import multiprocessing
import numpy as np
import matplotlib.pyplot as plt


def get_dist_M_times(N, M, mu, rng):
    rho = rng.uniform(low=0, high=1, size=(M, N))
    dist = np.power(1 - rho, 1 / (1 - mu))
    return dist


# M number of walks, N number of steps
def generate_2Dflights(N, M, mu, rng):
    dist = get_dist_M_times(N, M, mu, rng)
    phi = rng.uniform(low=0, high=2 * np.pi, size=(M, N))
    x = dist * np.cos(phi)
    y = dist * np.sin(phi)

    x = np.cumsum(x, axis=1)
    y = np.cumsum(y, axis=1)

    t = np.array([i for i in range(1, N + 1)])
    return x, y, t


def generate_2Dwalks(N, M, mu, rng):
    x, y, og_t = generate_2Dflights(N, M, mu, rng)
    x_M = []
    y_M = []
    for m in range(M):
        prev = np.array([0, 0])
        t = 0
        new_x = np.zeros(N)
        new_y = np.zeros(N)
        ind = 0
        for n in range(N):
            point = np.array([x[m, n], y[m, n]])
            diff = point - prev
            dist = max(np.linalg.norm(diff), 1e-16)
            diff_norm = diff / dist

            if np.isinf(dist) or np.isnan(dist) or np.any(np.isinf(diff_norm) | np.isnan(diff_norm)): continue
            dt = dist + (t - int(t))
            for full in range(1, int(dt) + 1):
                if N <= ind: break
                new_point = prev + diff_norm * (full - (t - int(t)))
                new_x[ind] = new_point[0]
                new_y[ind] = new_point[1]
                ind += 1
            t += dist
            prev = point
            if N < ind: break
        x_M.append(new_x)
        y_M.append(new_y)
    return np.array(x_M), np.array(y_M), og_t


def fit_function(t, var):
    i = np.linspace(1, t.size - 1, 10, dtype=int)
    j = np.arange(1, t.size - 1, 10, dtype=int)

    i_grid, j_grid = np.meshgrid(i, j, indexing='ij')

    mask = i_grid != j_grid

    i = i_grid[mask]
    j = j_grid[mask]
    gamma_grid = np.log(var[j] / var[i]) / np.log(t[j] / t[i])
    a_grid = var[i_grid[mask]] / np.power(i_grid[mask], gamma_grid)

    # Compute averages
    gamma = np.mean(gamma_grid)
    a = np.mean(a_grid)
    return a, gamma


def calculate_MAD(x, y):
    distances = np.sqrt(np.power(x, 2) + np.power(y, 2))
    median = np.median(distances.T, axis=1)[:, np.newaxis]
    MAD = np.median(np.abs(distances.T - median), axis=1)
    return 1.4826 * MAD


def get_gamma_value(mu, N, M, gen_fun):
    stds = []
    for i in range(10):
        rng = np.random.default_rng()
        x, y, t = gen_fun(N, M, mu, rng)
        stds.append(calculate_MAD(x, y))
    vars = np.power(stds, 2)
    vars_avg = np.mean(vars, axis=0)

    ind = int(len(t) / 5)
    ind = 100
    a, gamma = fit_function(t[ind:], vars_avg[ind:])
    gamma_err = np.std(np.log(vars))
    return gamma, gamma_err


def theoretical_gamma_flights(mus):
    y = 2 / (mus - 1)
    y[3 < mus] = 1
    return y


def theoretical_gamma_walks(mus):
    y = 4 - mus
    y[mus < 2] = 2
    y[3 < mus] = 1
    return y


def draw_flights():
    global gamma_flights
    N = 10000
    M = 1000

    def gamma_flights(mu):
        return get_gamma_value(mu, N, M, generate_2Dflights)

    mus = [mu * 0.01 for mu in range(101, 400, 10)]
    with multiprocessing.Pool() as pool:
        results = pool.map(gamma_flights, mus)

    gammas, gamma_errs = zip(*results)
    gammas = np.array(gammas)
    gamma_errs = np.array(gamma_errs)

    mus = np.array(mus)

    plt.plot(mus, 1 / theoretical_gamma_flights(mus), "--", label="teoretično")
    plt.errorbar(mus, 1 / gammas, yerr=(1 / gamma_errs), fmt="o", capsize=3, label="simulirano")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\frac{1}{\gamma}$")
    plt.legend()
    plt.savefig("images/gammamuflights.png")
    plt.close()


def draw_walks():
    global gamma_walks
    N = 1000
    M = 1000

    def gamma_walks(mu):
        return get_gamma_value(mu, N, M, generate_2Dwalks)

    mus = [mu * 0.01 for mu in range(101, 400, 10)]
    with multiprocessing.Pool() as pool:
        results = pool.map(gamma_walks, mus)

    gammas, gamma_errs = zip(*results)
    gammas = np.array(gammas)
    gamma_errs = np.array(gamma_errs)

    mus = np.array(mus)

    plt.plot(mus, theoretical_gamma_walks(mus), "--", label="teoretično")
    plt.errorbar(mus, gammas, yerr=(gamma_errs), fmt="o", capsize=3, label="simulirano")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\gamma$")
    plt.legend()
    plt.savefig("images/gammamuwalks.png")
    plt.close()


# draw_flights()
draw_walks()
# rng = np.random.default_rng()
# x, y, t = generate_2Dendpos(3, 2, 2, rng)
