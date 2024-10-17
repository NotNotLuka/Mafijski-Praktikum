import numpy as np
import matplotlib.pyplot as plt


def get_dist(N, mu, rng):
    rho = rng.uniform(low=0, high=1, size=N)
    dist = np.power(1 - rho, 1 / (1 - mu))
    return dist


def generate_2Dmoves(N, mu, rng):
    dist = get_dist(N, mu, rng)
    phi = rng.uniform(low=0, high=2 * np.pi, size=N)

    x = dist * np.cos(phi)
    y = dist * np.sin(phi)

    x = np.cumsum(x)
    y = np.cumsum(y)

    return x, y, dist


def make_images():
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    Ns = [10, 100, 1000, 10_000]
    mu = 2.5
    for i in range(len(Ns)):
        rng = np.random.default_rng()
        x, y, t = generate_2Dmoves(Ns[i], mu, rng)
        ax[i % 2][i // 2].set_title(fr'$N={Ns[i]}, \mu={mu}$')
        ax[i % 2][i // 2].plot(x, y)
    plt.savefig("images/nice_plots.png")
    plt.close()


def distributions():
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    mu = 2.5
    rng = np.random.default_rng()
    distances = get_dist(10_000, mu, rng)
    dist = distances[distances < 15]
    ax[0].hist(dist, density=True)
    ax[0].set_xlim(0, 15)
    phi = rng.uniform(low=0, high=2 * np.pi, size=10_000)
    ax[1].hist(phi, density=True)
    plt.savefig("images/distribution.png")
    plt.close()
make_images()
distributions()
