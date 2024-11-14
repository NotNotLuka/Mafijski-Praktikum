import numpy as np
import matplotlib.pyplot as plt


def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape(
        (N, 1)
    )  # in essence row->column transformation, k*n is then the dot product..
    # print("Vector",k)
    # print("Matrix",k*n)
    M = np.exp(-2j * np.pi * k * n / N)
    # print("Matrix",M)
    return np.dot(M, x)


def DFT_simplest(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    F = []
    for k in range(N):
        fk = 0.0
        for n in range(N):
            Mkn = np.exp(-2j * np.pi * k * n / N)
            fk += Mkn * x[n]
        F.append(fk)
        # print("k,F[k]",k,fk)
    return np.asarray(F)


if __name__ == "__main__":
    # sampling parameters 200,100 = default
    n = 200  # Number of data points
    T = 100.0  # Sampling period
    dt = T / n
    tmin = 0.0
    tmax = dt * n
    print("sampling freq:", 1.0 / dt)
    nuc = 0.5 / dt
    print("critical freq:", nuc)

    t = dt * np.arange(0, n)  # x coordinates - OK
    # t=np.linspace(tmin,tmax,n,endpoint=False) # OK (same)
    # t=np.linspace(tmin,tmax,n) # non-periodic endpoint
    # print(t)

    t01 = 100.0  # sine t0 # spremeni na 110, da vidis ne-periodicnost!
    t02 = 10  # cosine t0 # aliasing n=200,T=2000, t01=100,t02=10
    print("nu1 (sin)=", 1.0 / t01)
    print("nu2 (cos)=", 1.0 / t02)

    ht = np.sin(2 * np.pi * t / t01) + 2 * np.cos(2 * np.pi * t / t02)  # signal

    # ht = np.sin(2*np.pi*t/t01)
    # ht = np.cos(2*np.pi*x/t02)

    # plot the functions
    plt.plot(t, ht, "r.-")
    plt.xlabel("t")
    plt.ylabel("h(t)")
    plt.title("h(t)=sin($\omega_1$t)+2cos($\omega_2$t)")

    plt.show()

    nu = np.linspace(-nuc, nuc, n, endpoint=False)

    dft = DFT_slow(ht)
    # dft=DFT_simplest(fx)

    Hk = np.roll(dft, int(n / 2)) / n
    # fk=dft

    # plt.cla()
    f, ax = plt.subplots(3, 1, sharex=True)
    # Plot Cosine terms
    ax[0].plot(nu, np.real(Hk), color="b")
    ax[0].set_ylabel(r"$Re[H_k]$", size="x-large")
    # Plot Sine terms
    ax[1].plot(nu, np.imag(Hk), color="r")
    ax[1].set_ylabel(r"$Im[H_k]$", size="x-large")
    # Plot spectral power
    ax[2].plot(nu, np.absolute(Hk) ** 2, color="y")
    ax[2].set_ylabel(r"$\vert H_k \vert ^2$", size="x-large")
    ax[2].set_xlabel(r"$\nu$", size="x-large")
    f.suptitle("FT[cos(t)]")
    plt.show()
    input("Press Enter to continue... ")
