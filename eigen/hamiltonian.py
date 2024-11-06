import cupy as cp


def define_factorial_div():
    global factorial_div
    if cp.__name__ == "cupy":
        factorial_div = cp.ElementwiseKernel(
            "float64 x, float64 y",  # Input types
            "float64 z",  # Output type
            """
            z = 1;

            int start = min(x, y);
            int end = max(x, y);

            for (int k = start + 1; k <= end; k++) {
                z *= sqrtf(k);
            }
            if (x < y) {
                z = 1 / z;
            }
            """,
            "factorial_div_kernel",
        )

    if cp.__name__ == "numpy":
        def factorial_div(x, y):
            z = 1.0

            start = min(x, y)
            end = max(x, y)

            for k in range(int(start) + 1, int(end) + 1):
                z *= cp.sqrt(k)
            if x < y:
                z = 1 / z
            return z

        factorial_div = cp.vectorize(factorial_div)


def kronecker_delta(basis=None, N=None, i=0, j=0):
    if basis is None and N is None:
        raise ValueError("Both basis and N cannot be None")
    if basis is None:
        basis = cp.zeros((N, N), dtype=cp.float64)
    if N is None:
        N = basis.shape[0]

    tmp = cp.arange(N).reshape(1, N)
    x = cp.repeat(tmp, N, axis=0)
    y = cp.copy(x).T
    x += i
    y += j
    mask = x == y
    basis[mask] = 1
    return basis


def get_q(N):
    basis = cp.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if abs(i - j) == 1:
                basis[i, j] = 0.5 * cp.sqrt(i + j + 1)
    return basis


def get_q2(N):
    deltajm2 = kronecker_delta(N=N, j=-2)
    deltajp2 = kronecker_delta(N=N, j=2)

    tmp = cp.arange(N, dtype=cp.float64).reshape(1, N)
    x = cp.repeat(tmp, N, axis=0)
    y = cp.copy(x).T

    return (
        1
        / 2
        * (
            cp.sqrt(y * (y - 1)) * deltajm2
            + (2 * y + 1) * cp.eye(N)
            + cp.sqrt((y + 1) * (y + 2)) * deltajp2
        )
    )


def get_q4(N):
    deltajm4 = kronecker_delta(N=N, j=-4)
    deltajm2 = kronecker_delta(N=N, j=-2)
    deltajp2 = kronecker_delta(N=N, j=2)
    deltajp4 = kronecker_delta(N=N, j=4)

    tmp = cp.arange(N, dtype=cp.float64).reshape(1, N)
    x = cp.repeat(tmp, N, axis=0)
    y = cp.copy(x).T

    right = (
        deltajp4
        + 4 * (2 * y + 3) * deltajp2
        + 12 * (2 * cp.power(y, 2) + 2 * y + 1) * cp.eye(N)
        + 16 * y * (2 * cp.power(y, 2) - 3 * y + 1) * deltajm2
        + 16 * y * (cp.power(y, 3) - 6 * cp.power(y, 2) + 11 * y - 6) * deltajm4
    )

    x[right == 0] = 0
    y[right == 0] = 0
    return 1 / (2**4) * cp.sqrt(cp.power(2, x - y)) * factorial_div(x, y) * right


def harmonic_hamiltonian(N):
    return (cp.arange(N) + 1 / 2) * cp.eye(N)


def anharmonic_hamiltonian(N, lamda, qpower=4):
    harmonic = harmonic_hamiltonian(N)
    match qpower:
        case 1:
            q = get_q(N)
            q2 = cp.matmul(q, q)
            q4 = cp.matmul(q2, q2)
            lambdapower = lamda * q4
        case 2:
            q2 = get_q2(N)
            q4 = cp.matmul(q2, q2)
            lambdapower = lamda * q4
        case 4:
            lambdapower = lamda * get_q4(N)
        case _:
            raise ValueError("Invalid q power")

    anharmonic = harmonic + lambdapower
    return anharmonic


def extra_hamiltonian(N):
    harmonic = harmonic_hamiltonian(N)
    q2 = get_q2(N)
    q4 = get_q4(N)
    extra = harmonic - 5 / 2 * q2 + 1 / 10 * q4
    return extra


if __name__ == "__main__":
    define_factorial_div()
    print(get_q(5))
