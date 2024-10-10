from mpmath import sqrt, power, sin, cos, exp, pi, airyai, airybi, diff, mpf


class Airy:
    def __init__(self):
        self.alpha = airyai(0)
        self.beta = - diff(airyai, 0)

        self.u_cache = []
        self.u_cache.append(1)

    def Ai(self, x, exact=False):
        if exact:
            return airyai(x)
        if abs(x) < 30:
            return self._Ai_taylor(x)
        elif 0 < x:
            return self._Ai_asy_pos(x)
        else:
            return self._Ai_asy_neg(x)

    def Bi(self, x, exact=False):
        if exact:
            return airybi(x)
        if abs(x) < 30:
            return self._Bi_taylor(x)
        elif 0 < x:
            return self._Bi_asy_pos(x)
        else:
            return self._Bi_asy_neg(x)

    def _u(self, s):
        if len(self.u_cache) <= s:
            for i in range(s + 1):
                if i + 1 < len(self.u_cache): continue
                self.u_cache.append(mpf((6 * i + 5) * (6 * i + 1)) / mpf(72 * (i + 1)) * self.u_cache[i])
        return self.u_cache[s]

    def _L(self, z):
        prev = None
        cur = None
        L_sum = 0
        s = 0
        while prev is None or abs(cur) < abs(prev):
            L_sum += cur if cur is not None else 0
            prev = cur
            cur = self._u(s) / power(z, s)
            s += 1
        return L_sum

    def _P(self, z):
        prev = None
        cur = None
        P_sum = 0
        s = 0
        while prev is None or abs(cur) < abs(prev):
            P_sum += cur if cur is not None else 0
            prev = cur
            cur = power(mpf(-1), s) * self._u(2 * s) / power(z, mpf(2 * s))
            s += 1
        return P_sum

    def _Q(self, z):
        prev = None
        cur = None
        Q_sum = 0
        s = 0
        while prev is None or abs(cur) < abs(prev):
            Q_sum += cur if cur is not None else 0
            prev = cur
            cur = power(mpf(-1), s) * self._u(2 * s + 1) / power(z, mpf(2 * s + 1))
            s += 1
        return Q_sum

    @staticmethod
    def _get_ksi(x):
        if x < 0: x = -x
        return mpf(2) / mpf(3) * power(x, mpf(3) / mpf(2))

    def _Ai_asy_pos(self, x):
        ksi = self._get_ksi(x)
        return exp(-ksi) / (mpf(2) * sqrt(pi) * power(x, mpf(1) / mpf(4))) * self._L(-ksi)

    def _Bi_asy_pos(self, x):
        ksi = self._get_ksi(x)
        return exp(ksi) / (sqrt(pi) * power(x, mpf(1) / mpf(4))) * self._L(ksi)

    def _Ai_asy_neg(self, x):
        ksi = self._get_ksi(x)
        return (mpf(1) / (sqrt(pi) * power(-x, mpf(1) / mpf(4)))) * \
            (sin(ksi - pi / mpf(4)) * self._Q(ksi) +
                cos(ksi - pi / mpf(4)) * self._P(ksi))

    def _Bi_asy_neg(self, x):
        ksi = self._get_ksi(x)
        return (1 / (sqrt(pi) * power(-x, mpf(1) / mpf(4)))) * \
            (-sin(ksi - pi / mpf(4)) * self._P(ksi) +
                cos(ksi - pi / mpf(4)) * self._Q(ksi))

    def _f(self, x, N):
        a_prev = 1
        f_sum = a_prev

        for k in range(0, N):
            a_prev = (power(x, mpf(3))) / ((mpf(3) * mpf(k) + mpf(3)) * (mpf(3) * mpf(k) + mpf(2))) * a_prev
            f_sum += a_prev

        return f_sum

    def _g(self, x, N):
        b_prev = x
        g_sum = b_prev

        for k in range(0, N):
            b_prev = (power(x, 3)) / ((mpf(3) * mpf(k) + mpf(4)) * (mpf(3) * mpf(k) + mpf(3))) * b_prev
            g_sum += b_prev

        return g_sum

    def _Ai_taylor(self, x, N=300):
        return self.alpha * self._f(x, N) - self.beta * self._g(x, N)

    def _Bi_taylor(self, x, N=300):
        return sqrt(3) * (self.alpha * self._f(x, N) + self.beta * self._g(x, N))

    def _f_zeros(self, z):
        return power(z, mpf(2) / mpf(3)) * \
            (mpf(1)
             + mpf(5) / mpf(48) * power(z, -2)
             - mpf(5) / mpf(36) * power(z, -4)
             + mpf(77125) / mpf(82944) * power(z, -6)
             - mpf(108056875) / mpf(6967296) * power(z, -8))

    def Ai_zero(self, s):
        return - self._f_zeros((3 * pi * (4 * s - 1)) / mpf(8))

    def Bi_zero(self, s):
        return - self._f_zeros((3 * pi * (4 * s - 3)) / mpf(8))
