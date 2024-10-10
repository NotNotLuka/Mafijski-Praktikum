import matplotlib.pyplot as plt
from airy import Airy
from mpmath import mp, linspace, mpf, airyaizero, airybizero
import multiprocessing

plt.rcParams.update({"font.size": 14})
mp.dps = 100

diff_Ai_rel = mpf(0)
diff_Ai_abs = mpf(0)

x = mpf(5)


def calculate_max(start, delta, exact_fun, approximate_fun, diff_fun):
    xs = []
    diffs = []
    x = mpf(start)
    diff = 0
    while diff < 10e-10:
        xs.append(x)
        exact_pos = exact_fun(x)
        approximate_pos = approximate_fun(x)

        diff_pos = diff_fun(exact_pos, approximate_pos)

        diff += diff_pos
        diffs.append(diff_pos)

        x += mpf(delta)
    return xs, diffs


def get_error(args):
    x, exact_fun, approximate_fun, err_fun = args
    exact = exact_fun(x)
    approximate = approximate_fun(x)

    return err_fun(exact, approximate)


def interval_error(start, end, step, exact_fun, approximate_fun, err_fun):
    x = start
    xs = []
    while x < end:
        xs.append((x, exact_fun, approximate_fun, err_fun))
        x += step
    with multiprocessing.Pool() as pool:
        errors = pool.map(get_error, xs)

    return [x[0] for x in xs], errors


def abs_diff(x, y):
    return abs(x - y)


def rel_diff(x, y):
    return abs(x - y) / abs(x) if x != 0 else 0


airy = Airy()


def exact_Ai(x):
    return airy.Ai(x, exact=True)


def exact_Bi(x):
    return airy.Bi(x, exact=True)


def draw_errors_n(start, end, step, name, approximate, exact):
    global taylor  # have to globalize and redefine as multiprocessing cannot pickle it otherwise
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    graph_rel = []
    graph_abs = []
    for N in range(start, end, step):
        def taylor(x):
            return approximate(x, N=N)
        xs_rel, err_rel = interval_error(mpf(-20), mpf(20), mpf(1) / mpf(2), exact, taylor, rel_diff)
        graph_rel.append(sum(err_rel))
        xs_abs, err_abs = interval_error(mpf(-20), mpf(20), mpf(1) / mpf(2), exact, taylor, abs_diff)
        graph_abs.append(sum(err_abs))

    Ns = [i for i in range(start, end, step)]
    fig.suptitle(f'Napaka taylorjevega razvoja {name} v odvisnosti od števila členov')
    ax[0].scatter(Ns, graph_rel)
    ax[0].set_xlabel("N")
    ax[0].set_yscale("log")
    ax[0].set_title("Relativna napaka")

    ax[1].scatter(Ns, graph_abs)
    ax[1].set_xlabel("x")
    ax[1].set_yscale("log")
    ax[1].set_title("Absolutna napaka")
    fig.savefig(f'images/taylor_cleni_{name}.png')
    plt.close()


def draw_errors(name, type, exact, neg, pos=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    if pos is None: pos = neg
    xs_neg, err_neg = interval_error(mpf(-60), mpf(-1) / mpf(10), mpf(1) / mpf(2), exact, neg, rel_diff)
    xs_pos, err_pos = interval_error(mpf(1) / mpf(10), mpf(60), mpf(1) / mpf(2), exact, pos, rel_diff)
    xs = xs_neg + xs_pos
    err = err_neg + err_pos

    ax[0].scatter(xs, err)
    ax[0].set_xlabel("x")
    ax[0].set_yscale("log")
    ax[0].set_title(f'Relativna napaka {type} razvoja {name}')

    xs_neg, err_neg = interval_error(mpf(-60), mpf(-1) / mpf(10), mpf(1) / mpf(2), exact, neg, abs_diff)
    xs_pos, err_pos = interval_error(mpf(1) / mpf(10), mpf(60), mpf(1) / mpf(2), exact, pos, abs_diff)
    xs = xs_neg + xs_pos
    err = err_neg + err_pos

    ax[1].scatter(xs, err)
    ax[1].set_xlabel("x")
    ax[1].set_yscale("log")
    ax[1].set_title(f'Absolutna napaka {type} razvoja {name}')
    fig.savefig(f'images/{name}_{type}_razvoj.png')
    plt.close()


def draw_function(airy):
    x = linspace(-15, 5, 100)
    Ai = [airy.Ai(x_val, exact=True) for x_val in x]
    Bi = [airy.Bi(x_val, exact=True) for x_val in x]

    plt.plot(x, Ai, label="Ai")
    plt.plot(x, Bi, label="Bi")
    plt.ylim(-0.5, 0.7)
    plt.title("Airyjeve funkcije")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("images/both_funcs.png")
    plt.close()


def draw_comparison(airy):
    x = linspace(-15, 5, 500)
    Ai = [airy.Ai(x_val, exact=True) for x_val in x]
    Bi = [airy.Bi(x_val, exact=True) for x_val in x]
    Ai_taylor = [airy._Ai_taylor(x_val) for x_val in x]
    Bi_taylor = [airy._Bi_taylor(x_val) for x_val in x]

    plt.plot(x, Ai, label="Ai")
    plt.plot(x, Ai_taylor, "--", label="Ai_taylor")
    plt.plot(x, Bi, label="Bi")
    plt.plot(x, Bi_taylor, "--", label="Bi_taylor")
    plt.ylim(-0.5, 0.7)
    plt.title("Airyjeve funkcije")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("images/taylor_comparison.png")
    plt.close()


def compare_errors(name, exact, approximate,
                   aproximate_neg, aproximate_pos):
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    xs_neg, err_neg = interval_error(mpf(-60), mpf(-10), mpf(1) / mpf(2), exact, aproximate_neg, rel_diff)
    xs_pos, err_pos = interval_error(mpf(10), mpf(60), mpf(1) / mpf(2), exact, aproximate_pos, rel_diff)
    xs = xs_neg + xs_pos
    err = err_neg + err_pos

    xs_taylor, err_taylor = interval_error(mpf(-40), mpf(40), mpf(1) / mpf(2),
                                           exact, approximate, rel_diff)
    ax[0].scatter(xs, err, label="Asimptotski razvoj")
    ax[0].scatter(xs_taylor, err_taylor, label="Taylorjev razvoj")
    ax[0].set_xlabel("x")
    ax[0].set_yscale("log")
    ax[0].set_title(f'Relativna napaka {name}')
    ax[0].legend()

    xs_neg, err_neg = interval_error(mpf(-60), mpf(-10), mpf(1) / mpf(2), exact, aproximate_neg, rel_diff)
    xs_pos, err_pos = interval_error(mpf(10), mpf(60), mpf(1) / mpf(2), exact, aproximate_pos, rel_diff)
    xs = xs_neg + xs_pos
    err = err_neg + err_pos

    xs_taylor, err_taylor = interval_error(mpf(-40), mpf(40), mpf(1) / mpf(2),
                                           exact, approximate, rel_diff)
    ax[1].scatter(xs, err, label="Asimptotski razvoj")
    ax[1].scatter(xs_taylor, err_taylor, label="Taylorjev razvoj")
    ax[1].set_xlabel("x")
    ax[1].set_yscale("log")
    ax[1].set_title(f'Absolutna napaka {name}')
    ax[1].legend()

    fig.savefig(f'images/primerjava_razvojev_{name}.png')
    plt.close()


functions = [("Ai", "asimptotski", exact_Ai, airy._Ai_asy_neg, airy._Ai_asy_pos),
             ("Bi", "asimptotski", exact_Bi, airy._Bi_asy_neg, airy._Bi_asy_pos),
             ("Ai", "taylorjev", exact_Ai, airy._Ai_taylor),
             ("Bi", "taylorjev", exact_Bi, airy._Bi_taylor)]
for args in functions:
    draw_errors(*args)

draw_function(airy)
draw_comparison(airy)
# draw_errors_n(1, 1000, 20, "Ai", airy._Ai_taylor, exact_Ai)  # slower function
# draw_errors_n(1, 1000, 20, "Bi", airy._Bi_taylor, exact_Bi)  # slower function
compare_errors("Ai", exact_Ai, airy._Ai_taylor, airy._Ai_asy_neg, airy._Ai_asy_pos)
compare_errors("Bi", exact_Bi, airy._Bi_taylor, airy._Bi_asy_neg, airy._Bi_asy_pos)


def zeros100(ax, approximate, exact, name):
    diff = []
    for i in range(1, 101):
        diff.append(approximate(i) - exact(i))
    ax.scatter([i for i in range(1, 101)], diff)
    ax.set_yscale("log")
    ax.set_title(f'Absolutna napaka Nte ničle funkcije {name}')
    ax.set_xlabel("N")


fig, ax = plt.subplots(2, 1, figsize=(15, 12))
zeros100(ax[0], airy.Ai_zero, airyaizero, "Ai")
zeros100(ax[1], airy.Bi_zero, airybizero, "Bi")
fig.savefig("images/zeros.png")
