import numpy as np


def track_particle(line, part, n_turns, verbose=False):

    x_tbt = np.zeros(n_turns)
    px_tbt = np.zeros(n_turns)
    y_tbt = np.zeros(n_turns)
    py_tbt = np.zeros(n_turns)
    sigma_tbt = np.zeros(n_turns)
    delta_tbt = np.zeros(n_turns)

    for i_turn in range(n_turns):
        if verbose:
            print('Turn %d/%d'%(i_turn, n_turns))

        for name, etype, ele in line:
            ele.track(part)

        x_tbt[i_turn] = part.x
        px_tbt[i_turn] = part.px
        y_tbt[i_turn] = part.y
        py_tbt[i_turn] = part.py
        sigma_tbt[i_turn] = part.sigma
        delta_tbt[i_turn] = part.delta

    return x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt


def betafun_from_ellip(x_tbt, px_tbt):

    x_max = np.max(x_tbt)
    mask = np.logical_and(np.abs(x_tbt) < x_max / 5., px_tbt > 0)
    x_masked = x_tbt[mask]
    px_masked = px_tbt[mask]
    ind_sorted = np.argsort(x_masked)
    x_sorted = np.take(x_masked, ind_sorted)
    px_sorted = np.take(px_masked, ind_sorted)

    px_cut = np.interp(0, x_sorted, px_sorted)

    beta_x = x_max / px_cut

    return beta_x, x_max, px_cut
