import pickle
import pysixtrack
import numpy as np
import NAFFlib


epsn_x = 3.5e-6
epsn_y = 3.5e-6
r_max_sigma = 5.
N_r_footp = 5.
N_theta_footp = 5.


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


with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

part = pysixtrack.Particles(**partCO)

# Track a particle to get betas
part.x += 1e-5
part.y += 1e-5

x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = track_particle(line, part, verbose=True)

beta_x, x_max, px_cut = betafun_from_ellip(x_tbt, px_tbt)
beta_y, y_max, py_cut = betafun_from_ellip(y_tbt, py_tbt)

sigmax = np.sqrt(beta_x * epsn_x / part.beta0 / part.gamma0)
sigmay = np.sqrt(beta_y * epsn_y / part.beta0 / part.gamma0)

import footprint
xy_norm = footprint.initial_xy_polar(r_min=1e-2, r_max=r_max_sigma, r_N=N_r_footp + 1,
                                     theta_min=np.pi / 100, theta_max=np.pi / 2 - np.pi / 100,
                                     theta_N=N_theta_footp)

DpxDpy_wrt_CO = np.zeros_like(xy_norm)

for ii in range(xy_norm.shape[0]):
    for jj in range(xy_norm.shape[1]):
        print('FP track %d/%d (%d/%d)'%(ii, xy_norm.shape[0], jj, xy_norm.shape[1]))

        DpxDpy_wrt_CO[ii, jj, 0] = xy_norm[ii, jj, 0] * np.sqrt(epsn_x / part.beta0 / part.gamma0 / beta_x)
        DpxDpy_wrt_CO[ii, jj, 1] = xy_norm[ii, jj, 1] * np.sqrt(epsn_y / part.beta0 / part.gamma0 / beta_y)
