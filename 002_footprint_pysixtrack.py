import pickle
import pysixtrack
import numpy as np

n_turns = 100
epsn_x = 3.5e-6
epsn_y = 3.5e-6
N_sigma_footp = 7.
N_theta_footp = 5.

def track_particle(line, part):

    x_tbt = np.zeros(n_turns)
    px_tbt = np.zeros(n_turns)
    y_tbt = np.zeros(n_turns)
    py_tbt = np.zeros(n_turns)
    sigma_tbt = np.zeros(n_turns)
    delta_tbt = np.zeros(n_turns)

    for i_turn in range(n_turns):
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


with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

part = pysixtrack.Particles(**partCO)

# Track a particle to get betas
part.x = 1e-5
part.y = 1e-5

x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = track_particle(line, part)

beta_x, x_max, px_cut = betafun_from_ellip(x_tbt, px_tbt)
beta_y, y_max, py_cut = betafun_from_ellip(y_tbt, py_tbt)

sigmax = np.sqrt(beta_x * epsn_x / part.beta0 / part.gamma0)
sigmay = np.sqrt(beta_y * epsn_y / part.beta0 / part.gamma0)

import footprint
xy_norm = footprint.initial_xy_polar(r_min = 1e-2, r_max=N_sigma_footp, r_N=N_sigma_footp+1,
    theta_min=0., theta_max=np.pi/2, theta_N = N_theta_footp)

# Tracking for footprint
#...

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
spx = fig1.add_subplot(2, 1, 1)
spy = fig1.add_subplot(2, 1, 2, sharex=spx)

spx.plot(x_tbt)
spy.plot(y_tbt)

fig2 = plt.figure(2)
spex = fig2.add_subplot(2, 1, 1)
spey = fig2.add_subplot(2, 1, 2)

spex.plot(x_tbt, px_tbt, '.')
spey.plot(y_tbt, py_tbt, '.')

spex.plot(0, px_cut, 'xr')
spey.plot(0, py_cut, 'xr')

fig3 = plt.figure(3)
axcoord = fig3.add_subplot(1, 1, 1)
footprint.draw_footprint(xy_norm, axis_object=axcoord)
axcoord.set_xlim(right=np.max(xy_norm[:, :, 0]))
axcoord.set_ylim(top=np.max(xy_norm[:, :, 1]))

plt.show()
