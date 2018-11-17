import pickle
import pysixtrack
import numpy as np
import helpers as hp
import footprint
import matplotlib.pyplot as plt

n_turns = 100

with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

Dx_m = 1e-4
Dpx_rad = 1e-6
Dy_m = 2e-4
Dpy_rad = 3e-6
Dsigma_m = 2e-3
Ddelta = 0.

part = pysixtrack.Particles(**partCO)

part.x += Dx_m
part.px += Dpx_rad
part.y += Dy_m
part.py += Dpy_rad
part.sigma += Dsigma_m
part.delta = part.delta + Ddelta

print('Tracking PyST')
x_tbt_pyST, px_tbt_pyST, y_tbt_pyST, py_tbt_pyST, sigma_tbt_pyST, delta_tbt_pyST = hp.track_particle_pysixtrack(line, part, n_turns)

print('Tracking ST')
x_tbt_ST, px_tbt_ST, y_tbt_ST, py_tbt_ST, sigma_tbt_ST, delta_tbt_ST = hp.track_particle_sixtrack(
    Dx_m, Dpx_rad, Dy_m, Dpy_rad, Dsigma_m, Ddelta, n_turns)

plt.close('all')
fig1 = plt.figure(1, figsize=(8 * 1.5, 6 * 1.2))
axx = fig1.add_subplot(3, 2, 1)
axx.plot(x_tbt_pyST)
axx.plot(x_tbt_ST)

axy = fig1.add_subplot(3, 2, 2, sharex=axx)
axy.plot(y_tbt_pyST)
axy.plot(y_tbt_ST)

axpx = fig1.add_subplot(3, 2, 3, sharex=axx)
axpx.plot(px_tbt_pyST)
axpx.plot(px_tbt_ST)

axpy = fig1.add_subplot(3, 2, 4, sharex=axx)
axpy.plot(py_tbt_pyST)
axpy.plot(py_tbt_ST)

axsigma = fig1.add_subplot(3, 2, 5, sharex=axx)
axsigma.plot(sigma_tbt_pyST)
axsigma.plot(sigma_tbt_ST)

axdelta = fig1.add_subplot(3, 2, 6, sharex=axx)
axdelta.plot(delta_tbt_pyST)
axdelta.plot(delta_tbt_ST)

plt.show()
