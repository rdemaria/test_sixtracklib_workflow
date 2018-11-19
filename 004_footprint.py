import pickle
import pysixtrack
import numpy as np
import NAFFlib
import helpers as hp
import footprint
import matplotlib.pyplot as plt

track_with = 'PySixtrack'
# track_with = 'Sixtrack'

n_turns = 100

with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

with open('DpxDpy_for_footprint.pkl', 'rb') as fid:
    temp_data = pickle.load(fid)

xy_norm = temp_data['xy_norm']
DpxDpy_wrt_CO = temp_data['DpxDpy_wrt_CO']


if track_with == 'PySixtrack':

    part = pysixtrack.Particles(**partCO)

    x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_pysixtrack(
        line, part=part, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 0].flatten(),
        Dy_wrt_CO_m=0, Dpy_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 1].flatten(),
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=0., n_turns=n_turns, verbose=True)


'''

    x_tbt = []
    px_tbt = []
    y_tbt = []
    py_tbt = []
    sigma_tbt = []
    delta_tbt = []

    part = pysixtrack.Particles(**partCO)

    part.px += DpxDpy_wrt_CO[:, :, 0].flatten()
    part.py += DpxDpy_wrt_CO[:, :, 1].flatten()

    part.x += 0 * DpxDpy_wrt_CO[:, :, 0].flatten()
    part.y += 0 * DpxDpy_wrt_CO[:, :, 0].flatten()

    part.sigma += 0 * DpxDpy_wrt_CO[:, :, 0].flatten()
    part.delta += 0 * DpxDpy_wrt_CO[:, :, 0].flatten()

    for i_turn in range(n_turns):
        if verbose:
            print('Turn %d/%d' % (i_turn, n_turns))

        x_tbt.append(part.x.copy())
        px_tbt.append(part.px.copy())
        y_tbt.append(part.y.copy())
        py_tbt.append(part.py.copy())
        sigma_tbt.append(part.sigma.copy())
        delta_tbt.append(part.delta.copy())

        for name, etype, ele in line:
            ele.track(part)

    x_tbt = np.array(x_tbt)
    px_tbt = np.array(px_tbt)
    y_tbt = np.array(y_tbt)
    py_tbt = np.array(py_tbt)
    sigma_tbt = np.array(sigma_tbt)
    delta_tbt = np.array(delta_tbt)
'''
n_part = x_tbt.shape[1]
Qx = np.zeros(n_part)
Qy = np.zeros(n_part)

for i_part in range(n_part):
    Qx[i_part] = NAFFlib.get_tune(x_tbt[:, i_part])
    Qy[i_part] = NAFFlib.get_tune(y_tbt[:, i_part])

Qxy_fp = np.zeros_like(xy_norm)

Qxy_fp[:, :, 0] = np.reshape(Qx, Qxy_fp[:, :, 0].shape)
Qxy_fp[:, :, 1] = np.reshape(Qy, Qxy_fp[:, :, 1].shape)

"""

# Tracking for footprint

for ii in range(xy_norm.shape[0]):
    for jj in range(xy_norm.shape[1]):
        print('FP track %d/%d (%d/%d)'%(ii, xy_norm.shape[0], jj, xy_norm.shape[1]))

        if track_with == 'PySixtrack':
            part = pysixtrack.Particles(**partCO)

            part.px += DpxDpy_wrt_CO[ii, jj, 0]
            part.py += DpxDpy_wrt_CO[ii, jj, 1]

            x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_pysixtrack(line, part, n_turns)
        elif track_with == 'Sixtrack':
            x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_sixtrack(
                    0., DpxDpy_wrt_CO[ii, jj, 0], 0., DpxDpy_wrt_CO[ii, jj, 1], 0., 0., n_turns)
        else:
            raise ValueError('What?!')

        qx = NAFFlib.get_tune(x_tbt)
        qy = NAFFlib.get_tune(y_tbt)

        Qxy_fp[ii, jj, 0] = qx
        Qxy_fp[ii, jj, 1] = qy
"""
plt.close('all')

fig3 = plt.figure(3)
axcoord = fig3.add_subplot(1, 1, 1)
footprint.draw_footprint(xy_norm, axis_object=axcoord)
axcoord.set_xlim(right=np.max(xy_norm[:, :, 0]))
axcoord.set_ylim(top=np.max(xy_norm[:, :, 1]))

fig4 = plt.figure(4)
axFP = fig4.add_subplot(1, 1, 1)
footprint.draw_footprint(Qxy_fp, axis_object=axFP)
# axFP.set_xlim(right=np.max(Qxy_fp[:, :, 0]))
# axFP.set_ylim(top=np.max(Qxy_fp[:, :, 1]))

plt.show()
