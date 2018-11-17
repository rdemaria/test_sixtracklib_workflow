import pickle
import pysixtrack
import numpy as np
import NAFFlib
import helpers as hp
import footprint
import matplotlib.pyplot as plt

track_with = 'PySixtrack'
track_with = 'Sixtrack'

n_turns = 100

with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

with open('DpxDpy_for_footprint.pkl', 'rb') as fid:
    temp_data = pickle.load(fid)

xy_norm = temp_data['xy_norm']
DpxDpy_wrt_CO = temp_data['DpxDpy_wrt_CO']

# Tracking for footprint
Qxy_fp = np.zeros_like(xy_norm)
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
