import pickle
import pysixtrack
import numpy as np

n_turns = 10


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


with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

part = pysixtrack.Particles(**partCO)

x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = track_particle(line, part)

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
spx = fig.add_subplot(2, 1, 1)
spy = fig.add_subplot(2, 1, 2, sharex=spx)

spx.plot(x_tbt)
spy.plot(y_tbt)
plt.show()
