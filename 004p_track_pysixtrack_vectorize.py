import pickle
import pysixtrack
import numpy as np
import NAFFlib
import helpers as hp
import footprint
import matplotlib.pyplot as plt

track_with = 'PySixtrack'
track_with = 'Sixtrack'

n_turns = 3

with open('line.pkl', 'rb') as fid:
    line = pickle.load(fid)

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

with open('DpxDpy_for_footprint.pkl', 'rb') as fid:
    temp_data = pickle.load(fid)

xy_norm = temp_data['xy_norm']
DpxDpy_wrt_CO = temp_data['DpxDpy_wrt_CO']

### PySixtracK

part = pysixtrack.Particles(**partCO)

part.px += DpxDpy_wrt_CO[:, :, 0].flatten()
part.py += DpxDpy_wrt_CO[:, :, 1].flatten()


for name, etype, ele in line:
    ele.track(part)
