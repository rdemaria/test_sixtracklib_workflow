import pickle
import os
import pysixtrack
import numpy as np
import NAFFlib
import helpers as hp
import sixtracktools
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

# PySixtracK

part = pysixtrack.Particles(**partCO)

part.px += DpxDpy_wrt_CO[:, :, 0].flatten()
part.py += DpxDpy_wrt_CO[:, :, 1].flatten()


for name, etype, ele in line:
    ele.track(part)

# SixTrack
Dx_wrt_CO_m, Dpx_wrt_CO_rad,\
    Dy_wrt_CO_m, Dpy_wrt_CO_rad,\
    Dsigma_wrt_CO_m, Ddelta_wrt_CO = hp.vectorize_all_coords(
        Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 0].flatten(),
        Dy_wrt_CO_m=0., Dpy_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 1].flatten(),
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=0.)

n_part = len(Dx_wrt_CO_m)

wfold = 'temp_trackfun'

if not os.path.exists(wfold):
    os.mkdir(wfold)

os.system('cp fort.* %s' % wfold)

with open('fort.3', 'r') as fid:
    lines_f3 = fid.readlines()

# Set initial coordinates
i_start_ini = None
for ii, ll in enumerate(lines_f3):
    if ll.startswith('INITIAL COO'):
        i_start_ini = ii
        break

lines_f3[i_start_ini + 2] = '    0.\n'
lines_f3[i_start_ini + 3] = '    0.\n'
lines_f3[i_start_ini + 4] = '    0.\n'
lines_f3[i_start_ini + 5] = '    0.\n'
lines_f3[i_start_ini + 6] = '    0.\n'
lines_f3[i_start_ini + 7] = '    0.\n'

lines_f3[i_start_ini + 2 + 6] = '    0.\n'
lines_f3[i_start_ini + 3 + 6] = '    0.\n'
lines_f3[i_start_ini + 4 + 6] = '    0.\n'
lines_f3[i_start_ini + 5 + 6] = '    0.\n'
lines_f3[i_start_ini + 6 + 6] = '    0.\n'
lines_f3[i_start_ini + 7 + 6] = '    0.\n'


lines_f13 = []

temp_part = pysixtrack.Particles(**partCO)

for i_part in range(n_part):
    lines_f13.append('%e\n' % ((Dx_wrt_CO_m[i_part] + temp_part.x) * 1e3))
    lines_f13.append('%e\n' % ((Dpx_wrt_CO_rad[i_part] + temp_part.px) * temp_part.rpp * 1e3))
    lines_f13.append('%e\n' % ((Dy_wrt_CO_m[i_part] + temp_part.y) * 1e3))
    lines_f13.append('%e\n' % ((Dpy_wrt_CO_rad[i_part] + temp_part.py) * temp_part.rpp * 1e3))
    lines_f13.append('%e\n' % ((Dsigma_wrt_CO_m[i_part] + temp_part.sigma) * 1e3))
    lines_f13.append('%e\n' % ((Ddelta_wrt_CO[i_part] + temp_part.delta)))
    if i_part % 2 == 1:
        lines_f13.append(lines_f3[i_start_ini + 7 + 6 + 1].replace(' ', ''))
        lines_f13.append(lines_f3[i_start_ini + 7 + 6 + 2].replace(' ', ''))
        lines_f13.append(lines_f3[i_start_ini + 7 + 6 + 3].replace(' ', ''))

with open(wfold + '/fort.13', 'w') as fid:
    fid.writelines(lines_f13)

if np.mod(n_part, 2) != 0:
    raise ValueError('SixTrack does not like this!')

i_start_tk = None
for ii, ll in enumerate(lines_f3):
    if ll.startswith('TRACKING PAR'):
        i_start_tk = ii
        break
# Set number of turns and number of particles
temp_list = lines_f3[i_start_tk + 1].split(' ')
temp_list[0] = '%d' % n_turns
temp_list[2] = '%d' % (n_part / 2)
lines_f3[i_start_tk + 1] = ' '.join(temp_list)
# Set number of idfor = 2
temp_list = lines_f3[i_start_tk + 2].split(' ')
temp_list[2] = '2'
lines_f3[i_start_tk + 2] = ' '.join(temp_list)

# Setup turn-by-turn dump
i_start_dp = None
for ii, ll in enumerate(lines_f3):
    if ll.startswith('DUMP'):
        i_start_dp = ii
        break

lines_f3[i_start_dp + 1] = 'StartDUMP 1 664 101 dumtemp.dat\n'

with open(wfold + '/fort.3', 'w') as fid:
    fid.writelines(lines_f3)

os.system('./runsix_trackfun')

# Load sixtrack tracking data
sixdump_all = sixtracktools.SixDump101('%s/dumtemp.dat' % wfold)


x_tbt = np.zeros((n_turns, n_part))
px_tbt = np.zeros((n_turns, n_part))
y_tbt = np.zeros((n_turns, n_part))
py_tbt = np.zeros((n_turns, n_part))
sigma_tbt = np.zeros((n_turns, n_part))
delta_tbt = np.zeros((n_turns, n_part))

for i_part in range(n_part):
    sixdump_part = sixdump_all[i_part::n_part]
    x_tbt[:, i_part] = sixdump_part.x
    px_tbt[:, i_part] = sixdump_part.px
    y_tbt[:, i_part] = sixdump_part.y
    py_tbt[:, i_part] = sixdump_part.py
    sigma_tbt[:, i_part] = sixdump_part.sigma
    delta_tbt[:, i_part] = sixdump_part.delta
