import sixtracktools
import numpy as np
import matplotlib.pyplot as plt
import os

wfold = 'temp_trackfun'

Dx_wrt_CO_m = 1e-4
Dpx_wrt_CO_mrad = 1e-5
Dy_wrt_CO_m = 2e-4
Dpy_wrt_CO_mrad = 2e-5
Dsigma_wrt_CO_m = 1e-2
Ddelta_wrt_CO = 1e-5

n_turns = 100

if not os.path.exists(wfold):
    os.mkdir(wfold)

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

lines_f3[i_start_ini + 2 + 6] = '    %e\n'%(Dx_wrt_CO_m * 1e3)
lines_f3[i_start_ini + 3 + 6] = '    %e\n'%(Dpx_wrt_CO_mrad * 1e3)
lines_f3[i_start_ini + 4 + 6] = '    %e\n'%(Dy_wrt_CO_m * 1e3)
lines_f3[i_start_ini + 5 + 6] = '    %e\n'%(Dpy_wrt_CO_mrad * 1e3)
lines_f3[i_start_ini + 6 + 6] = '    %e\n'%(Dsigma_wrt_CO_m * 1e3)
lines_f3[i_start_ini + 7 + 6] = '    %e\n'%(Ddelta_wrt_CO)


# Set number of turns
i_start_tk = None
for ii, ll in enumerate(lines_f3):
    if ll.startswith('TRACKING PAR'):
        i_start_tk = ii
        break
temp_list = lines_f3[i_start_tk + 1].split(' ')
temp_list[0] = '%d'%n_turns
lines_f3[i_start_tk + 1] = ' '.join(temp_list)

with open(wfold + '/fort.3', 'w') as fid:
    fid.writelines(lines_f3)


prrrr

# Load sixtrack tracking data
sixdump_all = sixtracktools.SixDump101('res_part/dumpg.dat')

sixdump_CO = sixdump_all[::2]   # Particle on CO
sixdump_part = sixdump_all[1::2]  #

plt.close('all')
fig1 = plt.figure(1)
axx = fig1.add_subplot(2, 1, 1)
axy = fig1.add_subplot(2, 1, 2, sharex=axx)

axx.plot(sixdump_CO.x, 'b')
axx.plot(sixdump_part.x, 'r')
plt.show()
