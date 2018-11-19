import numpy as np
import os
import sixtracktools


def vectorize_all_coords(Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                         Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                         Dsigma_wrt_CO_m, Ddelta_wrt_CO):

    n_part = None
    for var in [Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                Dsigma_wrt_CO_m, Ddelta_wrt_CO]:
        if hasattr(var, '__iter__'):
            if n_part is None:
                n_part = len(var)
            assert len(var) == n_part

    for var in [Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                Dsigma_wrt_CO_m, Ddelta_wrt_CO]:
        if not hasattr(var, '__iter__'):
            var += np.zeros(n_part)

    return Dx_wrt_CO_m[:], Dpx_wrt_CO_rad[:],\
           Dy_wrt_CO_m[:], Dpy_wrt_CO_rad[:],\
           Dsigma_wrt_CO_m[:;;], Ddelta_wrt_CO[:]



def track_particle_sixtrack(
    Dx_wrt_CO_m, Dpx_wrt_CO_rad,
    Dy_wrt_CO_m, Dpy_wrt_CO_rad,
    Dsigma_wrt_CO_m, Ddelta_wrt_CO, n_turns
):
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
    lines_f3[i_start_ini + 7] = '    %e\n' % (Ddelta_wrt_CO)

    lines_f3[i_start_ini + 2 + 6] = '    %e\n' % (Dx_wrt_CO_m * 1e3)
    lines_f3[i_start_ini + 3 + 6] = '    %e\n' % (Dpx_wrt_CO_rad * 1e3)
    lines_f3[i_start_ini + 4 + 6] = '    %e\n' % (Dy_wrt_CO_m * 1e3)
    lines_f3[i_start_ini + 5 + 6] = '    %e\n' % (Dpy_wrt_CO_rad * 1e3)
    lines_f3[i_start_ini + 6 + 6] = '    %e\n' % (Dsigma_wrt_CO_m * 1e3)
    lines_f3[i_start_ini + 7 + 6] = '    %e\n' % (Ddelta_wrt_CO)

    # Set number of turns
    i_start_tk = None
    for ii, ll in enumerate(lines_f3):
        if ll.startswith('TRACKING PAR'):
            i_start_tk = ii
            break
    temp_list = lines_f3[i_start_tk + 1].split(' ')
    temp_list[0] = '%d' % n_turns
    lines_f3[i_start_tk + 1] = ' '.join(temp_list)

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

    sixdump_part = sixdump_all[1::2]  #

    x_tbt = sixdump_part.x
    px_tbt = sixdump_part.px
    y_tbt = sixdump_part.y
    py_tbt = sixdump_part.py
    sigma_tbt = sixdump_part.sigma
    delta_tbt = sixdump_part.delta

    return x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt


def track_particle_pysixtrack(line, part, Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                              Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                              Dsigma_wrt_CO_m, Ddelta_wrt_CO, n_turns, verbose=False):

    n_part = None

    for var in [Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                Dsigma_wrt_CO_m, Ddelta_wrt_CO]:

        if hasattr(var, '__iter__'):
            if n_part is None:
                n_part = len(var)
            assert len(var) == n_part

    for var, data in zip('x px y py zeta delta'.split(),
                         [Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                          Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                          Dsigma_wrt_CO_m, Ddelta_wrt_CO]):
        if hasattr(data, '__iter__'):
            setattr(part, var, getattr(part, var) + data)
        else:
            setattr(part, var, getattr(part, var) + (np.zeros(n_part) + data))

    x_tbt = []
    px_tbt = []
    y_tbt = []
    py_tbt = []
    sigma_tbt = []
    delta_tbt = []

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
