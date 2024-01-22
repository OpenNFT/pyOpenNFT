# Function to scale time-series.
#
# input:
# in_time_series - input cumulative time-series
# ind_vol       - volume(scan) index
# length_sl_wind - sliding window length
# blockLength  - length of the vaselien condition block
# init_lim      - initial time-series limits, see preprSig.m
# tmp_pos_min   - recursive dynamic lower limit
# tmp_pos_max   - recursive dynamic upper limit
#
# output:
# out_data       - scaled current time-series point
# tmp_pos_min    - updated recursive dynamic lower limit
# tmp_pos_max    - updated recursive dynamic upper limit
#
# Note, scaling based on sliding window is disabled using a very big
# number of inclusive points, i.e. it is larger than a run. For algorithms
# involving sliding-window, the simulations and clear hypothesis are
# advised, because of the very high influence on operant conditioning.
#
# For generic aspects see:
# Koush Y., Ashburner J., Prilepin E., Ronald S., Zeidman P., Bibikov S.,
# Scharnowski F., Nikonorov A., Van De Ville D.: OpenNFT: An open-source
# Python/Matlab framework for real-time fMRI neurofeedback training based
# on activity, connectivity and multivariate pattern analysis.(pending)
#
# Koush Y., Zvyagintsev M., Dyck M., Mathiak K.A., Mathiak K. (2012):
# Signal quality and Bayesian signal processing in neurofeedback based on
# real-time fMRI. Neuroimage 59:478-89.
#
# Scharnowski F., Hutton C., Josephs O., Weiskopf N., Rees G. (2012):
# Improving visual perception through neurofeedback. JofNeurosci,
# 32(49), 17830-17841.
# __________________________________________________________________________
# Copyright (C) 2016-2022 OpenNFT.org
#
# Written by Yury Koush
# Adapted by Nikita Davydov

import numpy as np


def scale_time_series(in_time_series, ind_vol, length_sl_wind, init_lim, tmp_pos_min, tmp_pos_max, vect_end_cond, bas_block_length):

    if ind_vol < 20:

        tmp_max = np.max(in_time_series)
        tmp_min = np.min(in_time_series)

    else:

        sorted_time_series = np.array(in_time_series, copy=True)
        sorted_time_series.sort()
        nr_elem = np.round(0.05*len(sorted_time_series))
        tmp_max = np.median(sorted_time_series[int(-1-nr_elem-1):-1])
        tmp_min = np.median(sorted_time_series[0:int(nr_elem+1)])

    # zero bas_block_length stands for auto_rtqa mode
    if bas_block_length > 0:
        if (ind_vol <= bas_block_length) or (ind_vol < length_sl_wind):

            if tmp_max > init_lim:
                tmp_pos_max = tmp_max
            else:
                tmp_pos_max = init_lim
            if tmp_min < -init_lim:
                tmp_pos_min = tmp_min
            else:
                tmp_pos_min = -init_lim

        else:

            chk_max = np.max(in_time_series[ind_vol - length_sl_wind + 1: ind_vol])
            chk_min = np.min(in_time_series[ind_vol - length_sl_wind + 1: ind_vol])

            if (ind_vol > bas_block_length) and not (vect_end_cond[ind_vol] == vect_end_cond[ind_vol-1]):
                if tmp_max > chk_max:
                    tmp_pos_max = chk_max
                else:
                    tmp_pos_max = tmp_max
                if tmp_min < chk_min:
                    tmp_pos_min = chk_min
                else:
                    tmp_pos_min = tmp_min
            else:
                if (in_time_series[ind_vol]) > tmp_pos_max:
                    tmp_pos_max = in_time_series[ind_vol]
                if (in_time_series[ind_vol]) > tmp_pos_min:
                    tmp_pos_min = in_time_series[ind_vol]

    else:

        if tmp_max > init_lim:
            tmp_pos_max = tmp_max
        else:
            tmp_pos_max = init_lim

        if tmp_min < -init_lim:
            tmp_pos_min = tmp_min;
        else:
            tmp_pos_min = -init_lim;

    out_data = (in_time_series[ind_vol]-tmp_pos_min) / (tmp_pos_max - tmp_pos_min)

    if out_data != out_data:
        out_data = 0

    return out_data, tmp_pos_min, tmp_pos_max



























