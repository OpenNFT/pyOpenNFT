# Function to perform Kalman low-pass filtering and despiking
#
# input:
# kalmTh           - spike-detection threshold
# kalmIn           - input data
# S                - parameter structure
# fPositDerivSpike - counter for spikes with positive derivative
# fNegatDerivSpike - counter for spikes with negative derivative
#
# output:
# kalmOut          - filtered otuput
# S                - parameters structure
# fPositDerivSpike - counter for spikes with positive derivative
# fNegatDerivSpike - counter for spikes with negative derivative
#
# For generic aspects see:
# Koush Y., Zvyagintsev M., Dyck M., Mathiak K.A., Mathiak K. (2012):
# Signal quality and Bayesian signal processing in neurofeedback based on
# real-time fMRI. Neuroimage 59:478-89.
# _______________________________________________________________________
# Copyright (C) 2016-2022 OpenNFT.org
#
# Written by Yury Koush
# Adapted by Nikita Davydov

import numpy as np


def modif_kalman(kalman_threshold, kalm_in, s, flag_pos_deriv_spike, flag_neg_deriv_spike):

    # Preset
    a = 1
    h = 1
    i = 1

    # Kalman filter
    s["x"] = a * s["x"]
    s["P"] = s["P"] * (a ** 2) + s["Q"]
    k = s["P"] * h * np.linalg.pinv(np.array(s["P"] * (h ** 2) + s["R"], ndmin=2))[0][0]
    tmp_x = s["x"]
    tmp_p = s["P"]
    diff = k * (kalm_in - h * s["x"])
    s["x"] = s["x"] + diff
    s["P"] = (i - k * h) * s["P"]

    # spikes identification and correction
    if np.abs(diff) < kalman_threshold:
        kalm_out = h * s["x"]
        flag_neg_deriv_spike = 0
        flag_pos_deriv_spike = 0
    else:
        if diff > 0:
            if flag_pos_deriv_spike < 1:
                kalm_out = h * tmp_x
                s["x"] = tmp_x
                s["P"] = tmp_p
                flag_pos_deriv_spike = flag_pos_deriv_spike + 1
            else:
                kalm_out = h * s["x"]
                flag_pos_deriv_spike = 0
        else:
            if flag_neg_deriv_spike < 1:
                kalm_out = h * tmp_x
                s["x"] = tmp_x
                s["P"] = tmp_p
                flag_neg_deriv_spike = flag_neg_deriv_spike + 1
            else:
                kalm_out = h * s["x"]
                flag_neg_deriv_spike = 0

    return kalm_out, s, flag_pos_deriv_spike, flag_neg_deriv_spike
