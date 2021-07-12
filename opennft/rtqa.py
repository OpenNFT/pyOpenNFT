import numpy as np


def snr_vol(rtqa_vol_data, vol):

    mean_non_smooth = rtqa_vol_data['mean']
    m2_non_smooth = rtqa_vol_data['m2']
    snr_non_smooth = rtqa_vol_data['snr']
    iteration = rtqa_vol_data['iteration']

    if mean_non_smooth.size == 0:
        rtqa_vol_data['mean'] = vol.copy()
        rtqa_vol_data['m2'] = np.zeros(vol.shape)
        rtqa_vol_data['snr'] = np.zeros(vol.shape)
        rtqa_vol_data['iteration'] = 1
        return rtqa_vol_data

    iteration += 1
    mean_prev = mean_non_smooth
    mean_non_smooth += (vol - mean_non_smooth) / iteration
    m2_non_smooth += (vol - mean_prev) * (vol - mean_non_smooth)

    variance = m2_non_smooth / (iteration - 1)

    snr_non_smooth = mean_non_smooth / (variance ** 0.5)

    rtqa_vol_data['mean'] = mean_non_smooth
    rtqa_vol_data['m2'] = m2_non_smooth
    rtqa_vol_data['snr'] = snr_non_smooth
    rtqa_vol_data['iteration'] = iteration

    return rtqa_vol_data
