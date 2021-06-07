/*
 * $Id: spm_slice_vol.c 4452 2011-09-02 10:45:26Z guillaume $
 * John Ashburner
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include "spm_mapping.h"
#include "spm_slice_vol.h"

void spm_slice_vol(DoubleArray vol, DoubleArray out_img, DoubleArray affine_matrix, DoubleArray hld)
{
    MAPTYPE *map;
    int m, n, hold, status;
    double *mat, *img, background=0.0;

    map = get_maps(vol, &n);

    py::buffer_info mat_info = affine_matrix.request();
    mat = static_cast<double *>(mat_info.ptr);

    py::buffer_info img_info = out_img.request();
    img = static_cast<double *>(img_info.ptr);
    m = img_info.shape[0];
    n = img_info.shape[1];

    py::buffer_info hold_info = hld.request();
    double *temp = static_cast<double*>(hold_info.ptr);
    hold = (int)temp[0];

    background = temp[1];

    status = slice(mat, img, m, n, map, hold, background);
    free_maps(map, 1);

}
