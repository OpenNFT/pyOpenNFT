#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

namespace py = pybind11;

using DoubleArray = py::array_t<double>;

void spm_slice_vol(DoubleArray vol, DoubleArray out_img, DoubleArray affine_matrix, DoubleArray hld);
