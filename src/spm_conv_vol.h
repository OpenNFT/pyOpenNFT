#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

namespace py = pybind11;

using DoubleArray = py::array_t<double>;

DoubleArray spm_conv_vol(DoubleArray Coef, DoubleArray V, DoubleArray x, DoubleArray y, DoubleArray z, DoubleArray off);
