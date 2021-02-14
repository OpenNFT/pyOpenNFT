
#ifndef _SPM_BSPLINS_H_
#define _SPM_BSPLINS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

namespace py = pybind11;

using DoubleArray = py::array_t<double>;
using Dictionary = py::dict;

DoubleArray spm_bsplins(DoubleArray C, DoubleArray y1, DoubleArray y2, DoubleArray y3, DoubleArray d);
std::tuple<DoubleArray, DoubleArray, DoubleArray, DoubleArray> spm_bsplins_multi(DoubleArray C, DoubleArray y1, DoubleArray y2, DoubleArray y3, DoubleArray d);

#endif // _SPM_BSPLINS_H_
