
#ifndef _SPM_BSPLINS_H_
#define _SPM_BSPLINS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

namespace py = pybind11;

py::array_t<double> spm_bsplins(py::array C, py::array y1, py::array y2, py::array y3, py::array d);

#endif // _SPM_BSPLINS_H_
