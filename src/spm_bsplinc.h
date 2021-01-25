
#ifndef _SPM_BSPLINC_H_
#define _SPM_BSPLINC_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include "spm_mapping.h"

namespace py = pybind11;

py::array_t<double> spm_bsplinc(py::array v, py::array C, py::array splDgr);

#endif // _SPM_BSPLINC_H_
