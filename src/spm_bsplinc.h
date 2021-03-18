
#ifndef _SPM_BSPLINC_H_
#define _SPM_BSPLINC_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include "spm_mapping.h"

namespace py = pybind11;

using DoubleArray = py::array_t<double>;

DoubleArray spm_bsplinc(DoubleArray v, DoubleArray splDgr);

#endif // _SPM_BSPLINC_H_
