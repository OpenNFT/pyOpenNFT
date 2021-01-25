
#include <pybind11/pybind11.h>

#include "spm_bsplins.h"
#include "spm_bsplinc.h"


PYBIND11_MODULE(pyspm, m) {
    m.doc() = R"pbdoc(
        Python adaptation of SPM for OpenNFT project
        --------------------------------------------

        .. codeauthor:: Nikita Davydov and OpenNFT Team

        .. currentmodule:: pyspm
        .. autosummary::
           :toctree: _generate

           bsplins
           bsplinc
    )pbdoc";

    m.def("bsplins", &spm_bsplins);
    m.def("bsplinc", &spm_bsplinc);
}
