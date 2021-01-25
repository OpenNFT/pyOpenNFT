/* 
 * $Id: spm_mapping.h 938 2007-10-12 19:09:31Z john $
 * John Ashburner
 */
 
/* Matlab dependent high level data access and map manipulation routines */

#ifndef _SPM_MAPPING_H_
#define _SPM_MAPPING_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "spm_vol_access.h"


void free_maps(MAPTYPE *maps, int n);

MAPTYPE *get_maps(const pybind11::array ptr, int *n);

void voxdim(MAPTYPE *map, double vdim[3]);

//int get_dtype(const mxArray *ptr);

#endif /* _SPM_MAPPING_H_ */
