/*
 * $Id: spm_bsplins.c 4624 2012-01-13 13:27:08Z john $
 * John Ashburner
 */

/*
 * This code is based on that of Philippe Thevenaz, which I took from:
 *  http://bigwww.epfl.ch/algorithms.html
 *
 * It has been substantially modified, so blame me (John Ashburner) if there
 * are any bugs. Many thanks to Philippe Thevenaz for advice with the code.
 *
 * See:
 *  M. Unser.
 *  "Splines: A Perfect Fit for Signal and Image Processing,"
 *  IEEE Signal Processing Magazine, 16(6):22-38 (1999)
 *
 *  P. Thevenaz and T. Blu and M. Unser.
 *  "Interpolation Revisited"
 *  IEEE Transactions on Medical Imaging 19(7):739-758 (2000).
*/

#include <math.h>

#include "bsplines.h"
#include "spm_bsplins.h"


/***************************************************************************************
Loop through data and resample the points
    c   - Volume of B-spline coefficients
    m0,m1,m2    - dimensions of c
    n   - number of points to resample
    x0,x1,x2    - array of co-ordinate to sample
    d   - degree of spline used
    cond    - code determining boundaries to mask at
    bnd - functions for dealing with edges
    f   - resampled data
*/
#define TINY 5e-2

static void fun(double c[], int m0, int m1, int m2,
    int n, double x0[], double x1[], double x2[], int d[],
    int cond, int (*bnd[])(int, int), double f[])
{
    int j;

    for(j=0; j<n; j++)
    {
        if (((cond&1) | (x0[j]>=1-TINY && x0[j]<=m0+TINY)) &&
            ((cond&2) | (x1[j]>=1-TINY && x1[j]<=m1+TINY)) &&
            ((cond&4) | (x2[j]>=1-TINY && x2[j]<=m2+TINY)))
            f[j] = sample3(c, m0, m1, m2, x0[j]-1, x1[j]-1, x2[j]-1, d, bnd);
        else
            f[j] = 0.0;  // FIXME: must be NaN instead of Zero
    }
}


/***************************************************************************************
Loop through data and resample the points and their derivatives
    c   - Volume of B-spline coefficients
    m0,m1,m2    - dimensions of c
    n   - number of points to resample
    x0,x1,x2    - array of co-ordinate to sample
    d   - degrees of splines used
    cond    - code determining boundaries to mask at
    bnd - functions for dealing with edges
    f   - resampled data
    df0, df1, df2   - gradients
*/
//static void dfun(double c[], int m0, int m1, int m2,
//    int n, double x0[], double x1[], double x2[],int d[],
//    int cond, int (*bnd[])(int, int),
//    double f[], double df0[], double df1[], double df2[])
//{
//    int j;
//    double NaN = mxGetNaN();
//
//    for(j=0; j<n; j++)
//    {
//        if (((cond&1) | (x0[j]>=1-TINY && x0[j]<=m0+TINY)) &&
//            ((cond&2) | (x1[j]>=1-TINY && x1[j]<=m1+TINY)) &&
//            ((cond&4) | (x2[j]>=1-TINY && x2[j]<=m2+TINY)))
//            f[j] = dsample3(c, m0,m1,m2, x0[j]-1,x1[j]-1,x2[j]-1, d,
//                &df0[j],&df1[j],&df2[j], bnd);
//        else
//            f[j] = NaN;
//    }
//}


/***************************************************************************************/
DoubleArray spm_bsplins(DoubleArray C, DoubleArray y1, DoubleArray y2, DoubleArray y3, DoubleArray d)
{
    int k, dd[3], nd;
    int m0=1, m1=1, m2=1;
    int cond;
    int (*bnd[3])(int, int);

    /* Usage:
            f = function(c,x0,x1,x2,d)
                c - B-spline coefficients
                x0, x1, x2 - co-ordinates
                d   - B-spline degree
                f   - sampled function
       or:
            [f,df0,df1,df2] = function(c,x0,x1,x2,d)
                c - B-spline coefficients
                x0, x1, x2 - co-ordinates
                d   - B-spline degree
                f   - sampled function
                df0, df1, df2   - sampled derivatives
    */

    py::buffer_info d_info = d.request();
    double *ptr = static_cast<double *>(d_info.ptr);
    int n = d_info.shape[0];
    int m = d_info.shape[1];

    /* Degree of spline */
    for(k=0; k<3; k++)
    {
        dd[k] = floor(ptr[k*m]+0.5);
        if (dd[k]<0 || dd[k]>7)
            printf("\033[0;31m SPM ERROR: Bad spline degree. \033[0m");
    }

    cond = 0;

    for(k=0; k<3; k++)
    {
        bnd[k] = mirror;
    }

    if (n*m == 6)
    {
        for(k=0; k<3; k++)
        {
            if (ptr[k*m+1])
            {
                bnd[k] = wrap;
                cond += 1<<k;
            }
        }
    }

    /* Dimensions of coefficient volume */
    py::buffer_info C_info = C.request();

    if (C_info.ndim>=1) m0 = C_info.shape[0];
    if (C_info.ndim>=2) m1 = C_info.shape[1];
    if (C_info.ndim>=3) m2 = C_info.shape[2];

    /* Dimensions of sampling co-ordinates */
    py::buffer_info y1_info = y1.request();
    py::buffer_info y2_info = y2.request();
    py::buffer_info y3_info = y3.request();

    nd = y1_info.ndim;
    if (y2_info.ndim != nd || y3_info.ndim != nd)
        printf("\033[0;31m SPM ERROR: Incompatible dimensions. \033[0m");
    n = 1;
    for(k=0; k<nd; k++)
    {
        if (y2_info.shape[k] != y1_info.shape[k] || y3_info.shape[k] != y1_info.shape[k])
            printf("\033[0;31m SPM ERROR: Incompatible dimensions. \033[0m");
        n *=y1_info.shape[k];
    }

    /* Sampled data same size as sampling co-ords */
    auto func = DoubleArray(y1_info.size);
    py::buffer_info func_info = func.request();

    /* Pointers to double precision data */
    double *f = static_cast<double *>(func_info.ptr);
    double *c  = static_cast<double *>(C_info.ptr);
    double *x0 = static_cast<double *>(y1_info.ptr);
    double *x1 = static_cast<double *>(y2_info.ptr);
    double *x2 = static_cast<double *>(y3_info.ptr);

//    if (nlhs<=1)
    fun(c, m0, m1, m2, n, x0, x1, x2, dd, cond, bnd, f);
//    else
//    {
//        plhs[1] = mxCreateNumericArray(nd,dims, mxDOUBLE_CLASS, mxREAL);
//        plhs[2] = mxCreateNumericArray(nd,dims, mxDOUBLE_CLASS, mxREAL);
//        plhs[3] = mxCreateNumericArray(nd,dims, mxDOUBLE_CLASS, mxREAL);
//        df0 = mxGetPr(plhs[1]);
//        df1 = mxGetPr(plhs[2]);
//        df2 = mxGetPr(plhs[3]);
//        dfun(c, m0,m1,m2, n, x0,x1,x2, d, cond,bnd, f,df0,df1,df2);
//    }
    func.resize(y1_info.shape);
    return func;
}
