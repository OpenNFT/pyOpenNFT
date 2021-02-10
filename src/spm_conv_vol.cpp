/*
 * $Id: spm_conv_vol.c 4452 2011-09-02 10:45:26Z guillaume $
 * John Ashburner
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include <math.h>
#include "spm_conv_vol.h"
#include "spm_mapping.h"
#include "spm_datatypes.h"
#define RINT(A) floor((A)+0.5)

static void convxy(double out[], int xdim, int ydim, double filtx[], double filty[], int fxdim, int fydim, int xoff, int yoff, double buff[])
{
    int x,y,k;
    for(y=0; y<ydim; y++)
    {
        for(x=0; x<xdim; x++)
        {
            buff[x] = out[x+y*xdim];
            if (isinf(buff[x]))
                buff[x] = 0.0;
        }
        for(x=0; x<xdim; x++)
        {
            double sum1 = 0.0;
            int fstart, fend;
            fstart = ((x-xoff >= xdim) ? x-xdim-xoff+1 : 0);
            fend = ((x-(xoff+fxdim) < 0) ? x-xoff+1 : fxdim);

            for(k=fstart; k<fend; k++)
                sum1 += buff[x-xoff-k]*filtx[k];
            out[x+y*xdim] = sum1;
        }
    }
    for(x=0; x<xdim; x++)
    {
        for(y=0; y<ydim; y++)
            buff[y] = out[x+y*xdim];

        for(y=0; y<ydim; y++)
        {
            double sum1 = 0.0;
            int fstart, fend;
            fstart = ((y-yoff >= ydim) ? y-ydim-yoff+1 : 0);
            fend = ((y-(yoff+fydim) < 0) ? y-yoff+1 : fydim);

            for(k=fstart; k<fend; k++)
                sum1 += buff[y-yoff-k]*filty[k];
            out[y*xdim+x] = sum1;
        }
    }
}


static int convxyz(MAPTYPE *vol, double filtx[], double filty[], double filtz[],
    int fxdim, int fydim, int fzdim, int xoff, int yoff, int zoff,
    double *oVol, int dtype)
//    DoubleArray *wplane_args[3],
{
    double *tmp, *buff, **sortedv;
    int xy, z, k, fstart, fend, startz, endz;
    static double mat[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    int xdim, ydim, zdim;

    xdim = vol->dim[0];
    ydim = vol->dim[1];
    zdim = vol->dim[2];

    tmp = (double *)malloc(xdim*ydim*fzdim*sizeof(double));
    buff = (double *)malloc(((ydim>xdim) ? ydim : xdim)*sizeof(double));
    sortedv = (double **)malloc(fzdim*sizeof(double *));

    startz = ((fzdim+zoff-1<0) ? fzdim+zoff-1 : 0);
    endz   = zdim+fzdim+zoff-1;

    for (z=startz; z<endz; z++)
    {
        double sum2 = 0.0;

        if (z >= 0 && z<zdim)
        {
            mat[14] = z+1.0;
            slice(mat, tmp+((z%fzdim)*xdim*ydim), vol->dim[0], vol->dim[1], vol, 0, 0);
            convxy(tmp+((z%fzdim)*xdim*ydim),xdim, ydim,
                filtx, filty, fxdim, fydim, xoff, yoff, buff);
        }
        if (z-fzdim-zoff+1>=0 && z-fzdim-zoff+1<zdim)
        {
            fstart = ((z >= zdim) ? z-zdim+1 : 0);
            fend = ((z-fzdim < 0) ? z+1 : fzdim);

            for(k=0; k<fzdim; k++)
            {
                int z1 = (((z-k)%fzdim)+fzdim)%fzdim;
                sortedv[k] = &(tmp[z1*xdim*ydim]);
            }

            for(k=fstart, sum2=0.0; k<fend; k++)
                sum2 += filtz[k];

            if (!oVol || dtype == SPM_DOUBLE)
            {
                double *obuf;
//                if (!oVol)
//                    obuf = mxGetPr(wplane_args[1]);
//                else
                obuf = &oVol[(z-fzdim-zoff+1)*ydim*xdim];
                if (sum2)
                {
                    for(xy=0; xy<xdim*ydim; xy++)
                    {
                        double sum1=0.0;
                        for(k=fstart; k<fend; k++)
                            sum1 += filtz[k]*sortedv[k][xy];

                        obuf[xy] = sum1/sum2;
                    }
                }
                else
                    for(xy=0; xy<xdim*ydim; xy++)
                        obuf[xy] = 0.0;

//                if (!oVol)
//                {
//                    mxGetPr(wplane_args[2])[0] = z-fzdim-zoff+2.0;
//                    mexCallMATLAB(0, NULL, 3, wplane_args, "spm_write_plane");
//                }
            }
            else
            {
                double tmp;
                if (dtype == SPM_FLOAT)
                {
                    float *obuf;
                    obuf = (float *)oVol;
                    obuf = &obuf[(z-fzdim-zoff+1)*ydim*xdim];
                    if (sum2)
                    {
                        for(xy=0; xy<xdim*ydim; xy++)
                        {
                            double sum1=0.0;
                            for(k=fstart; k<fend; k++)
                                sum1 += filtz[k]*sortedv[k][xy];

                            obuf[xy] = (float)(sum1/sum2);
                        }
                    }
                    else
                        for(xy=0; xy<xdim*ydim; xy++)
                            obuf[xy] = 0.0;
                }
                else if (dtype == SPM_SIGNED_INT)
                {
                    int *obuf;
                    obuf = (int *)oVol;
                    obuf = &obuf[(z-fzdim-zoff+1)*ydim*xdim];
                    if (sum2)
                    {
                        for(xy=0; xy<xdim*ydim; xy++)
                        {
                            double sum1=0.0;
                            for(k=fstart; k<fend; k++)
                                sum1 += filtz[k]*sortedv[k][xy];
                            tmp = sum1/sum2;
                            if (tmp< -2147483648.0) tmp = -2147483648.0;
                            else if (tmp>2147483647.0) tmp = 2147483647.0;
                            obuf[xy] = RINT(tmp);
                        }
                    }
                    else
                        for(xy=0; xy<xdim*ydim; xy++)
                            obuf[xy] = 0;
                }
                else if (dtype == SPM_UNSIGNED_INT)
                {
                    unsigned int *obuf;
                    obuf = (unsigned int *)oVol;
                    obuf = &obuf[(z-fzdim-zoff+1)*ydim*xdim];
                    if (sum2)
                    {
                        for(xy=0; xy<xdim*ydim; xy++)
                        {
                            double sum1=0.0;
                            for(k=fstart; k<fend; k++)
                                sum1 += filtz[k]*sortedv[k][xy];
                            tmp = sum1/sum2;
                            if (tmp<0) tmp = 0.0;
                            else if (tmp>4294967295.0) tmp = 4294967295.0;
                            obuf[xy] = RINT(tmp);
                        }
                    }
                    else
                        for(xy=0; xy<xdim*ydim; xy++)
                            obuf[xy] = 0;
                }
                else if (dtype == SPM_SIGNED_SHORT)
                {
                    double tmp;
                    short *obuf;
                    obuf = (short *)oVol;
                    obuf = &obuf[(z-fzdim-zoff+1)*ydim*xdim];
                    if (sum2)
                    {
                        for(xy=0; xy<xdim*ydim; xy++)
                        {
                            double sum1=0.0;
                            for(k=fstart; k<fend; k++)
                                sum1 += filtz[k]*sortedv[k][xy];
                            tmp = sum1/sum2;
                            if (tmp<-32768) tmp = -32768;
                            else if (tmp>32767) tmp = 32767;
                            obuf[xy] = RINT(tmp);
                        }
                    }
                    else
                        for(xy=0; xy<xdim*ydim; xy++)
                            obuf[xy] = 0;
                }
                else if (dtype == SPM_UNSIGNED_SHORT)
                {
                    unsigned short *obuf;
                    obuf = (unsigned short *)oVol;
                    obuf = &obuf[(z-fzdim-zoff+1)*ydim*xdim];
                    if (sum2)
                    {
                        for(xy=0; xy<xdim*ydim; xy++)
                        {
                            double sum1=0.0;
                            for(k=fstart; k<fend; k++)
                                sum1 += filtz[k]*sortedv[k][xy];
                            tmp = sum1/sum2;
                            if (tmp<0) tmp = 0;
                            else if (tmp>65535) tmp = 65535;
                            obuf[xy] = RINT(tmp);
                        }
                    }
                    else
                        for(xy=0; xy<xdim*ydim; xy++)
                            obuf[xy] = 0;
                }
                else if (dtype == SPM_SIGNED_CHAR)
                {
                    char *obuf;
                    obuf = (char *)oVol;
                    obuf = &obuf[(z-fzdim-zoff+1)*ydim*xdim];
                    if (sum2)
                    {
                        for(xy=0; xy<xdim*ydim; xy++)
                        {
                            double sum1=0.0;
                            for(k=fstart; k<fend; k++)
                                sum1 += filtz[k]*sortedv[k][xy];
                            tmp = sum1/sum2;
                            if (tmp<-128) tmp = -128;
                            else if (tmp>127) tmp = 127;
                            obuf[xy] = RINT(tmp);
                        }
                    }
                    else
                        for(xy=0; xy<xdim*ydim; xy++)
                            obuf[xy] = 0;
                }
                else if (dtype == SPM_UNSIGNED_CHAR)
                {
                    unsigned char *obuf;
                    obuf = (unsigned char *)oVol;
                    obuf = &obuf[(z-fzdim-zoff+1)*ydim*xdim];
                    if (sum2)
                    {
                        for(xy=0; xy<xdim*ydim; xy++)
                        {
                            double sum1=0.0;
                            for(k=fstart; k<fend; k++)
                                sum1 += filtz[k]*sortedv[k][xy];
                            tmp = sum1/sum2;
                            if (tmp<0) tmp = 0;
                            else if (tmp>255) tmp = 255;
                            obuf[xy] = RINT(tmp);
                        }
                    }
                    else
                        for(xy=0; xy<xdim*ydim; xy++)
                            obuf[xy] = 0;
                }
                else printf("\033[0;31m SPM ERROR: Unknown output datatype. \033[0m");
            }
        }
    }
    free((char *)tmp);
    free((char *)buff);
    free((char *)sortedv);
    return(0);
}



DoubleArray spm_conv_vol(DoubleArray Coef, DoubleArray V, DoubleArray x, DoubleArray y, DoubleArray z, DoubleArray off)
{
    MAPTYPE *map;
    int k, dtype = SPM_DOUBLE;
    double *offsets, *oVol = NULL;
//    DoubleArray *wplane_args[3];

    map=get_maps(Coef, &k);
    if (k!=1)
    {
        free_maps(map, k);
        printf("\033[0;31m SPM ERROR: Too many images to smooth at once . \033[0m");
    }

    py::buffer_info V_info = V.request();
    oVol = static_cast<double *>(V_info.ptr);
    dtype = SPM_DOUBLE;

    py::buffer_info off_info = off.request();
    if (off_info.shape[0]*off_info.shape[1] != 3)
    {
        free_maps(map, 1);
        printf("\033[0;31m SPM ERROR: Offsets must have three values. \033[0m");
    }
    offsets = static_cast<double *>(off_info.ptr);

    py::buffer_info x_info = x.request();
    py::buffer_info y_info = y.request();
    py::buffer_info z_info = z.request();

    if (convxyz(map,
        static_cast<double *>(x_info.ptr),
        static_cast<double *>(y_info.ptr),
        static_cast<double *>(z_info.ptr),
        x_info.shape[0]*x_info.shape[1],
        y_info.shape[0]*y_info.shape[1],
        z_info.shape[0]*z_info.shape[1],
        (int)floor(offsets[0]), (int)floor(offsets[1]), (int)floor(offsets[2]),
        oVol,dtype) != 0)
//         wplane_args,
    {
        free_maps(map, 1);
        printf("\033[0;31m SPM ERROR: Error writing data. \033[0m");
    }
    free_maps(map, 1);
//    if (!oVol)
//    {
//        mxDestroyArray(wplane_args[1]);
//        mxDestroyArray(wplane_args[2]);
//    }

    return V;

}
