import numpy as np

class Utils():

    def __init__(self, parent=None):
        super()

    def img2Dvol3D(self, img2D, slNrImg2DdimX, slNrImg2DdimY, dim3D):

        sl = 0
        vol3D = np.zeros(dim3D)
        for sy in range(0,slNrImg2DdimY):
            for sx in range(0, slNrImg2DdimX):
                if sl>dim3D[2]:
                    break
                else:
                    vol3D[:,:,sl] = img2D[sy * dim3D[0] : (sy+1) * dim3D[0],
                                          sx * dim3D[0] : (sx+1) * dim3D[0]]
                vol3D[:,:,sl] = np.rot90(vol3D[:,:,sl],3)
                sl = sl+1

        return vol3D

    def getMosaicDim(self, dim3D):

        slNrImg2DdimX = round(np.sqrt(dim3D[2]))
        tmpDim = dim3D[2] - slNrImg2DdimX ** 2

        if tmpDim == 0:
            slNrImg2DdimY = slNrImg2DdimX
        else:
            if tmpDim > 0:
                slNrImg2DdimY = slNrImg2DdimX
                slNrImg2DdimX = slNrImg2DdimX + 1
            else:
                slNrImg2DdimX = slNrImg2DdimX
                slNrImg2DdimY = slNrImg2DdimX

        img2DdimX = slNrImg2DdimX * dim3D[0]
        img2DdimY = slNrImg2DdimY * dim3D[0]

        return slNrImg2DdimX, slNrImg2DdimY, img2DdimX, img2DdimY