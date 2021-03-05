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

    def spm_matrix(self, P):

        if P.size == 3:
            A = np.eye(4)
            A[0:3,3] = P[:]
            return A

        q = np.array([0,0,0,0,0,0,1,1,1,0,0,0])
        P = np.append(P, q[P.size:12])

        T = np.array([[1, 0, 0, P[0]],
                      [0, 1, 0, P[1]],
                      [0, 0, 1, P[2]],
                      [0, 0, 0, 1]])

        R1 = np.array([[1, 0, 0, 0],
                      [0, np.cos(P[3]), np.sin(P[3]), 0],
                      [0, -np.sin(P[3]), np.cos(P[3]), 0],
                      [0, 0, 0, 1]])

        R2 = np.array([[np.cos(P[4]), 0, np.sin(P[4]), 0],
                      [0, 1, 0, 0],
                      [-np.sin(P[4]), 0, np.cos(P[4]), 0],
                      [0, 0, 0, 1]])

        R3 = np.array([[np.cos(P[5]), np.sin(P[5]), 0, 0],
                      [-np.sin(P[5]), np.cos(P[5]), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        R = R1@R2@R3

        Z = np.array([[P[6], 0, 0, 0],
                      [0, P[7], 0, 0],
                      [0, 0, P[8], 0],
                      [0, 0, 0, 1]])

        S = np.array([[1, P[9], P[10], 0],
                      [0, 1, P[11], 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        A = T@R@Z@S

        return A

    def spm_imatrix(self, M):

        R = M[0:3, 0:3]
        C = np.linalg.cholesky(R.T@R)
        P = np.append(np.append(np.append(M[0:3,3].T,np.zeros(3,)),np.diag(C).T),np.zeros(3,))
        if np.linalg.det(R)<0:
            P[6]=-P[6]

        C = np.linalg.solve(np.diag(np.diag(C)),C)
        P[9:12] = C.flatten()[[3, 6, 7]]
        R0 = self.spm_matrix(np.append(np.zeros(6,), P[6:12]))
        R0 = R0[0:3,0:3]
        R1 = np.linalg.solve(R0.T,R.T).T

        rang = lambda x: np.minimum(np.maximum(x,-1),1)

        P[4] = np.arcsin(rang(R1[0,2]))
        if (np.abs(P[4])-np.pi/2)**2 < 1e-9:
            P[3] = 0
            P[5] = np.arctan2(-rang(R1[1,0]), rang( -R1[2,0]/R1[0,2] ))
        else:
            c = np.cos(P[4])
            P[3] = np.arctan2(rang(R1[1,2]/c), rang(R1[2,2]/c))
            P[5] = np.arctan2(rang(R1[0,1]/c), rang(R1[0,0]/c))

        return P