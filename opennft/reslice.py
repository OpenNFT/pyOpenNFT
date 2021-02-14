
import numpy as np
import pyspm as spm
import sys

class Reslicing():

    def __init__(self, parent=None):
        super()

    def spm_reslice(self, P, flags):

        if int(flags['mask']) or int(flags['mean']):
            temp_x1 = np.transpose(np.array(range(1,P[0]['dim'][0]+1), ndmin=2))
            x1 = np.tile(temp_x1,(1,P[0]['dim'][1]))
            temp_x2 = np.transpose(np.array(range(1,P[0]['dim'][1]+1), ndmin=2))
            x2 = np.transpose(np.tile(temp_x2,(1,P[0]['dim'][1])))

            if int(flags['mean']):
                Count = np.zeros((int(P[0]['dim'][0]),int(P[0]['dim'][1]),int(P[0]['dim'][2])))
                Integral = np.zeros((int(P[0]['dim'][0]),int(P[0]['dim'][1]),int(P[0]['dim'][2])))

            if int(flags['mask']):
                msk = [[] for i in range(P[0]['dim'][2])] #[None]*P['dim'][0][2]

            for x3 in range(0,P[0]['dim'][2]):
                tmp = np.zeros((P[0]['dim'][0],P[0]['dim'][1]))
                for i in range(0,len(P)):
                    try:
                        tmpDivision = np.linalg.solve(P[0]['mat'], P[i]['mat'])
                    except np.linalg.LinAlgError as err:
                        # TODO: Something
                        print(err)
                        raise
                    temp_tmp, y1, y2, y3 = self.getmask(np.linalg.inv(tmpDivision),x1,x2,x3+1,P[i]['dim'][0:3],flags['wrap'])
                    tmp = tmp + temp_tmp

                if int(flags['mask']):
                    msk[x3] = np.argwhere(tmp != len(P))

                if int(flags['mean']):
                    Count[:,:,x3] = tmp

        nread = len(P)
        if not int(flags['mean']):
            if int(flags['which']) == 1:
                nread = nread - 1

            if int(flags['which']) == 0:
                nread = 0

        x1, x2 = np.mgrid[1:P[0]['dim'][0]+1,1:P[0]['dim'][1]+1]
        nread = 0
        tempD = np.array([1, 1, 1], ndmin=2)*int(flags['interp'])
        d = np.hstack((tempD.T, np.array(flags['wrap'],ndmin=2)))

        for i in range(1,len(P)): # range(0,P.size)

            if (i>1 and int(flags['which'])==1) or int(flags['which'])==2:
                write_vol = 1
            else:
                write_vol=0
            if write_vol or int(flags['mean']):
                read_vol = 1
            else:
                read_vol = 0

            if read_vol:

                v = np.zeros(P[0]['dim'])
                for x3 in range(0,P[0]['dim'][2]):
                    try:
                        tmpDivision = np.linalg.solve(P[0]['mat'],P[i]['mat'])
                    except np.linalg.LinAlgError as err:
                        # TODO: Something
                        print(err)
                        raise
                    tmp, y1, y2, y3 = self.getmask(np.linalg.inv(tmpDivision),x1,x2,x3+1,P[i]['dim'][0:3],flags['wrap'])
                    v[:,:,x3] = spm.bsplins(P[i]['C'], y1, y2, y3, d)

                    if int(flags['mean']):
                        Integral[:, :, x3] += self.nan2zero(v[:,:,:x3])

                    if int(flags['mask']):
                        tmp = v[:, :, x3]
                        tmp[msk[x3]] = 0
                        v[:, :, x3] = tmp

                if write_vol:
                    V0 = v

                nread = nread + 1

        return V0


    def kspace3d(self, v, M):

        S0, S1, S2, S3 = self.shear_decomp(M)

        d = np.append(np.asarray(v.shape),[1, 1, 1]).astype(int)
        g = (2 ** np.ceil(np.log2(d))).astype(int)
        if (g != d).any():
            tmp = v
            v = np.zeros(g)
            v[1:d[0], 1:[1], 1:d[3]] = tmp

        # XY-shear
        tmp1_1 = np.append(np.array(range(0,int((g[2]-1)/2)+1)), 0)
        tmp1_2 = np.append(tmp1_1, np.array(range(int((-g[2]/2))+1,0)))
        tmp1 = -1j * 2*np.pi * tmp1_2 / g[2]
        for j in range(1,g[1]+1):
            tempT = np.exp(np.transpose(j*S3[2][1] + S3[2][0]*np.array(range(1,g[0]+1)) + S3[2,3])*tmp1)
            t = tempT.reshape(g[0], 1, g[2])
            v[:,j-1,:] = np.real(np.fft.ifftn(np.fft.fftn(v[:,j-1,:], axis=2)*t,axis=2))

        # XZ-shear
        tmp1_1 = np.append(np.array(range(0, int((g[1] - 1) / 2) + 1)), 0)
        tmp1_2 = np.append(tmp1_1, np.array(range(int((-g[1] / 2)) + 1, 0)))
        tmp1 = -1j * 2 * np.pi * tmp1_2 / g[1]
        for k in range(1, g[2]+1):
            t = np.exp(np.transpose(k * S2[2][1] + S2[2][0] * np.array(range(1, g[0] + 1)) + S2[2, 3]) * tmp1)
            v[:, :, k-1] = np.real(np.fft.ifftn(np.fft.fftn(v[:, :, k-1], axis=1) * t, axis=1))

        # YZ-shear
        tmp1_1 = np.append(np.array(range(0, int((g[2] - 1) / 2) + 1)), 0)
        tmp1_2 = np.append(tmp1_1, np.array(range(int((-g[2] / 2)) + 1, 0)))
        tmp1 = -1j * 2 * np.pi * tmp1_2 / g[2]
        for k in range(1, g[2]+1):
            t = np.exp(tmp1.T*(k * S1[0][2] + S1[0][1] * np.array(range(1, g[1] + 1)) + S1[0, 3]))
            v[:, :, k-1] = np.real(np.fft.ifftn(np.fft.fftn(v[:, :, k-1], axis=0) * t, axis=0))

        # XY-shear
        tmp1_1 = np.append(np.array(range(0,int((g[2]-1)/2)+1)), 0)
        tmp1_2 = np.append(tmp1_1, np.array(range(int((-g[2]/2))+1,0)))
        tmp1 = -1j * 2*np.pi * tmp1_2 / g[2]
        for j in range(1,g[1]+1):
            tempT = np.exp(np.transpose(j*S0[2][1] + S0[2][0]*np.array(range(1,g[0]+1)) + S0[2,3])*tmp1)
            t = tempT.reshape(g[0], 1, g[2])
            v[:,j-1,:] = np.real(np.fft.ifftn(np.fft.fftn(v[:,j-1,:], axis=2)*t,axis=2))

        if (g != d).any():
            v = v[1:d[0], 1:d[1], 1:d[2]]

        return v

    def shear_decomp(self, A):

        A0 = A[0:3][0:3]
        if (np.abs(np.linalg.svd(A0)-1) > 1e-7).any():
            # TODO: error message class needed
            # error('Can''t decompose matrix')
            o=1

        t = A0[1][2]
        if t==0:
            t=sys.float_info.epsilon

        tmp = np.array([[A0[0][1], A0[0][2]],[A0[1][1], A0[1][2]]]).T
        a0 = np.linalg.pinv(tmp)@np.array([(A0[2][1]-(A0[1][1]-1)/t), (A0[2][2]-1)], ndmin=2).T
        S0 = np.array([[1, 0, 0], [0, 1, 0], [a0[0], a0[1], 1]])
        A1 = S0 / A0

        tmp = np.array([[A0[1][1], A0[1][2]],[A0[2][1], A0[2][2]]]).T
        a1 = tmp@np.array([A1[0][1], A1[0][2]], ndmin=2).T
        S1 = [[1, a1[0], a1[1]],[0, 1, 0],[0, 0, 1]]
        A2 = S1 / A1

        tmp = np.array([[A0[0][0], A0[0][2]],[A0[0][2], A0[2][2]]]).T
        a2 = tmp@np.array([A2[1][0], A2[1][2]], ndmin=2).T
        S2 = [[1, 0, 0], [a2[0], 1, a2[1]], [0, 0, 1]]
        A3 = S2 / A2

        tmp = np.array([[A0[0][0], A0[0][1]],[A0[0][1], A0[1][1]]]).T
        a3 = tmp@np.array([A3[2][0], A2[2][1]], ndmin=2).T
        S3 = [[1, 0, 0],[0, 1, 0],[a3[0], a3[1], 1]]

        s3 = A[2][3] - a0[0]*A[0][3] - a0[1]*A[1][3]
        s1 = A[0][3] - a1[0]*A[1][3]
        s2 = A[1][3]
        S0 = np.vstack((np.hstack((S0, np.array([0, 0, s3], ndmin=2).T)),np.array([0, 0, 0, 1], ndmin=2)))
        S1 = np.vstack((np.hstack((S1, np.array([s1, 0, 0], ndmin=2).T)),np.array([0, 0, 0, 1], ndmin=2)))
        S2 = np.vstack((np.hstack((S2, np.array([0, s2, 0], ndmin=2).T)),np.array([0, 0, 0, 1], ndmin=2)))
        S3 = np.vstack((np.hstack((S3, np.array([0, 0, 0], ndmin=2).T)),np.array([0, 0, 0, 1], ndmin=2)))

        return {'S0' : S0, 'S1' : S1, 'S2': S2, 'S3': S3}

    def getmask(self, M, x1, x2, x3, dim, wrp):

        tiny = 5e-2 # From spm_vol_utils.c
        y1 = M[0][0]*x1 + M[0][1]*x2 + (M[0][2]*x3 + M[0][3])
        y2 = M[1][0]*x1 + M[1][1]*x2 + (M[1][2]*x3 + M[1][3])
        y3 = M[2][0]*x1 + M[2][1]*x2 + (M[2][2]*x3 + M[2][3])
        Mask = np.array([True]*y1.size).reshape(y1.shape)
        if wrp[0] != 0: Mask = Mask and (y1 >= (1-tiny) and y1 <= (dim[0]+tiny))
        if wrp[1] != 0: Mask = Mask and (y1 >= (1-tiny) and y1 <= (dim[1]+tiny))
        if wrp[2] != 0: Mask = Mask and (y1 >= (1-tiny) and y1 <= (dim[2]+tiny))

        return Mask, y1, y2, y3

    def nan2zero(self, vi):
        return np.nan_to_num(vi, copy=True, nan=0, posinf=0, neginf=0)