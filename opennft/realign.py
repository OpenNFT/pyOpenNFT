import numpy as np
import pyspm as spm
import sys

class Realign():

    def __init__(self, parent=None):
        super()

    def spm_realign(self, P, flags, indVol, indFirstVol, A0, x1, x2, x3, wt, deg, b):

        fNFB = True

        lkp = flags["lkp"]
        # TODO: matlab lkp starts from 1, here we need to check input flags
        if lkp[0] == 1:
            lkp -= 1
        if indVol == indFirstVol:

            skip = np.sqrt( (P["mat"][0][0:2][0:2] ** 2).sum() ) ** (-1) * flags["sep"]
            d = P["dim"][0][0:2]
            rng = np.random.default_rng()
            if d[2] < 3:
                lkp = np.array([0, 1, 5])
                x1, x2, x3 = np.mgrid[1:d[0]+0.5:skip[0], 1:d[1]+0.5:skip[1], 1:d[2]:skip[2]]
                x1 = x1 + rng.random(x1.size) * 0.5
                x2 = x2 + rng.random(x2.size) * 0.5
            else:
                x1, x2, x3 = np.mgrid[1:d[0] + 0.5:skip[0], 1:d[1] + 0.5:skip[1], 1:d[2]:skip[2]+0.5]
                x1 = x1 + rng.random(x1.size) * 0.5
                x2 = x2 + rng.random(x2.size) * 0.5
                x3 = x3 + rng.random(x3.size) * 0.5

            x1 = x1.reshape([x1.size, 1])
            x2 = x2.reshape([x2.size, 1])
            x3 = x3.reshape([x3.size, 1])

            wt = np.array([])

            V, P["C"][0] = self.smooth_vol(P[0], flags["interp"], flags["wrap"], flags["fwhm"])
            tempD = np.array([1, 1, 1], ndmin=2) * int(flags['interp'])
            deg = np.hstack((tempD.T, np.array(flags['wrap'],ndmin=2).T))

            G, dG1, dG2, dG3 = spm.bsplins(V, x1, x2, x3, deg)
            # clear V
            A0 = self.make_A(P["mat"][0], x1, x2, x3, dG1, dG2, dG3, wt, lkp)

            b = G
            if not wt:
                b = b * wt

            # if not fNFB:
            #     Alpha = np.array([A0, b])
            #     Alpha = Alpha.T @ Alpha
            #     det0 = np.linalg.det(Alpha)
            #     det1 = det0

        if fNFB:
            thAcc = 0.01
            nrIter = 10
        else:
            thAcc = 1e-8
            nrIter = 64

        V, P["C"][1] = self.smooth_vol(P[1], flags["interp"], flags["wrap"], flags["fwhm"])
        d = np.array(V.shape)
        ss = np.inf
        countdown = -1
        for iter in range(1,nrIter+1):
            y1, y2, y3 = self.coords(np.zeros((6,1)), P["mat"][0], P["mat"][1], x1, x2, x3)
            msk = np.where( y1 >= 1 and y1 <= d[0] and y2 >= 1 and y2 <= d[1] and y3 >= 1 and y3 <= d[2] )
            if msk.size < 32:
                self.error_message(P[1])
            F = spm.bsplins(V, y1[msk], y2[msk], y3[msk], deg)
            if not wt:
                F = F * wt[msk]

            if fNFB:
                if iter == 1:
                    fixA0 = A0.T @ A0

            A = A0[msk,:]
            b1 = b[msk]
            sc = np.sum(b1) / np.sum(F)
            b1 = b1 - F * sc
            if not fNFB:
                soln = np.linalg.solve((A.T @ A),(A.T @ b1))
            else:
                soln = np.linalg.solve(fixA0, (A.T @ b1))

            p = np.array([0,0,0,0,0,0,1,1,1,0,0,0])
            p[lkp] = p[lkp] + soln.T
            P["mat"][1] = np.linalg.solve(self.spm_matrix(p),P["mat"][1])

            pss = ss
            ss = np.sum(b1 ** 2) / b1.size
            if ((pss-ss)/pss < thAcc) and (countdown == -1):
                countdown = 2
            if not countdown == -1:
                if countdown == 0:
                    break
                countdown -= 1

        nrIter = iter

        return P, A0, x1, x2, x3, wt, deg, b, nrIter

    def coords(self, p, M1, M2, x1, x2, x3):

        M = np.linalg.inv(M2) * np.linalg.inv(self.spm_matrix(p)) * M1
        y1 = M[0, 0] * x1 + M[0, 1] * x2 + M[0, 2] * x3 + M[0, 3]
        y2 = M[1, 0] * x1 + M[1, 1] * x2 + M[1, 2] * x3 + M[1, 3]
        y3 = M[2, 0] * x1 + M[2, 1] * x2 + M[2, 2] * x3 + M[2, 3]

        return y1, y2, y3

    def smooth_vol(self, P, hld, wrp, fwhm):

        s = np.sqrt(np.sum(P["mat"][0:2,0:2] ** 2)) ** (-1) * (fwhm / np.sqrt(8 * np.log(2)))

        x = round(6 * s[0])
        x = np.array(range(-x,x+1))

        y = round(6 * s[1])
        y = np.array(range(-y, y + 1))

        z = round(6 * s[2])
        z = np.array(range(-z, z + 1))

        x = np.exp( -x ** 2 / (2 * s[0] ** 2) )
        y = np.exp( -y ** 2 / (2 * s[1] ** 2) )
        z = np.exp( -z ** 2 / (2 * s[2] ** 2) )

        x = x / np.sum(x)
        y = y / np.sum(y)
        z = z / np.sum(z)

        i = (x.size - 1) / 2
        j = (y.size - 1) / 2
        k = (z.size - 1) / 2

        tempD = np.array([1, 1, 1], ndmin=2) * int(hld)
        d = np.hstack((tempD.T, np.array(wrp,ndmin=2).T))

        Coef = spm.bsplins(P["Vol"], d)
        V = np.zeros(P["Vol"].shape)
        V = spm.spm_conv_vol(Coef, V, x, y, z, np.array([-i, -j, -k]))

        return V, Coef


    def make_A(self, M, x1, x2, x3, dG1, dG2, dG3, wt, lkp):

        p0 = np.array([0,0,0,0,0,0,1,1,1,0,0,0])
        A = np.zeros((x1.size),lkp.size)
        for i in range(0,lkp.size+1):
            pt = p0
            pt[lkp[i]] = pt[i] + 1e-6
            y1, y2, y3 = self.coords(pt, M, M, x1, x2, x3)
            tmp = np.sum( np.array([y1-x1, y2-x2, y3-x3]) * np.array([dG1, dG2, dG3]), 2) / (-1e-6)
            if not wt:
                A[:,i] = tmp * wt
            else:
                A[:,i] = tmp

        return A

    def spm_matrix(self, P, order):

        if P.size == 3:
            A = np.eye(4)
            A[0:2,3] = P[:]
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

    def error_message(self, P):
        print('There is not enough overlap in the images to obtain a solution. Offending image:',P.fname,)

