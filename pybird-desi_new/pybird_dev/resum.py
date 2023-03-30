import os
import numpy as np
from numpy import pi, cos, sin, log, exp, sqrt, trapz
from scipy.interpolate import interp1d
from .fftlog import FFTLog, MPC, CoefWindow
from .common import co
from .resumfactor import Qa, Qawithhex, Qawithhex20


class Resum(object):
    """
    given a Bird() object, performs the IR-resummation of the power spectrum.
    There are two options:
    1.  fullresum: the FFTLog's are performed on the full integrands from s = .1 to s = 10000. in (Mpc/h) (default)
    2. 'optiresum: the FFTLog's are performed only on the BAO peak that is extracted by removing the smooth part of the correlation function. What is left is then padded with zeros and the FFTLog's run from s = .1 to s = 1000. in (Mpc/h).


    Attributes
    ----------
    co : class
        An object of type Common() used to share data
    LambdaIR : float
        Integral cutoff for IR-filters X and Y (fullresum: LambdaIR=.2 (default), optiresum: LambdaIR= 1 ; either value can do for either resummation)
    NIR : float
        Number of IR-correction terms in the sums over n and alpha, where n is the order of the Taylor expansion in powers of k^2 of the exponential of the bulk displacements, and for each n, alpha = { 0, 2 } are the orders of spherical Bessel functions. The ordering of the IR-corrections is given by (n,alpha), where alpha is running faster, e.g. (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2), ...
    k2p: ndarray
        powers of k^2
    alllpr : ndarray
        alpha = { 0, 2 } orders of spherical Bessel functions, for each n
    Q : ndarray
        IR-resummation bulk coefficients Q^{ll'}_{||N-j}(n, \alpha, f) of the IR-resummation matrices. f is the growth rate. Computed in method Ps().
    IRcorr : ndarray
        Q-independent pieces in the IR-correction sums over n and alpha of the power spectrum, for bird.which = 'full'. Computed in method Ps().
    IR11 : ndarray
        Q-independent in the IR-correction sums over n and alpha of the power spectrum linear part, for bird.which = 'all'. Computed in method Ps().
    IRct : ndarray
        Q-independent pieces in the IR-correction sums over n and alpha of the power spectrum counterterm, for bird.which = 'all'. Computed in method Ps().
    IRloop : ndarray
        Q-independent loop pieces in the IR-correction sums over n and alpha of the power spectrum loop part, for bird.which = 'all'. Computed in method Ps().
    IRresum : ndarray
        IR-corrections to the power spectrum, for bird.which = 'full'. Computed in method Ps().
    IR11resum : ndarray
        IR-corrections to the power spectrum linear parts, for bird.which = 'all'. Computed in method Ps().
    IRctresum : ndarray
        IR-corrections to the power spectrum counterterms, for bird.which = 'all'. Computed in method Ps().
    IRloopresum : ndarray
        IR-corrections to the power spetrum loop parts, for bird.which = 'all'. Computed in method Ps().
    fftsettings : dict
        Number of points and boundaries of the FFTLog's for the computing the IR-corrections
    fft : class
        An object of type FFTLog() to evaluate the IR-corrections
    M : ndarray
        spherical Bessel transform matrices to evaluate the IR-corrections
    kPow : ndarray
        k's to the powers on which to perform the FFTLog to evaluate the IR-corrections.
    Xfftsettings : dict
        Number of points and boundaries of the FFTLog's for evaluating the IR-filters X and Y
    Xfft : class
        An object of type FFTLog() to evaluate the IR-filters X and Y
    XM : ndarray
        spherical Bessel transform matrices to evaluate the IR-filters X and Y
    XsPow : ndarray
        s's to the powers on which to perform the FFTLog to evaluate the IR-filters X and Y
    """

    def __init__(self, LambdaIR=0.2, NFFT=192, co=co):

        self.co = co
        self.LambdaIR = LambdaIR

        if self.co.optiresum is True:
            self.sLow = 70.0
            self.sHigh = 190.0
            self.idlow = np.where(self.co.s > self.sLow)[0][0]
            self.idhigh = np.where(self.co.s > self.sHigh)[0][0]
            self.sbao = self.co.s[self.idlow : self.idhigh]
            self.snobao = np.concatenate([self.co.s[: self.idlow], self.co.s[self.idhigh :]])
            self.sr = self.sbao
        else:

            self.sr = self.co.s

        self.klow = 0.02
        self.kr = self.co.k[self.klow <= self.co.k]
        self.Nkr = self.kr.shape[0]
        self.Nlow = np.where(self.klow <= self.co.k)[0][0]
        k2pi = np.array([self.kr ** (2 * (p + 1)) for p in range(self.co.NIR)])
        self.k2p = np.concatenate((k2pi, k2pi))

        self.fftsettings = dict(Nmax=NFFT, xmin=0.1, xmax=10000.0, bias=-0.6)
        self.fft = FFTLog(**self.fftsettings)
        self.setM()
        self.setkPow()

        self.Xfftsettings = dict(Nmax=32, xmin=1.5e-5, xmax=10.0, bias=-2.6)
        self.Xfft = FFTLog(**self.Xfftsettings)
        self.setXM()
        self.setXsPow()

        self.Cfftsettings = dict(Nmax=256, xmin=1.0e-3, xmax=10.0, bias=-0.6)
        self.Cfft = FFTLog(**self.Cfftsettings)
        self.setMl()
        self.setsPow()

        # self.damping = CoefWindow(self.co.Nk-1, window=.2, left=False, right=True)
        self.kl2 = self.co.k[self.co.k < 0.5]
        Nkl2 = len(self.kl2)

        self.kl4 = self.co.k[self.co.k < 0.4]
        Nkl4 = len(self.kl4)

        self.dampPs = np.array(
            [
                CoefWindow(self.co.Nk - 1, window=0.25, left=False, right=True),
                np.pad(
                    CoefWindow(Nkl2 - 1, window=0.25, left=False, right=True), (0, self.co.Nk - Nkl2), mode="constant"
                ),
                np.pad(
                    CoefWindow(Nkl4 - 1, window=0.25, left=False, right=True), (0, self.co.Nk - Nkl4), mode="constant"
                ),
            ]
        )

        self.scut = self.co.s[self.co.s < 70.0]
        self.dampCf = np.pad(
            CoefWindow(self.co.Ns - len(self.scut) - 1, window=0.25, left=True, right=True),
            (len(self.scut), 0),
            mode="constant",
        )

    def setXsPow(self):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the IR-filters X and Y. """
        self.XsPow = exp(np.einsum("n,s->ns", -self.Xfft.Pow - 3.0, log(self.sr)))

    def setXM(self):
        """ Compute the matrices to evaluate the IR-filters X and Y. Called at instantiation. """
        self.XM = np.empty(shape=(2, self.Xfft.Pow.shape[0]), dtype="complex")
        for l in range(2):
            self.XM[l] = MPC(2 * l, -0.5 * self.Xfft.Pow)

    def IRFilters(self, bird, soffset=1.0, LambdaIR=None, RescaleIR=1.0, window=None):
        """ Compute the IR-filters X and Y. """
        if LambdaIR is None:
            LambdaIR = self.LambdaIR
        if self.co.exact_time and self.co.quintessence:
            Pin = bird.G1 ** 2 * bird.Pin
        else:
            Pin = bird.Pin
        Coef = self.Xfft.Coef(bird.kin, Pin * exp(-bird.kin ** 2 / LambdaIR ** 2) / bird.kin ** 2, window=window)
        CoefsPow = np.einsum("n,ns->ns", Coef, self.XsPow)
        X02 = np.real(np.einsum("ns,ln->ls", CoefsPow, self.XM))
        X0offset = np.real(np.einsum("n,n->", np.einsum("n,n->n", Coef, soffset ** (-self.Xfft.Pow - 3.0)), self.XM[0]))
        X02[0] = X0offset - X02[0]
        # if self.co.nonequaltime:
        #     X = RescaleIR * 2/3. * bird.D1*bird.D2/bird.D**2 * (X02[0] - X02[1]) + 1/3. * (bird.D1-bird.D2)**2/bird.D**2 * X0offset
        #     Y = 2. * bird.D1*bird.D2/bird.D**2 * X02[1]
        # else:
        X = RescaleIR * 2.0 / 3.0 * (X02[0] - X02[1])
        Y = 2.0 * X02[1]
        return X, Y

    def setkPow(self):
        """ Multiply the coefficients with the k's to the powers of the FFTLog to evaluate the IR-corrections. """
        self.kPow = exp(np.einsum("n,s->ns", -self.fft.Pow - 3.0, log(self.kr)))

    def setM(self, Nl=3):
        """ Compute the matrices to evaluate the IR-corrections. Called at instantiation. """
        self.M = np.empty(shape=(Nl, self.fft.Pow.shape[0]), dtype="complex")
        for l in range(Nl):
            self.M[l] = 8.0 * pi ** 3 * MPC(2 * l, -0.5 * self.fft.Pow)

    def IRn(self, XpYpC, window=None):
        """ Compute the spherical Bessel transform in the IR correction of order n given [XY]^n """
        Coef = self.fft.Coef(self.sr, XpYpC, extrap="padding", window=window)
        CoefkPow = np.einsum("n,nk->nk", Coef, self.kPow)
        return np.real(np.einsum("nk,ln->lk", CoefkPow, self.M[: self.co.Na]))

    def extractBAO(self, cf):
        """Given a correlation function cf,
        - if fullresum, return cf
        - if optiresum, extract the BAO peak"""
        if self.co.optiresum is True:
            cfnobao = np.concatenate([cf[..., : self.idlow], cf[..., self.idhigh :]], axis=-1)
            nobao = (
                interp1d(self.snobao, self.snobao ** 2 * cfnobao, kind="linear", axis=-1)(self.sbao) * self.sbao ** -2
            )
            bao = cf[..., self.idlow : self.idhigh] - nobao
            return bao
        else:
            return cf

    def setXpYp(self, bird):
        X, Y = self.IRFilters(bird)
        Xp = np.array([X ** (p + 1) for p in range(self.co.NIR)])
        XpY = np.array([Y * X ** p for p in range(self.co.NIR)])
        XpYp = np.concatenate((Xp, XpY))
        # return np.array([item for pair in zip(Xp, XpY + [0]) for item in pair])
        return XpYp

    def makeQ(self, f):
        """ Compute the bulk coefficients Q^{ll'}_{||N-j}(n, \alpha, f) """
        # if len(f) == 1:
        # Q = np.empty(shape=(2, self.co.Nl, self.co.Nl, self.co.Nn))
        Q = np.zeros(shape=(2, self.co.Nl, self.co.Nl, self.co.Nn))
        # for a in range(2):
        #     for l in range(self.co.Nl):
        #         for lpr in range(self.co.Nl):
        #             for u in range(self.co.Nn):
        #                 if self.co.NIR is 8:
        #                     Q[a][l][lpr][u] = Qa[1 - a][2 * l][2 * lpr][u](f)
        #                 elif self.co.NIR is 16:
        #                     Q[a][l][lpr][u] = Qawithhex[1 - a][2 * l][2 * lpr][u](f)
        #                 elif self.co.NIR is 20:
        #                     Q[a][l][lpr][u] = Qawithhex20[1 - a][2 * l][2 * lpr][u](f)
        
        for a in range(2):
            for l in range(3):
                for lpr in range(3):
                    for u in range(self.co.Nn):
                        if self.co.NIR is 8:
                            Q[a][l][lpr][u] = Qa[1 - a][2 * l][2 * lpr][u](f)
                        elif self.co.NIR is 16:
                            Q[a][l][lpr][u] = Qawithhex[1 - a][2 * l][2 * lpr][u](f)
                        elif self.co.NIR is 20:
                            Q[a][l][lpr][u] = Qawithhex20[1 - a][2 * l][2 * lpr][u](f)
    
        # else:
        #     Q = np.zeros(shape=(len(f), 2, self.co.Nl, self.co.Nl, self.co.Nn))
        #     for i in range(len(f)):
        #         for a in range(2):
        #             for l in range(3):
        #                 for lpr in range(3):
        #                     for u in range(self.co.Nn):
        #                         if self.co.NIR is 8:
        #                             Q[i][a][l][lpr][u] = Qa[1 - a][2 * l][2 * lpr][u](f[i])
        #                         elif self.co.NIR is 16:
        #                             Q[i][a][l][lpr][u] = Qawithhex[1 - a][2 * l][2 * lpr][u](f[i])
        #                         elif self.co.NIR is 20:
        #                             Q[i][a][l][lpr][u] = Qawithhex20[1 - a][2 * l][2 * lpr][u](f[i])
                  
        return Q

    def setMl(self):
        """ Compute the power spectrum to correlation function spherical Bessel transform matrices. Called at the instantiation. """
        self.Ml = np.empty(shape=(self.co.Nl, self.Cfft.Pow.shape[0]), dtype="complex")
        for l in range(self.co.Nl):
            self.Ml[l] = 1j ** (2 * l) * MPC(2 * l, -0.5 * self.Cfft.Pow)

    def setsPow(self):
        """ Multiply the coefficients with the s's to the powers of the FFTLog to evaluate the IR corrections in configuration space. """
        self.sPow = exp(np.einsum("n,s->ns", -self.Cfft.Pow - 3.0, log(self.co.s)))

    def Ps2Cf(self, P, l=0):
        Coef = self.Cfft.Coef(self.co.k, P * self.dampPs[l], extrap="padding", window=None)
        CoefsPow = np.einsum("n,ns->ns", Coef, self.sPow)
        return np.real(np.einsum("ns,n->s", CoefsPow, self.Ml[l])) * self.dampCf

    def IRCf(self, bird, window=None):
        """ Compute the IR corrections in configuration space by spherical Bessel transforming the IR corrections in Fourier space.  """

        if bird.with_bias:
            for a, IRa in enumerate(
                bird.fullIRPs
            ):  # this can be speedup x2 by doing FFTLog[lin+loop] instead of separately
                for l, IRal in enumerate(IRa):
                    bird.fullIRCf[a, l] = self.Ps2Cf(IRal, l=l)
        else:
            for l, IRl in enumerate(bird.fullIRPs11):
                for j, IRlj in enumerate(IRl):
                    bird.fullIRCf11[l, j] = self.Ps2Cf(IRlj, l=l)
            for l, IRl in enumerate(bird.fullIRPsct):
                for j, IRlj in enumerate(IRl):
                    bird.fullIRCfct[l, j] = self.Ps2Cf(IRlj, l=l)
            for l, IRl in enumerate(bird.fullIRPsloop):
                for j, IRlj in enumerate(IRl):
                    bird.fullIRCfloop[l, j] = self.Ps2Cf(IRlj, l=l)

    def PsCf(self, bird, makeIR=True, makeQ=True, setPs=True, setCf=True, window=None):

        self.Ps(bird, makeIR=makeIR, makeQ=makeQ, setPs=setPs, window=window)
        self.IRCf(bird, window=window)
        if setCf:
            bird.setresumCf()

    def Ps(self, bird, makeIR=True, makeQ=True, setPs=True, window=None, init=False):

        if makeQ:
            bird.Q = self.makeQ(bird.f)
        if makeIR:
            self.IRPs(bird, window=window)
        if init == False:
            bird.setIRPs()
        if setPs:
            bird.setresumPs()

    def IRPs(self, bird, window=None, IRPs_all = None):
        """ This is the main method of the class. Compute the IR corrections in Fourier space. """

        XpYp = self.setXpYp(bird)

        if bird.with_bias:
            for a, cf in enumerate(self.extractBAO(bird.Cf)):
                for l, cl in enumerate(cf):
                    for j, xy in enumerate(XpYp):
                        IRcorrUnsorted = np.real((-1j) ** (2 * l)) * self.k2p[j] * self.IRn(xy * cl, window=window)
                        for v in range(self.co.Na):
                            bird.IRPs[a, l, j * self.co.Na + v, self.Nlow :] = IRcorrUnsorted[v]

        else:
            if IRPs_all is None:
                for l, cl in enumerate(self.extractBAO(bird.C11)):
                    for j, xy in enumerate(XpYp):
                        IRcorrUnsorted = np.real((-1j) ** (2 * l)) * self.k2p[j] * self.IRn(xy * cl, window=window)
                        for v in range(self.co.Na):
                            bird.IRPs11[l, j * self.co.Na + v, self.Nlow :] = IRcorrUnsorted[v]
                for l, cl in enumerate(self.extractBAO(bird.Cct)):
                    for j, xy in enumerate(XpYp):
                        IRcorrUnsorted = np.real((-1j) ** (2 * l)) * self.k2p[j] * self.IRn(xy * cl, window=window)
                        for v in range(self.co.Na):
                            bird.IRPsct[l, j * self.co.Na + v, self.Nlow :] = IRcorrUnsorted[v]
                for l, cl in enumerate(self.extractBAO(bird.Cloopl)):
                    for i, cli in enumerate(cl):
                        for j, xy in enumerate(XpYp):
                            IRcorrUnsorted = np.real((-1j) ** (2 * l)) * self.k2p[j] * self.IRn(xy * cli, window=window)
                            for v in range(self.co.Na):
                                bird.IRPsloop[l, i, j * self.co.Na + v, self.Nlow :] = IRcorrUnsorted[v]
            else:
                IRPs11, IRPsct, IRPsloop = IRPs_all
                for l, cl in enumerate(self.extractBAO(bird.C11)):
                    for j, xy in enumerate(XpYp):
                        IRcorrUnsorted = np.real((-1j) ** (2 * l)) * self.k2p[j] * self.IRn(xy * cl, window=window)
                        for v in range(self.co.Na):
                            IRPs11[l, j * self.co.Na + v, self.Nlow :] = IRcorrUnsorted[v]
                for l, cl in enumerate(self.extractBAO(bird.Cct)):
                    for j, xy in enumerate(XpYp):
                        IRcorrUnsorted = np.real((-1j) ** (2 * l)) * self.k2p[j] * self.IRn(xy * cl, window=window)
                        for v in range(self.co.Na):
                            IRPsct[l, j * self.co.Na + v, self.Nlow :] = IRcorrUnsorted[v]
                for l, cl in enumerate(self.extractBAO(bird.Cloopl)):
                    for i, cli in enumerate(cl):
                        for j, xy in enumerate(XpYp):
                            IRcorrUnsorted = np.real((-1j) ** (2 * l)) * self.k2p[j] * self.IRn(xy * cli, window=window)
                            for v in range(self.co.Na):
                                IRPsloop[l, i, j * self.co.Na + v, self.Nlow :] = IRcorrUnsorted[v]
                                
                return IRPs11, IRPsct, IRPsloop
