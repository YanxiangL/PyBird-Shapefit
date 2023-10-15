# This file contains useful routines that should work regardless of whether you are fitting
# with fixed or varying template, and for any number of cosmological parameters

import os
import sys
import copy
import numpy as np
import scipy as sp
from scipy.linalg import lapack, cholesky, block_diag
from scipy.interpolate import splrep, splev, InterpolatedUnivariateSpline
from scipy.ndimage import map_coordinates
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.special import legendre, spherical_jn, j1, loggamma
from scipy.misc import derivative
from abc import ABC
from scipy.stats import linregress
from scipy.interpolate import interp1d

sys.path.append("../")
from tbird.Grid import grid_properties, run_camb, run_class
from tbird.computederivs import get_grids, get_PSTaylor, get_ParamsTaylor

# Wrapper around the pybird data and model evaluation
class BirdModel:
    def __init__(self, pardict, redindex=0, template=False, direct=False, window=None, fittingdata=None, Shapefit = False):

        self.redindex = redindex
        self.pardict = pardict
        self.Nl = 3 if pardict["do_hex"] else 2
        self.template = template
        self.direct = direct
        self.window = window
        self.Shapefit = Shapefit
        self.corr_convert = pardict['corr_convert']

        # Some constants for the EFT model
        # self.k_m, self.k_nl = 0.7, 0.7
        
        if pardict['do_corr']:
            self.k_m, self.k_nl = 1.0, 1.0
        else:
            self.k_m, self.k_nl = 0.7, 0.7
        
        # self.k_m, self.k_nl = 0.7, 0.7
        
        # self.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0])
        # self.eft_priors = np.array([2.0, 2.0, 2.0, 2.0, 0.2, 1.0, 1.0])
        self.eft_priors = None

        # Get some values at the grid centre
        if pardict["code"] == "CAMB":
            (
                self.kmod,
                self.Pmod,
                self.Om,
                self.Da,
                self.Hz,
                self.D,
                self.fN,
                self.sigma8,
                self.sigma8_0,
                self.sigma12,
                self.r_d,
            ) = run_camb(pardict)
            self.omega_nu = float(self.pardict["Sum_mnu"]) / 93.14
        else:
            (
                self.kmod,
                self.Pmod,
                self.Om,
                self.Da,
                self.Hz,
                self.D,
                self.fN,
                self.sigma8,
                self.sigma8_0,
                self.sigma12,
                self.r_d,
            ) = run_class(pardict)
            self.omega_nu = float(self.pardict["m_ncdm"]) / 93.14
            
        if (len(self.D) == 1 and not isinstance(self.D, float)):
            self.D = self.D[0]
            self.fN = self.fN[0]
            self.Da = self.Da[0]
            self.Hz = self.Hz[0]
            self.sigma8 = self.sigma8[0]
            self.sigma12 = self.sigma12[0]

        # Prepare the model
        if self.direct:
            self.valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
            self.delta = np.array(pardict["dx"], dtype=np.float)
            if self.template:
                self.correlator = self.setup_pybird(fittingdata)
                first = len(self.pardict["z_pk"]) // 2
                # self.correlator = self.setup_pybird()
                # print(self.D, isinstance(self.D, float))
                self.correlator.compute(
                    {
                        "k11": self.kmod,
                        "P11": self.Pmod[first],
                        "z": float(self.pardict["z_pk"][self.redindex]),
                        "Omega0_m": self.Om,
                        "D": self.D,
                        "f": self.fN,
                        "DA": self.Da,
                        "H": self.Hz,
                    },
                Templatefit = self.template, corr_convert=self.corr_convert)
                self.linmod, self.loopmod = None, None
                self.kin = self.correlator.co.k
            else:
                self.correlator = self.setup_pybird(fittingdata)
                self.kin = self.correlator.co.k
        else:
            self.valueref, self.delta, self.flattenedgrid, self.truecrd = grid_properties(pardict)
            self.kin, self.paramsmod, self.linmod, self.loopmod = self.load_model()
            if self.template:
                self.correlator = self.setup_pybird()
                self.correlator.compute(
                    {
                        "k11": self.kmod,
                        "P11": self.Pmod,
                        "z": float(self.pardict["z_pk"][self.redindex]),
                        "Omega0_m": self.Om,
                        "f": self.fN,
                        "DA": self.Da,
                        "H": self.Hz,
                    }, Templatefit = self.template, corr_convert=self.corr_convert
                )
                
            # if bool(self.corr_convert) == True:
            #     self.kmode = np.logspace(np.log10(0.001), np.log10(40.0), 10000)
            #     # self.dist = np.logspace(0.0, 3.0, 5000)
            #     self.pk2xi_0 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=0)
            #     self.pk2xi_2 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=2)
            #     self.pk2xi_4 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=4)
            #     self.kmode_in = self.kmode[np.where(self.kmode <= 0.5)[0]]
            #     self.kmode_out = self.kmode[np.where(self.kmode > 0.5)[0]]
                
    #     if self.corr_convert == True:
    #         self.kmode = np.logspace(np.log10(np.min(self.kin)), np.log10(np.max(self.kin)), 5000)
    #         self.dist = np.logspace(0.0, 3.0, 5000)
    #         self.getxbin_mat_cf(fittingdata)
            
    #         from pybird_dev.pybird import PowerToCorrelation, PowerToCorrelationSphericalBessel
            
    #         self.pk2xi_0 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=0)
    #         self.pk2xi_2 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=2)
    #         if pardict['do_hex']:
    #             self.pk2xi_4 = PowerToCorrelationSphericalBessel(qs=self.kmode, ell=4)
    
            
    # def getxbin_mat_cf(self, fittingdata):
        
    #     ss = fittingdata.data['xdata'][0]
        
    #     ds = ss[-1] - ss[-2]
    #     # dk = ks[1] - ks[0]
    #     ss_input = self.dist
    #     # ks_input = np.concatenate([np.geomspace(1e-5, 0.015, 100, endpoint=False), np.arange(0.015, self.co.kmax, 1e-3)])
        
    #     # print(self.kmat)
        
    #     # self.p = np.concatenate([np.geomspace(1e-5, 0.015, 100, endpoint=False), np.arange(0.015, self.co.kmax, 1e-3)])

    #     binmat = np.zeros((len(ss), len(ss_input)))
    #     for ii in range(len(ss_input)):

    #         # Define basis vector
    #         cfvec = np.zeros_like(ss_input)
    #         cfvec[ii] = 1
    #         # print(pkvec)

    #         # Define the spline:
    #         cfvec_spline = splrep(ss_input, cfvec)

    #         # Now compute binned basis vector:
    #         tmp = np.zeros_like(ss)
    #         for i, sk in enumerate(ss):
    #             if i == 0 or i == len(ss) - 1:
    #                 sl = sk - ds / 2
    #                 sr = sk + ds / 2
    #             else:
    #                 sl = (sk + ss[i-1])/2.0
    #                 sr = (ss[i+1] + sk)/2.0
                
    #             s_in = np.linspace(sl, sr, 100)
    #             tmp[i] = np.trapz(s_in**2 * splev(s_in, cfvec_spline, ext=2), x=s_in) * 3 / (sr**3 - sl**3)
                
    #         binmat[:, ii] = tmp
        
        
    #     # if self.co.Nl == 2:
    #     #     self.xbin_mat = block_diag(*[binmat, binmat])
    #     # else:
    #     #     self.xbin_mat = block_diag(*[binmat, binmat, binmat])
    #     self.xbin_mat = binmat

    def setup_pybird(self, fittingdata=None):

        from pybird_dev.pybird import Correlator

        # Nl = 3 if self.pardict["do_hex"] else 2
        Nl = 3
        optiresum = True if self.pardict["do_corr"] else False
        output = "bCf" if self.pardict["do_corr"] else "bPk"
        
        # kmax = None if self.pardict["do_corr"] else 0.35
        if self.pardict['do_corr']:
            kmax = None
        elif self.corr_convert:
            kmax = 0.55
        else:
            kmax = 0.35
        
        with_binning = True if self.window is None else False
        correlator = Correlator()

        # Set up pybird
        correlator.set(
            {
                "output": output,
                "multipole": Nl,
                "skycut": len(self.pardict["z_pk"]),
                "z": np.float64(self.pardict["z_pk"]),
                "optiresum": False,
                "with_bias": False,
                "with_time": not (self.template),
                "xdata": [max(x, key=len) for x in fittingdata.data["x_data"]],
                "with_AP": True,
                "kmax": kmax,
                "DA_AP": self.Da,
                "H_AP": self.Hz,
                "with_window": False if self.window is None else True,
                "windowPk": str(self.window),
                "windowCf": str(self.window) + ".dat",
                'with_stoch': True,
                "with_fibercol": False,
                "with_binning": with_binning,
                "with_resum": True,
                "corr_convert": bool(self.corr_convert),
            }
        )
        
        # print(with_binning)

        return correlator

    def load_model(self):

        # Load in the model components
        outgrids = np.loadtxt(self.pardict["outgrid"], dtype=str)
        gridnames = np.loadtxt(self.pardict["gridname"], dtype=str)
        gridname = self.pardict["code"].lower() + "-" + gridnames[self.redindex]
        if self.pardict["taylor_order"] > 0:
            paramsmod = np.load(
                os.path.join(outgrids[self.redindex], "DerParams_%s.npy" % gridname),
                allow_pickle=True,
            )
            if self.template:
                paramsmod = None
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerClin_%s_noAP.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerCloop_%s_noAP.npy" % gridname),
                        allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPlin_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPloop_%s_noAP.npy" % gridname),
                        allow_pickle=True,
                    )
            else:
                if self.pardict["do_corr"]:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerClin_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerCloop_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                else:
                    linmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPlin_%s.npy" % gridname),
                        allow_pickle=True,
                    )
                    loopmod = np.load(
                        os.path.join(outgrids[self.redindex], "DerPloop_%s.npy" % gridname),
                        allow_pickle=True,
                    )
            kin = linmod[0][0, :, 0]
            print(os.path.join(outgrids[self.redindex], "DerPlin_%s.npy" % gridname))
        elif self.pardict["taylor_order"] < 0:
            paramstab, lintab, looptab = get_grids(
                self.pardict, outgrids[self.redindex], gridname, pad=False, cf=self.pardict["do_corr"]
            )
            paramsmod = paramstab
            kin = lintab[..., 0, :, 0][(0,) * len(self.pardict["freepar"])]
            linmod = lintab
            loopmod = looptab
        else:
            paramstab, lintab, looptab = get_grids(
                self.pardict, outgrids[self.redindex], gridname, pad=False, cf=self.pardict["do_corr"]
            )
            paramsmod = sp.interpolate.RegularGridInterpolator(self.truecrd, paramstab)
            kin = lintab[..., 0, :, 0][(0,) * len(self.pardict["freepar"])]
            if self.template:
                linmod = sp.interpolate.RegularGridInterpolator(self.truecrd, lintab_noAP)
                loopmod = sp.interpolate.RegularGridInterpolator(self.truecrd, looptab_noAP)
            else:
                linmod = sp.interpolate.RegularGridInterpolator(self.truecrd, lintab)
                loopmod = sp.interpolate.RegularGridInterpolator(self.truecrd, looptab)
        

        return kin, paramsmod, linmod, loopmod

    def compute_params(self, coords):

        if self.pardict["taylor_order"]:
            dtheta = np.array(coords) - self.valueref
            Params = get_ParamsTaylor(dtheta, self.paramsmod, self.pardict["taylor_order"])
        else:
            Params = self.paramsmod(coords)[0]

        return Params

    def compute_pk(self, coords):

        if self.direct:
            Plins, Ploops = [], []
            for i in range(np.shape(coords)[1]):
                Plin, Ploop = self.compute_model_direct(coords[:, i])
                Plins.append(Plin)
                Ploops.append(Ploop)
            Plin = np.transpose(np.array(Plins), axes=[1, 2, 3, 0])
            Ploop = np.transpose(np.array(Ploops), axes=[1, 3, 2, 0])
        else:
            if self.pardict["taylor_order"] > 0:
                dtheta = coords - self.valueref[:, None]
                # print()
                Plin = get_PSTaylor(dtheta, self.linmod, self.pardict["taylor_order"])
                Ploop = get_PSTaylor(dtheta, self.loopmod, self.pardict["taylor_order"])
                Plin = np.transpose(Plin, axes=[1, 3, 2, 0])[:, 1:, :, :]
                Ploop = np.transpose(Ploop, axes=[1, 2, 3, 0])[:, :, 1:, :]
            elif self.pardict["taylor_order"] < 0:
                Nl, Nk = np.shape(self.linmod)[np.shape(coords)[0]], np.shape(self.linmod)[np.shape(coords)[0] + 1]
                Nlin = np.shape(self.linmod)[np.shape(coords)[0] + 2]
                Nloop = np.shape(self.loopmod)[np.shape(coords)[0] + 2]
                scaled_coords = (coords - self.valueref[:, None]) / self.delta[:, None] + float(self.pardict["order"])
                Plin = np.zeros((Nl, Nlin - 1, Nk, np.shape(coords)[1]))
                Ploop = np.zeros((Nl, Nk, Nloop - 1, np.shape(coords)[1]))
                for i in range(Nl):
                    for j in range(Nk):
                        for k in range(1, Nlin):
                            Plin[i, k - 1, j] = map_coordinates(
                                self.linmod[..., i, j, k], scaled_coords, order=3, prefilter=False
                            )
                        for k in range(1, Nloop):
                            Ploop[i, j, k - 1] = map_coordinates(
                                self.loopmod[..., i, j, k], scaled_coords, order=3, prefilter=False
                            )
            else:
                Plin = np.transpose(self.linmod(coords.T), axes=[1, 3, 2, 0])[:, 1:, :, :]
                Ploop = np.transpose(self.loopmod(coords.T), axes=[1, 2, 3, 0])[:, :, 1:, :]

        return Plin, Ploop

    def compute_model_direct(self, coords, fittingdata = None, redindex = 0):

        parameters = copy.deepcopy(self.pardict)

        for k, var in enumerate(self.pardict["freepar"]):
            parameters[var] = coords[k]
        if parameters["code"] == "CAMB":
            kin, Pin, Om, Da_fid, Hz_fid, DN_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12, r_d = run_camb(parameters)
        else:
            kin, Pin, Om, Da_fid, Hz_fid, DN_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12, r_d = run_class(parameters)
            
        first = len(parameters["z_pk"]) // 2

        # Get non-linear power spectrum from pybird
        self.correlator.compute(
            {
                "k11": kin,
                "P11": Pin[first],
                "z": float(self.pardict["z_pk"][self.redindex]),
                "Omega0_m": Om,
                "f": fN_fid[0],
                "DA": DN_fid[0],
                "H": Hz_fid[0],
            }, Templatefit = self.template, corr_convert=self.corr_convert
        )
        
        Plin, Ploop = (
            self.correlator.bird.formatTaylorCf() if self.pardict["do_corr"] else self.correlator.bird.formatTaylorPs(kdata=fittingdata.data['x_data'][redindex][0])
        )

        Plin = np.swapaxes(Plin.reshape((self.correlator.co.Nl, Plin.shape[-2] // self.correlator.co.Nl, Plin.shape[-1], 1)), axis1=1, axis2=2)[
            :, 1:, :
        ]
        Ploop = np.swapaxes(Ploop.reshape((self.correlator.co.Nl, Ploop.shape[-2] // self.correlator.co.Nl, Ploop.shape[-1], 1)), axis1=1, axis2=2)[
            :, 1:, :
        ]
            
        Ploop = np.transpose(Ploop, axes=[0, 2, 1, 3])
        
        self.kin = self.correlator.projection.xout

        return Plin[:3], Ploop[:3]

    # def modify_template(self, params):
    #     # Modify the template power spectrum by scaling by f and then reapplying the AP effect.
    #     alpha_perp, alpha_par, fsigma8 = params
    #     self.correlator.bird.f = fsigma8 / self.sigma8

    #     P11l_AP, Pctl_AP, Ploopl_AP, Pnlol_AP = self.correlator.projection.AP(
    #         bird=self.correlator.bird, q=[alpha_perp, alpha_par], overwrite=False
    #     )
    #     Plin, Ploop = self.correlator.bird.formatTaylorPs(Ps=[P11l_AP, Ploopl_AP, Pctl_AP, Pnlol_AP])

    #     sum_cols = [0, 1, 4, 7, 9, 10, 12, 15, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    #     Ploop = np.array([np.sum(Ploop[:, sum_cols[i] : sum_cols[i + 1]], axis=1) for i in range(20)]).T

    #     Plin = np.swapaxes(Plin.reshape((3, Plin.shape[-2] // 3, Plin.shape[-1])), axis1=1, axis2=2)[:, 1:, :]
    #     Ploop = np.swapaxes(Ploop.reshape((3, Ploop.shape[-2] // 3, Ploop.shape[-1])), axis1=1, axis2=2)[:, 1:, :]

    #     return Plin, Ploop
    
    def modify_template(self, params, fittingdata=None, factor_m = None, factor_a = 0.6, factor_kp = 0.03, redindex=0, one_nz = False, sigma8 = None, resum = True, power = 1.0):
        # Modify the template power spectrum by scaling by f and then reapplying the AP effect.
        
        if one_nz == False:
            raise ValueError('PyBird is not able to fit multiple redshift bins with Shapefit currently.')
            # alpha_perp, alpha_par, fsigma8 = params
            # if sigma8 is None:
            #     factor = 1.0
            #     # factor = self.correlator.scale_interp[redindex](factor_m)
            #     # factor = np.abs(1.0/(alpha_perp**2*alpha_par)*(np.exp(factor_m/factor_a) - np.exp(-factor_m/factor_a)))
            #     self.correlator.birds[redindex].f = fsigma8 / (self.sigma8[redindex]*factor)
            #     sigma8_ratio = factor**2
            # else:
            #     self.correlator.birds[redindex].f = fsigma8 / sigma8
            #     sigma8_ratio = (sigma8/self.sigma8[redindex])**2
            
            # if resum == True:
            #     # self.correlator.birds[redindex].Q = self.correlator.resum.makeQ(self.correlator.birds[redindex].f)
            #     # # IRPs11, IRPsct, IRPsloop = self.correlator.resum.IRPs(self.correlator.birds[redindex], IRPs_all = [IRPs11, IRPsct, IRPsloop])
            #     # # IRPs11 += self.correlator.birds[redindex].IRPs11_new
            #     # # IRPsct += self.correlator.birds[redindex].IRPsct_new
            #     # # IRPsloop += self.correlator.birds[redindex].IRPsloop_new
                
            #     if self.Shapefit == True:
            #         # P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.birds[redindex].setShapefit(factor_m, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio, 
            #         # IRPs_all = [self.correlator.birds[redindex].IRPs11_new, self.correlator.birds[redindex].IRPsct_new, self.correlator.birds[redindex].IRPsloop_new], power=power)
            #         # P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.birds[redindex].setShapefit(factor_m, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio)
            #         # self.correlator.resum.Ps(self.correlator.bird)
            #         Plin = self.Pmod[len(self.pardict["z_pk"]) // 2]
            #         self.correlator.setShapefit_full(Plin, factor_m = factor_m, kmode=self.kmod, redindex=redindex, factor_a = factor_a, factor_kp = factor_kp)
            #         P11l, Ploopl, Pctl = self.correlator.birds[redindex].P11l, self.correlator.birds[redindex].Ploopl, self.correlator.birds[redindex].Pctl
                    
            #     else:
            #         self.correlator.birds[redindex].Q = self.correlator.resum.makeQ(self.correlator.birds[redindex].f)
            #         P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.birds[redindex].setShapefit(0.0, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio, 
            #         IRPs_all = [self.correlator.birds[redindex].IRPs11_new, self.correlator.birds[redindex].IRPsct_new, self.correlator.birds[redindex].IRPsloop_new], power=power)
            #         # P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.birds[redindex].setShapefit(0.0, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio)
                
            #         fullIRPs11 = np.einsum("lpn,pnk,pi->lik", self.correlator.birds[redindex].Q[0], IRPs11, self.correlator.birds[redindex].co.l11)
            #         fullIRPsct = np.einsum("lpn,pnk,pi->lik", self.correlator.birds[redindex].Q[1], IRPsct, self.correlator.birds[redindex].co.lct)
            #         fullIRPsloop = np.einsum("lpn,pink->lik", self.correlator.birds[redindex].Q[1], IRPsloop)
            #         P11l += fullIRPs11
            #         Pctl += fullIRPsct
            #         Ploopl += fullIRPsloop
                
            # else:
            #     self.correlator.birds[redindex].Q = self.correlator.resum.makeQ(self.correlator.birds[redindex].f)

            #     if self.Shapefit == False:
            #         factor_m = 0.0
            #     ratio = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(self.correlator.co.k/factor_kp)))*sigma8_ratio
                
            #     # P11l = np.einsum("jk, mnk->jmnk", ratio, self.P11l)
            #     # Pctl = np.einsum("jk, mnk->jmnk", ratio, self.Pctl)
            #     # Ploopl = np.einsum("jk, mnk->jmnk", ratio, self.Ploopl)
            
            #     P11l = self.correlator.birds[redindex].P11l*ratio
            #     Pctl = self.correlator.birds[redindex].Pctl*ratio
            #     Ploopl = self.correlator.birds[redindex].Ploopl*ratio ** 2
            
            # # P11l_AP, Pctl_AP, Ploopl_AP = self.correlator.projection[redindex].AP(
            # #     bird=self.correlator.birds[redindex], q=[alpha_perp, alpha_par], overwrite=False,
            # #     PS=[P11l, Ploopl, Pctl]
            # # )
            
            # # Pstl = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(self.correlator.co.k/factor_kp)))*sigma8_ratio*self.correlator.birds[redindex].Pstl
            
            # P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = self.correlator.projection[redindex].AP(
            #     bird=self.correlator.birds[redindex], q=[alpha_perp, alpha_par], overwrite=False,
            #     PS=[P11l, Ploopl, Pctl, self.correlator.birds[redindex].Pstl]
            # )
            
            # # P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = self.correlator.projection[redindex].AP(
            # #     bird=self.correlator.birds[redindex], q=[alpha_perp, alpha_par], overwrite=False,
            # #     PS=[P11l, Ploopl, Pctl, Pstl]
            # # )
            
            # # print(np.shape(P11l_AP))
            
            # # a, b, c = list([P11l_AP, Ploopl_AP, Pctl_AP])
            # if self.correlator.config["with_window"] == True:
            #     # P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP = self.correlator.projection[redindex].Window(self.correlator.birds[redindex], PS = list([P11l_AP, Ploopl_AP, Pctl_AP]))
            #     P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP = self.correlator.projection[redindex].Window(self.correlator.birds[redindex], PS = list([P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP]))
            # # else:
            # #     Pstl_AP = self.correlator.birds[redindex].Pstl
            
            # if self.window is None:
            #     P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP = self.correlator.projection[redindex].xbinning(self.correlator.birds[redindex], PS_all = list([P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP]))
            # else:
            #     P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = self.correlator.projection[redindex].xdata(self.correlator.birds[redindex], PS=[P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP])
            
            # Plin, Ploop = self.correlator.birds[redindex].formatTaylorPs(Ps=[P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP], kdata=self.correlator.projection[redindex].xout)
            
            # Plin = np.swapaxes(np.reshape(Plin, (self.correlator.co.Nl, Plin.shape[-2]//self.correlator.co.Nl, Plin.shape[-1]))[:, :, 1:], axis1=1, axis2=2)
            
            # Ploop = np.swapaxes(np.reshape(Ploop, (self.correlator.co.Nl, Ploop.shape[-2]//self.correlator.co.Nl, Ploop.shape[-1]))[:, :, 1:], axis1=1, axis2=2)
            
            # self.kin = self.correlator.projection[redindex].xout
        
        else:
            alpha_perp, alpha_par, fsigma8 = params
            if sigma8 is None:
                # factor = np.abs(1.0/(alpha_perp**2*alpha_par)*(np.exp(factor_m/factor_a) - np.exp(-factor_m/factor_a)))
                # factor = self.correlator.scale_interp(factor_m)
                factor = 1.0
                self.correlator.bird.f = fsigma8 / (self.sigma8*factor)
                sigma8_ratio = factor**2
                # self.correlator.bird.f = self.fN
                # sigma8_ratio = ((fsigma8/self.fN)/self.sigma8)**2
            else:
                self.correlator.bird.f = fsigma8 / sigma8
                sigma8_ratio = (sigma8/self.sigma8)**2
            
            # self.correlator.bird.Q = self.correlator.resum.makeQ(self.correlator.bird.f)
            # IRPs11_new, IRPsct_new, IRPsloop_new = self.correlator.resum.IRPs(self.correlator.bird, IRPs_all = [IRPs11, IRPsct, IRPsloop])
            # print(np.max(IRPs11_new - IRPs11), np.max(IRPsct_new - IRPsct), np.max(IRPsloop_new - IRPsloop))
            # IRPs11 += self.correlator.IRPs11_new
            # IRPsct += self.correlator.IRPsct_new
            # IRPsloop += self.correlator.IRPsloop_new
            
            if resum == True:
                if self.Shapefit == True:
                    
                    # P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.bird.setShapefit(factor_m, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio, 
                    #                                             IRPs_all = [self.correlator.IRPs11_new, self.correlator.IRPsct_new, self.correlator.IRPsloop_new])
                    P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.bird.setShapefit(factor_m, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio)
                    
                    self.correlator.bird.Q = self.correlator.resum.makeQ(self.correlator.bird.f)
                    
                    fullIRPs11 = np.einsum("lpn,pnk,pi->lik", self.correlator.bird.Q[0], IRPs11, self.correlator.co.l11)
                    fullIRPsct = np.einsum("lpn,pnk,pi->lik", self.correlator.bird.Q[1], IRPsct, self.correlator.co.lct)
                    fullIRPsloop = np.einsum("lpn,pink->lik", self.correlator.bird.Q[1], IRPsloop)
                    
                    # fullIRPs11 = np.einsum("alpn,apnk,pi->alik", self.correlator.bird.Q[:, 0], IRPs11, self.correlator.co.l11)
                    # fullIRPsct = np.einsum("alpn,apnk,pi->alik", self.correlator.bird.Q[:, 1], IRPsct, self.correlator.co.lct)
                    # fullIRPsloop = np.einsum("alpn,apink->alik", self.correlator.bird.Q[:, 1], IRPsloop)
                    
                    P11l += fullIRPs11
                    Pctl += fullIRPsct
                    Ploopl += fullIRPsloop
                    
                    # self.correlator.resum.Ps(self.correlator.bird)
                    # Plin = self.Pmod[len(self.pardict["z_pk"]) // 2]
                    # self.correlator.setShapefit_full(Plin, factor_m = factor_m, kmode=self.kmod, factor_a = factor_a, factor_kp = factor_kp, sigma8_ratio=sigma8_ratio)
                    # P11l, Ploopl, Pctl = self.correlator.bird.P11l, self.correlator.bird.Ploopl, self.correlator.bird.Pctl
                    
                else:
                    # self.correlator.bird.Q = self.correlator.resum.makeQ(self.correlator.bird.f)
                    # # P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.bird.setShapefit(0.0, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio, 
                    # #                                             IRPs_all = [self.correlator.IRPs11_new, self.correlator.IRPsct_new, self.correlator.IRPsloop_new], power=power)
                    # # P11l, Pctl, Ploopl, IRPs11, IRPsct, IRPsloop = self.correlator.bird.setShapefit(0.0, factor_a = factor_a, factor_kp=factor_kp, xdata=self.correlator.co.k, sigma8_ratio=sigma8_ratio)
                
                    # fullIRPs11 = np.einsum("lpn,pnk,pi->lik", self.correlator.bird.Q[0], IRPs11, self.correlator.co.l11)
                    # fullIRPsct = np.einsum("lpn,pnk,pi->lik", self.correlator.bird.Q[1], IRPsct, self.correlator.co.lct)
                    # fullIRPsloop = np.einsum("lpn,pink->lik", self.correlator.bird.Q[1], IRPsloop)
                    
                    
                    # # ratio = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(self.correlator.co.k/factor_kp)))*sigma8_ratio
                    
                    # P11l += fullIRPs11
                    # Pctl += fullIRPsct
                    # Ploopl += fullIRPsloop
                    
                    self.correlator.resum.Ps(self.correlator.bird, setPs=False, makeIR=False, init=True)
                    
                    factor_2n = np.concatenate((2 * [self.correlator.co.Na * [sigma8_ratio ** (n + 1)] for n in range(self.correlator.co.NIR)]))

                    IRPs11 = np.einsum("n,lnk->lnk", sigma8_ratio*factor_2n, self.correlator.bird.IRPs11)
                    IRPsct = np.einsum("n,lnk->lnk", sigma8_ratio*factor_2n, self.correlator.bird.IRPsct)
                    IRPsloop = np.einsum("n,lmnk->lmnk", sigma8_ratio**2*factor_2n, self.correlator.bird.IRPsloop)
                    
                    fullIRPs11 = np.einsum("lpn,pnk,pi->lik", self.correlator.bird.Q[0], IRPs11, self.correlator.co.l11)
                    fullIRPsct = np.einsum("lpn,pnk,pi->lik", self.correlator.bird.Q[1], IRPsct, self.correlator.co.lct)
                    fullIRPsloop = np.einsum("lpn,pink->lik", self.correlator.bird.Q[1], IRPsloop)
                    
                    P11l = self.correlator.bird.P11l*sigma8_ratio + fullIRPs11
                    Pctl = self.correlator.bird.Pctl*sigma8_ratio + fullIRPsct
                    Ploopl = self.correlator.bird.Ploopl*sigma8_ratio**2 + fullIRPsloop
                    
            else:
                # self.correlator.bird.Q = self.correlator.resum.makeQ(self.correlator.bird.f)

                if self.Shapefit == False:
                    factor_m = 0.0
                ratio = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(self.correlator.co.k/factor_kp)))*sigma8_ratio
                
                # P11l = np.einsum("jk, mnk->jmnk", ratio, self.P11l)
                # Pctl = np.einsum("jk, mnk->jmnk", ratio, self.Pctl)
                # Ploopl = np.einsum("jk, mnk->jmnk", ratio, self.Ploopl)
            
                P11l = self.correlator.bird.P11l*ratio
                Pctl = self.correlator.bird.Pctl*ratio
                Ploopl = self.correlator.bird.Ploopl*ratio ** 2
            
            # Pstl = np.exp(factor_m/factor_a*np.tanh(factor_a*np.log(self.correlator.co.k/factor_kp)))*sigma8_ratio*self.correlator.bird.Pstl
            
            # P11l_AP, Pctl_AP, Ploopl_AP = self.correlator.projection.AP(
            #     bird=self.correlator.bird, q=[alpha_perp, alpha_par], overwrite=False,
            #     PS=[P11l, Ploopl, Pctl]
            # )
            
            # P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = [], [], [], []
            # for i in range(len(factor_m)):
            #     AP_P11l, AP_Pctl, AP_Ploopl, AP_Pstl = self.correlator.projection.AP(bird = self.correlator.bird, q = [alpha_perp[i], alpha_par[i]], 
            #     overwrite = False, PS=[P11l[i], Ploopl[i], Pctl[i], self.correlator.bird.Pstl])
            #     P11l_AP.append(AP_P11l)
            #     Pctl_AP.append(AP_Pctl)
            #     Ploopl_AP.append(AP_Ploopl)
            #     Pstl_AP.append(AP_Pstl)
                
            
            
            
            P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = self.correlator.projection.AP(
                bird=self.correlator.bird, q=[alpha_perp, alpha_par], overwrite=False,
                PS=[P11l, Ploopl, Pctl, self.correlator.bird.Pstl]
            )
            
            # P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = self.correlator.projection.AP(
            #     bird=self.correlator.bird, q=[alpha_perp, alpha_par], overwrite=False,
            #     PS=[P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP]
            # )

            
            # P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = self.correlator.projection.AP(
            #     bird=self.correlator.bird, q=[alpha_perp, alpha_par], overwrite=False,
            #     PS=[P11l, Ploopl, Pctl, Pstl]
            # )
            
            # print(np.shape(P11l_AP))
            
            # a, b, c = list([P11l_AP, Ploopl_AP, Pctl_AP])
            if self.correlator.config["with_window"] == True:
                # P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP = self.correlator.projection.Window(self.correlator.bird, PS = list([P11l_AP, Ploopl_AP, Pctl_AP]))
                P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP = self.correlator.projection.Window(self.correlator.bird, PS = list([P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP]))
            
            self.kin = self.correlator.projection.xout
            
            if self.corr_convert == False:
            
                if self.window is None:
                    P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP = self.correlator.projection.xbinning(self.correlator.bird, PS_all = list([P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP]))
                else:
                    P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP = self.correlator.projection.xdata(self.correlator.bird, PS=[P11l_AP, Pctl_AP, Ploopl_AP, Pstl_AP])
    
                Plin, Ploop = self.correlator.bird.formatTaylorPs(Ps=[P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP], kdata=self.correlator.projection.xout)
                # Plin, Ploop = self.correlator.bird.formatTaylorPs(Ps=[P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP], kdata=self.correlator.co.k)
                
                Plin = np.swapaxes(np.reshape(Plin, (self.correlator.co.Nl, Plin.shape[-2]//self.correlator.co.Nl, Plin.shape[-1]))[:, :, 1:], axis1=1, axis2=2)
                
                Ploop = np.swapaxes(np.reshape(Ploop, (self.correlator.co.Nl, Ploop.shape[-2]//self.correlator.co.Nl, Ploop.shape[-1]))[:, :, 1:], axis1=1, axis2=2)
                
                # print(np.shape(Plin), np.shape(Ploop))
                
                # if self.pardict['do_corr']:
                    
    
            # return Plin, Ploop
                return Plin[:3], Ploop[:3]
            
            else:
                C11l_AP, Cloopl_AP, Cctl_AP, Cstl_AP = self.correlator.pk2xi_fun(bird = [P11l_AP, Ploopl_AP, Pctl_AP, Pstl_AP], output=True)
                
                if self.window is None:
                    C11l_AP, Cloopl_AP, Cctl_AP, Cstl_AP = self.correlator.projection.xbinning(self.correlator.bird, CF_all = list([C11l_AP, Cloopl_AP, Cctl_AP, Cstl_AP]))
                else:
                    C11l_AP, Cctl_AP, Cloopl_AP, Cstl_AP = self.correlator.projection.xdata(self.correlator.bird, CF=[C11l_AP, Cctl_AP, Cloopl_AP, Cstl_AP])
    
                Clin, Cloop = self.correlator.bird.formatTaylorCf(CF=[C11l_AP, Cloopl_AP, Cctl_AP, Cstl_AP], sdata=self.correlator.projection.xout)
    
                Clin = np.swapaxes(np.reshape(Clin, (self.correlator.co.Nl, Clin.shape[-2]//self.correlator.co.Nl, Clin.shape[-1]))[:, :, 1:], axis1=1, axis2=2)
                
                Cloop = np.swapaxes(np.reshape(Cloop, (self.correlator.co.Nl, Cloop.shape[-2]//self.correlator.co.Nl, Cloop.shape[-1]))[:, :, 1:], axis1=1, axis2=2)
                    
    
            # return Plin, Ploop
                return Clin[:3], Cloop[:3]
            
        # return Pctl_AP, Ploopl_AP

    def compute_hybrid(self, params):

        omega_rat = self.valueref[3] / self.valueref[2]
        omega_cdm = (params[3] - self.omega_nu) / (1.0 + omega_rat)
        omega_b = omega_rat * omega_cdm

        coords = [self.valueref[0], self.valueref[1], omega_cdm, omega_b]
        Plin, Ploop = self.compute_pk(coords)

        self.correlator.bird.P11l = np.einsum(
            "n,lnk->lnk", 1.0 / np.array([1.0, 2.0 * self.fN, self.fN ** 2]), Plin[:, ::-1, :]
        )
        self.correlator.bird.Ploopl = np.einsum(
            "n,lnk->lnk",
            1.0
            / np.array(
                [
                    self.fN ** 2,
                    self.fN ** 3,
                    self.fN ** 4,
                    self.fN,
                    self.fN ** 2,
                    self.fN ** 3,
                    self.fN,
                    self.fN ** 2,
                    self.fN,
                    self.fN,
                    self.fN ** 2,
                    1.0,
                    self.fN,
                    self.fN ** 2,
                    1.0,
                    self.fN,
                    1.0,
                    1.0,
                    self.fN,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
            Ploop[:, :22, :],
        )
        self.correlator.bird.Pctl = np.einsum(
            "n,lnk->lnk",
            1.0 / np.array([2.0, 2.0, 2.0, 2.0 * self.fN, 2.0 * self.fN, 2.0 * self.fN]),
            Ploop[:, 22:28],
        )
        if self.correlator.bird.with_nlo_bias:
            self.correlator.bird.Pnlol[:, 0, :] = Ploop[:, 28]

        return self.modify_template(params[:3])

    def compute_model(self, cvals, plin, ploop, x_data):

        # if (self.direct or self.template) and not self.pardict["do_hex"]:
        #     plin0, plin2 = plin
        #     ploop0, ploop2 = ploop
        # else:
        #     plin0, plin2, plin4 = plin
        #     ploop0, ploop2, ploop4 = ploop
        try:
            plin0, plin2 = plin
            ploop0, ploop2 = ploop
        except:
            plin0, plin2, plin4 = plin
            ploop0, ploop2, ploop4 = ploop

        b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad = cvals
        
        # print(cvals)
        
        # cemono = 3.0*(cemono - ce1)
        # cequad = 1.5*cequad

        # the columns of the Ploop data files.
        cvals = np.array(
            [
                np.ones(np.shape(b1)),
                b1,
                b2,
                b3,
                b4,
                b1 * b1,
                b1 * b2,
                b1 * b3,
                b1 * b4,
                b2 * b2,
                b2 * b4,
                b4 * b4,
                b1 * cct / self.k_nl ** 2,
                b1 * cr1 / self.k_m ** 2,
                b1 * cr2 / self.k_m ** 2,
                cct / self.k_nl ** 2,
                cr1 / self.k_m ** 2,
                cr2 / self.k_m ** 2,
                ce1,
                cemono / self.k_m ** 2,
                cequad / self.k_m ** 2,
            ]
        )

        P0_lin, P0_loop = plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2], np.sum(cvals * ploop0, axis=1)
        P2_lin, P2_loop = plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2], np.sum(cvals * ploop2, axis=1)
        if self.pardict["do_hex"]:
            P4_lin, P4_loop = plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2], np.sum(cvals * ploop4, axis=1)
            
        # print(np.shape(P0_lin))
        
        # print(len(self.kin), len(x_data[0]))
        
        # print(np.shape(P0_lin), np.shape(P2_loop))
        
        # if self.corr_convert == True:
        #     if self.pardict["do_hex"]:
        #         P_model_lin = np.concatenate([[P0_lin], [P2_lin], [P4_lin]], axis=0)
        #         P_model_loop = np.concatenate([[P0_loop], [P2_loop], [P4_loop]], axis=0)
        #     else:
        #         P_model_lin = np.concatenate([[P0_lin], [P2_lin]], axis=0)
        #         P_model_loop = np.concatenate([[P0_loop], [P2_loop]], axis=0)

        #     return P_model_lin, P_model_loop, None
        
        # else:
        
        # x_data[0] = self.kin
        # x_data[1] = self.kin
            
        P0_interp_lin = [
            sp.interpolate.splev(x_data[0], sp.interpolate.splrep(self.kin, P0_lin[:, i])) for i in range(len(b1))
        ]
        P0_interp_loop = [
            sp.interpolate.splev(x_data[0], sp.interpolate.splrep(self.kin, P0_loop[:, i])) for i in range(len(b1))
        ]
        P2_interp_lin = [
            sp.interpolate.splev(x_data[1], sp.interpolate.splrep(self.kin, P2_lin[:, i])) for i in range(len(b1))
        ]
        P2_interp_loop = [
            sp.interpolate.splev(x_data[1], sp.interpolate.splrep(self.kin, P2_loop[:, i])) for i in range(len(b1))
        ]
        
        # print(np.shape(P0_lin), np.shape(P0_interp_lin))
        P0_interp = [P0_interp_lin[i] + P0_interp_loop[i] for i in range(len(b1))]
        P2_interp = [P2_interp_lin[i] + P2_interp_loop[i] for i in range(len(b1))]
        # P0_interp = [P0_interp_lin[i]  for i in range(len(b1))]
        # P2_interp = [P2_interp_lin[i]  for i in range(len(b1))]
        
        # P0_interp = [P0_lin[:, i] + P0_loop[:, i] for i in range(len(b1))]
        # P2_interp = [P2_lin[:, i] + P2_loop[:, i] for i in range(len(b1))]
        
        if self.pardict["do_hex"]:
            P4_interp_lin = [
                sp.interpolate.splev(x_data[2], sp.interpolate.splrep(self.kin, P4_lin[:, i])) for i in range(len(b1))
            ]
            P4_interp_loop = [
                sp.interpolate.splev(x_data[2], sp.interpolate.splrep(self.kin, P4_loop[:, i])) for i in range(len(b1))
            ]
            P4_interp = [P4_interp_lin[i] + P4_interp_loop[i] for i in range(len(b1))]
                
                # P4_interp = [P4_lin[:, i] + P4_loop[:, i] for i in range(len(b1))]
            # else:
            #     P0_interp_lin = [
            #         sp.interpolate.splev(self.kmode, sp.interpolate.splrep(self.kin, P0_lin[:, i])) for i in range(len(b1))
            #     ]
            #     P0_interp_loop = [
            #         sp.interpolate.splev(self.kmode, sp.interpolate.splrep(self.kin, P0_loop[:, i])) for i in range(len(b1))
            #     ]
            #     P2_interp_lin = [
            #         sp.interpolate.splev(self.kmode, sp.interpolate.splrep(self.kin, P2_lin[:, i])) for i in range(len(b1))
            #     ]
            #     P2_interp_loop = [
            #         sp.interpolate.splev(self.kmode, sp.interpolate.splrep(self.kin, P2_loop[:, i])) for i in range(len(b1))
            #     ]
            #     P0_interp = [P0_interp_lin[i] + P0_interp_loop[i] for i in range(len(b1))]
            #     P2_interp = [P2_interp_lin[i] + P2_interp_loop[i] for i in range(len(b1))]
                
            #     # P0_interp = [P0_lin[:, i] + P0_loop[:, i] for i in range(len(b1))]
            #     # P2_interp = [P2_lin[:, i] + P2_loop[:, i] for i in range(len(b1))]
                
            #     if self.pardict["do_hex"]:
            #         P4_interp_lin = [
            #             sp.interpolate.splev(self.kmode, sp.interpolate.splrep(self.kin, P4_lin[:, i])) for i in range(len(b1))
            #         ]
            #         P4_interp_loop = [
            #             sp.interpolate.splev(self.kmode, sp.interpolate.splrep(self.kin, P4_loop[:, i])) for i in range(len(b1))
            #         ]
            #         P4_interp = [P4_interp_lin[i] + P4_interp_loop[i] for i in range(len(b1))]
                    
                # P0_lin_all = []
                # P2_lin_all = []
                # P0_loop_all = []
                # P2_loop_all = []
                # P0_interp_all = []
                # P2_interp_all = []
                # if self.pardict['do_hex']:
                #     P4_lin_all = []
                #     P4_loop_all = []
                #     P4_interp_all = []
                
                
    
            # if self.pardict["do_corr"]:
            #     C0 = np.exp(-self.k_m * x_data[0]) * self.k_m ** 2 / (4.0 * np.pi * x_data[0])
            #     C1 = -self.k_m ** 2 * np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2)
            #     C2 = (
            #         np.exp(-self.k_m * x_data[1])
            #         * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
            #         / (4.0 * np.pi * x_data[1] ** 3)
            #     )
    
            #     P0_interp += np.outer(ce1, C0) + np.outer(cemono, C1)
            #     P2_interp += np.outer(cequad, C2)
    
            # if self.pardict["do_hex"]:
            #     P_model_lin = np.concatenate([P0_interp_lin, P2_interp_lin, P4_interp_lin], axis=1)
            #     P_model_loop = np.concatenate([P0_interp_loop, P2_interp_loop, P4_interp_loop], axis=1)
            #     P_model_interp = np.concatenate([P0_interp, P2_interp, P4_interp], axis=1)
            # else:
            #     P_model_lin = np.concatenate([P0_interp_lin, P2_interp_lin], axis=1)
            #     P_model_loop = np.concatenate([P0_interp_loop, P2_interp_loop], axis=1)
            #     P_model_interp = np.concatenate([P0_interp, P2_interp], axis=1)
            
        if self.pardict["do_hex"]:
            P_model_lin = np.concatenate([P0_lin, P2_lin, P4_lin], axis=1)
            P_model_loop = np.concatenate([P0_loop, P2_loop, P4_loop], axis=1)
            P_model_interp = np.concatenate([P0_interp, P2_interp, P4_interp], axis=1)
        else:
            P_model_lin = np.concatenate([P0_lin, P2_lin], axis=1)
            P_model_loop = np.concatenate([P0_loop, P2_loop], axis=1)
            P_model_interp = np.concatenate([P0_interp, P2_interp], axis=1)

        return P_model_lin.T, P_model_loop.T, P_model_interp.T

    # Ignore names, works for both power spectrum and correlation function
    def get_Pi_for_marg(self, ploop, b1, shot_noise, x_data, growth = None, MinF = False):

        if self.pardict["do_marg"]:

            # if (self.direct or self.template) and not self.pardict["do_hex"]:
            #     ploop0, ploop2 = ploop
            # else:
            #     ploop0, ploop2, ploop4 = ploop
            
            try:
                ploop0, ploop2 = ploop
            except:
                ploop0, ploop2, ploop4 = ploop
                
            # if self.corr_convert == True:
            #     Pb3 = np.concatenate(
            #         np.swapaxes(
            #             [
            #                 np.transpose(ploop0[:, 3, :] + b1[0]*ploop0[:, 7, :]), 
            #                 np.transpose(ploop2[:, 3, :] + b1[0]*ploop2[:, 7, :])
                            
            #             ],
            #             axis1=1,
            #             axis2=2,
            #         )
            #     )
            #     Pcct = np.concatenate(
            #         np.swapaxes(
            #             [
            #                 np.transpose(ploop0[:, 15, :] + b1[0]*ploop0[:, 12, :]), 
            #                 np.transpose(ploop2[:, 15, :] + b1[0]*ploop2[:, 12, :])
                            
            #             ],
            #             axis1=1,
            #             axis2=2,
            #         )
            #     )
            #     Pcr1 = np.concatenate(
            #         np.swapaxes(
            #             [
            #                 np.transpose(ploop0[:, 16, :] + b1[0]*ploop0[:, 13, :]), 
            #                 np.transpose(ploop2[:, 16, :] + b1[0]*ploop2[:, 13, :])
                        
            #             ],
            #             axis1=1,
            #             axis2=2,
            #         )
            #     )
            #     Pcr2 = np.concatenate(
            #         np.swapaxes(
            #             [
            #                 np.transpose(ploop0[:, 17, :] + b1[0]*ploop0[:, 14, :]), 
            #                 np.transpose(ploop2[:, 17, :] + b1[0]*ploop2[:, 14, :])
                            
            #             ],
            #             axis1=1,
            #             axis2=2,
            #         )
            #     )
            #     Pce1 = np.concatenate(
            #         np.swapaxes(
            #             [
            #                 np.transpose(ploop0[:, 18, :]), 
            #                 np.transpose(ploop2[:, 18, :])
            #             ],
            #             axis1=1,
            #             axis2=2,
            #         )
            #     )
            #     Pcemono = np.concatenate(
            #         np.swapaxes(
            #             [
            #                 np.transpose(ploop0[:, 19, :]), 
            #                 np.transpose(ploop2[:, 19, :])
            #                 # [splev(x_data[1], splrep(self.kin, (2.0/5.0/growth[i])*ploop2[:, 20, i])) for i, b in enumerate(b1)],
            #             ],
            #             axis1=1,
            #             axis2=2,
            #         )
            #     )
            #     Pcequad = np.concatenate(
            #         np.swapaxes(
            #             [
            #                 np.transpose(ploop0[:, 20, :]), 
            #                 np.transpose(ploop2[:, 20, :])
            #             ],
            #             axis1=1,
            #             axis2=2,
            #         )
            #     )

            #     if self.pardict["do_hex"]:

            #         Pb3 = np.concatenate(
            #             [
            #                 Pb3,
            #                 np.array(
            #                     [
            #                         np.transpose(ploop4[:, 3, :] + b1[0]*ploop4[:, 7, :]), 
            #                     ]
            #                 ).T,
            #             ]
            #         )

            #         Pcct = np.concatenate(
            #             [
            #                 Pcct,
            #                 np.array(
            #                     [
            #                         np.transpose(ploop4[:, 15, :] + b1[0]*ploop4[:, 12, :]), 
            #                     ]
            #                 ).T,
            #             ]
            #         )
            #         Pcr1 = np.concatenate(
            #             [
            #                 Pcr1,
            #                 np.array(
            #                     [
            #                         np.transpose(ploop4[:, 16, :] + b1[0]*ploop4[:, 13, :]), 
            #                     ]
            #                 ).T,
            #             ]
            #         )
            #         Pcr2 = np.concatenate(
            #             [
            #                 Pcr2,
            #                 np.array(
            #                     [
            #                         np.transpose(ploop4[:, 17, :] + b1[0]*ploop4[:, 14, :]), 
            #                     ]
            #                 ).T,
            #             ]
            #         )
            #         Pce1 = np.concatenate(
            #             [
            #                 Pce1,
            #                 np.transpose(ploop4[:, 18, :])
            #             ]
            #         )
            #         Pcemono = np.concatenate(
            #             [
            #                 Pcemono,
            #                 np.transpose(ploop4[:, 19, :])
                            
            #             ]
            #         )
            #         Pcequad = np.concatenate(
            #             [
            #                 Pcequad,
            #                 np.transpose(ploop4[:, 20, :])
                            
            #             ]
            #         )
            # else:

            Pb3 = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 3, i] + b * ploop0[:, 7, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 3, i] + b * ploop2[:, 7, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcct = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 15, i] + b * ploop0[:, 12, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 15, i] + b * ploop2[:, 12, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcr1 = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 16, i] + b * ploop0[:, 13, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 16, i] + b * ploop2[:, 13, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcr2 = np.concatenate(
                np.swapaxes(
                    [
                        [
                            splev(x_data[0], splrep(self.kin, ploop0[:, 17, i] + b * ploop0[:, 14, i]))
                            for i, b in enumerate(b1)
                        ],
                        [
                            splev(x_data[1], splrep(self.kin, ploop2[:, 17, i] + b * ploop2[:, 14, i]))
                            for i, b in enumerate(b1)
                        ],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pce1 = np.concatenate(
                np.swapaxes(
                    [
                        [splev(x_data[0], splrep(self.kin, ploop0[:, 18, i])) for i, b in enumerate(b1)],
                        [splev(x_data[1], splrep(self.kin, ploop2[:, 18, i])) for i, b in enumerate(b1)],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcemono = np.concatenate(
                np.swapaxes(
                    [
                        [splev(x_data[0], splrep(self.kin, ploop0[:, 19, i])) for i, b in enumerate(b1)],
                        [splev(x_data[1], splrep(self.kin, ploop2[:, 19, i])) for i, b in enumerate(b1)],
                        # [splev(x_data[1], splrep(self.kin, (2.0/5.0/growth[i])*ploop2[:, 20, i])) for i, b in enumerate(b1)],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            Pcequad = np.concatenate(
                np.swapaxes(
                    [
                        [splev(x_data[0], splrep(self.kin, ploop0[:, 20, i])) for i, b in enumerate(b1)],
                        [splev(x_data[1], splrep(self.kin, ploop2[:, 20, i])) for i, b in enumerate(b1)],
                    ],
                    axis1=1,
                    axis2=2,
                )
            )

            if self.pardict["do_hex"]:

                Pb3 = np.concatenate(
                    [
                        Pb3,
                        np.array(
                            [
                                splev(x_data[2], splrep(self.kin, ploop4[:, 3, i] + b * ploop4[:, 7, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )

                Pcct = np.concatenate(
                    [
                        Pcct,
                        np.array(
                            [
                                splev(x_data[2], splrep(self.kin, ploop4[:, 15, i] + b * ploop4[:, 12, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )
                Pcr1 = np.concatenate(
                    [
                        Pcr1,
                        np.array(
                            [
                                splev(x_data[2], splrep(self.kin, ploop4[:, 16, i] + b * ploop4[:, 13, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )
                Pcr2 = np.concatenate(
                    [
                        Pcr2,
                        np.array(
                            [
                                splev(x_data[2], splrep(self.kin, ploop4[:, 17, i] + b * ploop4[:, 14, i]))
                                for i, b in enumerate(b1)
                            ]
                        ).T,
                    ]
                )
                Pce1 = np.concatenate(
                    [
                        Pce1,
                        np.array([splev(x_data[2], splrep(self.kin, ploop4[:, 18, i])) for i, b in enumerate(b1)]).T,
                    ]
                )
                Pcemono = np.concatenate(
                    [
                        Pcemono,
                        np.array([splev(x_data[2], splrep(self.kin, ploop4[:, 19, i])) for i, b in enumerate(b1)]).T,
                    ]
                )
                Pcequad = np.concatenate(
                    [
                        Pcequad,
                        np.array([splev(x_data[2], splrep(self.kin, ploop4[:, 20, i])) for i, b in enumerate(b1)]).T,
                    ]
                )

            if self.pardict["do_corr"]:

                # C0 = np.concatenate(
                #     [np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0]), np.zeros(len(x_data[1]))]
                # )  # shot-noise mono
                # C1 = np.concatenate(
                #     [-np.exp(-self.k_m * x_data[0]) / (4.0 * np.pi * x_data[0] ** 2), np.zeros(len(x_data[1]))]
                # )  # k^2 mono
                # C2 = np.concatenate(
                #     [
                #         np.zeros(len(x_data[0])),
                #         np.exp(-self.k_m * x_data[1])
                #         * (3.0 + 3.0 * self.k_m * x_data[1] + self.k_m ** 2 * x_data[1] ** 2)
                #         / (4.0 * np.pi * x_data[1] ** 3),
                #     ]
                # )  # k^2 quad

                # if self.pardict["do_hex"]:
                #     C0 = np.concatenate([C0, np.zeros(len(x_data[2]))])  # shot-noise mono
                #     C1 = np.concatenate([C1, np.zeros(len(x_data[2]))])  # k^2 mono
                #     C2 = np.concatenate([C2, np.zeros(len(x_data[2]))])  # k^2 quad
                
                # length = np.int32(len(x_data[0]))
                # Pce1[:length, :] = np.einsum('ij, i -> ij', Pce1[:length, :], self.k_m**2*np.exp(x_data[0]*(1.0 - self.k_m))*shot_noise)
                # Pcemono[:length, :] = np.einsum('ij, i -> ij',Pcemono[:length, :], self.k_m**2*np.exp(x_data[0]*(1.0 - self.k_m))*shot_noise)
                # Pcequad[length:2*length, :] = np.einsum('ij, i -> ij',Pcequad[length:2*length, :], np.exp(x_data[1]*(1.0 - self.k_m))*(3.0 + 3.0*self.k_m*x_data[1] + self.k_m**2*x_data[1]**2)/(3.0 + 3.0*x_data[1] + x_data[1]**2))

                Pi = np.array(
                    [
                        Pb3,  # *b3
                        Pcct / self.k_nl ** 2,  # *cct
                        Pcr1 / self.k_m ** 2,  # *cr1
                        Pcr2 / self.k_m ** 2,  # *cr2
                        # np.tile(C0, (len(b1), 1)).T * self.k_m ** 2 * shot_noise,  # ce1
                        # np.tile(C1, (len(b1), 1)).T * self.k_m ** 2 * shot_noise,  # cemono
                        # np.tile(C2, (len(b1), 1)).T * shot_noise,  # cequad
                        # 2.0 * Pnlo / self.k_m ** 4,  # bnlo
                        # Pce1*self.k_m**2*np.exp(x_data[0]*(1.0 - self.k_m))*shot_noise,
                        # Pcemono*self.k_m**2*np.exp(x_data[0]*(1.0 - self.k_m))*shot_noise,
                        # Pcequad*np.exp(x_data[1]*(1.0 - self.k_m))*(3.0 + 3.0*self.k_m*x_data[1] + self.k_m**2*x_data[1]**2)/(3.0 + 3.0*x_data[1] + x_data[1]**2)
                        Pce1*shot_noise,
                        Pcemono*shot_noise,
                        Pcequad*shot_noise
                    ]
                )

            else:

                """Onel0 = np.concatenate([np.ones(len(x_data[0])), np.zeros(len(x_data[1]))])  # shot-noise mono
                kl0 = np.concatenate([x_data[0], np.zeros(len(x_data[1]))])  # k^2 mono
                kl2 = np.concatenate([np.zeros(len(x_data[0])), x_data[1]])  # k^2 quad

                if self.pardict["do_hex"]:
                    Onel0 = np.concatenate([Onel0, np.zeros(len(x_data[2]))])  # shot-noise mono
                    kl0 = np.concatenate([kl0, np.zeros(len(x_data[2]))])  # k^2 mono
                    kl2 = np.concatenate([kl2, np.zeros(len(x_data[2]))])  # k^2 quad"""
                    
                # print(np.shape(Pb3), np.shape(Pcct), np.shape(Pcr1), np.shape(Pcr2), np.shape(Pce1), np.shape(Pcemono), np.shape(Pcequad), shot_noise)
                
                if MinF == False:
                    if self.Nl == 3:
                        Pi = np.array(
                            [
                                Pb3,  # *b3
                                Pcct / self.k_nl ** 2,  # *cct
                                Pcr1 / self.k_m ** 2,  # *cr1
                                Pcr2 / self.k_m ** 2,  # *cr2
                                Pce1 * shot_noise,  # *ce1
                                # Pce1,  # *ce1
                                Pcemono / self.k_m ** 2 * shot_noise,  # *cemono
                                Pcequad / self.k_m ** 2 * shot_noise,  # *cequad
                                # (3.0*Pcequad / self.k_m ** 2 * shot_noise - Pcemono / self.k_m ** 2 * shot_noise)/2.0
                                # 2.0 * Pnlo / self.k_m ** 4,  # bnlo
                            ]
                        )
                    else:
                        if self.pardict['prior'] == 'MaxF':
                            Pi = np.array(
                                [
                                    Pb3,  # *b3
                                    Pcct / self.k_nl ** 2,  # *cct
                                    Pcr1 / self.k_m ** 2,  # *cr1
                                    Pce1 * shot_noise,  # *ce1
                                    # Pce1,  # *ce1
                                    # Pcemono / self.k_m ** 2 * shot_noise,  # *cemono
                                    Pcequad / self.k_m ** 2 * shot_noise,  # *cequad
                                    # (3.0*Pcequad / self.k_m ** 2 * shot_noise - Pcemono / self.k_m ** 2 * shot_noise)/2.0
                                    # 2.0 * Pnlo / self.k_m ** 4,  # bnlo
                                ]
                            )
                        else:
                            Pi = np.array(
                                [
                                    Pb3,  # *b3
                                    Pcct / self.k_nl ** 2,  # *cct
                                    Pcr1 / self.k_m ** 2,  # *cr1
                                    Pce1 * shot_noise,  # *ce1
                                    # Pce1,  # *ce1
                                    Pcemono / self.k_m ** 2 * shot_noise,  # *cemono
                                    Pcequad / self.k_m ** 2 * shot_noise,  # *cequad
                                    # (3.0*Pcequad / self.k_m ** 2 * shot_noise - Pcemono / self.k_m ** 2 * shot_noise)/2.0
                                    # 2.0 * Pnlo / self.k_m ** 4,  # bnlo
                                ]
                            )
                else:
                    if self.Nl == 3:
                        Pi = np.array(
                            [
                                Pcct / self.k_nl ** 2,  # *cct
                                Pcr1 / self.k_m ** 2,  # *cr1
                                Pcr2 / self.k_m ** 2,  # *cr2
                                Pce1 * shot_noise,  # *ce1
                                # Pce1,  # *ce1
                                Pcequad / self.k_m ** 2 * shot_noise,  # *cequad
                                # (3.0*Pcequad / self.k_m ** 2 * shot_noise - Pcemono / self.k_m ** 2 * shot_noise)/2.0
                                # 2.0 * Pnlo / self.k_m ** 4,  # bnlo
                            ]
                        )
                    else:
                        Pi = np.array(
                            [
                                Pcct / self.k_nl ** 2,  # *cct
                                Pcr1 / self.k_m ** 2,  # *cr1
                                Pce1 * shot_noise,  # *ce1
                                Pcequad / self.k_m ** 2 * shot_noise,  # *cequad
                            ]
                        )

        else:

            Pi = None

        return Pi

    def compute_chi2(self, P_model, data):

        # # Compute the chi_squared
        diff = P_model - data["fit_data"][:, None]
        return np.einsum("ip,ij,jp->p", diff, data["cov_inv"], diff)
        
        # result = np.dot(P_model.T, np.dot(data["cov_inv"], P_model))
        # - 2.0 * np.dot(data["invcovdata"], P_model)
        # + data["chi2data"]
        
        # result = np.einsum("kd,kd->d", P_model, np.dot(data["cov_inv"], P_model))
        # - 2.0 * np.dot(data["invcovdata"], P_model)
        # + data["chi2data"]
        
        # print(result)
        
        # return result[0]
    
    def compute_chi2_marginalised(self, P_model, Pi, data, onebin = False, eft_priors = None, MinF = False):


        Pi = np.transpose(Pi, axes=(2, 0, 1))
        Pimult = np.inner(Pi, data["cov_inv"])
        Covbi = np.einsum("dpk,dqk->dpq", Pimult, Pi)
        
        # print(np.shape(Covbi))
        # print(np.shape(Covbi))
        if self.eft_priors is not None:
            if onebin == False:
                if eft_priors is None:
                    Covbi += np.diag(1.0 / np.tile(self.eft_priors, len(data["x_data"]))) ** 2
                else:
                    Covbi += np.diag(1.0 / eft_priors)**2
            else:
                Covbi += np.reshape(np.diag(1.0 / np.tile(self.eft_priors, 1)) ** 2, (1, len(self.eft_priors), len(self.eft_priors)))
        else:
            # if MinF == False:
            #     Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0, 0, 0])]), len(Covbi), axis=0)
            # else:
            #     Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0])]), len(Covbi), axis=0)
            
            if MinF == False:
                if self.Nl == 3:
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0, 0, 0])]), len(Covbi), axis=0)
                else:
                    # Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0, 0])]), len(Covbi), axis=0)
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0])]), len(Covbi), axis=0)

            else:
                if self.Nl == 3:
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0])]), len(Covbi), axis=0)
                else:
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0])]), len(Covbi), axis=0)
            
            # Covbi += np.repeat(np.array([np.diag([0, 0, 0, 10e100, 0, 10e100, 0])]), len(Covbi), axis=0)

            # Covbi += np.repeat(np.array([np.diag([0, 0, 0, 10e100, 0, 0])]), len(Covbi), axis=0)

            
                
        vectorbi = np.einsum("dpk,kd->dp", Pimult, P_model) - np.dot(Pi, data["invcovdata"])
        # vectorbi = np.einsum("dq, qmd->dm",np.einsum("dp,pq->dq", P_model.T, data['cov_inv']), Pi) - np.einsum()
        chi2nomar = (
            np.einsum("kd,kd->d", P_model, np.dot(data["cov_inv"], P_model))
            - 2.0 * np.dot(data["invcovdata"], P_model)
            + data["chi2data"]
        )
        # chi2mar = -np.einsum("dp,dp->d", vectorbi, np.linalg.solve(Covbi, vectorbi)) + np.log(np.linalg.det(Covbi))
        chi2mar = -np.einsum("dp,dp->d", vectorbi, np.linalg.solve(Covbi, vectorbi)) + np.linalg.slogdet(Covbi)[1]
        chi_squared = chi2nomar + chi2mar
        # print(chi2mar, chi2nomar)

        return chi_squared
    
    def compute_bestfit_analytic(self, P_model, Pi, data, onebin = False, eft_priors = None, MinF=False):

        Pi = np.transpose(Pi, axes=(2, 0, 1))
        Pimult = np.inner(Pi, data["cov_inv"])
        Covbi = np.einsum("dpk,dqk->dpq", Pimult, Pi)
        if self.eft_priors is not None:
            if onebin == False:
                if eft_priors is None:
                    Covbi += np.diag(1.0 / np.tile(self.eft_priors, len(data["x_data"]))) ** 2
                else:
                    Covbi += np.diag(1.0 / eft_priors)**2
            else:
                Covbi += np.reshape(np.diag(1.0 / np.tile(self.eft_priors, 1)) ** 2, (1, len(self.eft_priors), len(self.eft_priors)))
        else:
            # Covbi += np.repeat(np.array([np.diag([0, 0, 0, 10e100, 0, 0, 0])]), len(Covbi), axis=0)
            # Covbi += np.repeat(np.array([np.diag([0, 0, 0, 10e100, 0, 0])]), len(Covbi), axis=0)
            if MinF == False:
                if self.Nl == 3:
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0, 0, 0])]), len(Covbi), axis=0)
                else:
                    # Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0, 0])]), len(Covbi), axis=0)
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0])]), len(Covbi), axis=0)

            else:
                if self.Nl == 3:
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0, 0])]), len(Covbi), axis=0)
                else:
                    Covbi += np.repeat(np.array([np.diag([0, 0, 0, 0])]), len(Covbi), axis=0)

        # Cinvbi = np.linalg.inv(Covbi)
        vectorbi = np.einsum("dpk,kd->dp", Pimult, P_model) - np.dot(Pi, data["invcovdata"])

        # return -np.einsum("dpq,dp->qd", Cinvbi, vectorbi)
        return -np.linalg.solve(Covbi, vectorbi)
    
    # def compute_chi2_marginalised(self, modelX, Pi, data, onebin):
        
    #     invcov = data['cov_inv']
    #     invcovdata = data['invcovdata']
    #     chi2data = data['chi2data']
        
    #     Pi = np.transpose(Pi, axes=(2, 0, 1))
        
    #     Pi = Pi[0]
        
    #     # print(np.shape(np.dot(invcov, Pi.T)))
        
    #     priormat = np.diagflat(1. / self.eft_priors**2)
        
    #     # Pimult = np.einsum('ab, bcd -> acd', invcov, Pi.T)
    #     # Covbi = np.einsum('abc, cde -> abde', Pi, Pimult)
    #     Covbi = np.dot(Pi, np.dot(invcov, Pi.T)) + priormat
    #     Cinvbi = np.linalg.inv(Covbi)
    #     vectorbi = np.dot(modelX.T, np.dot(invcov, Pi.T)) - np.dot(invcovdata, Pi.T)
    #     chi2nomar = np.dot(modelX.T, np.dot(invcov, modelX)) - 2. * np.dot(invcovdata, modelX) + chi2data
    #     chi2mar = - np.dot(vectorbi, np.dot(Cinvbi, vectorbi.T)) + np.log(np.abs(np.linalg.det(Covbi)))
    #     chi2tot = chi2mar + chi2nomar - priormat.shape[0] * np.log(2. * np.pi)
    #     #print (np.dot(modelX, np.dot(invcov, modelX)), -2. * np.dot(invcovdata, modelX), chi2data)
    #     #print (- np.dot(vectorbi, np.dot(Cinvbi, vectorbi)), np.log(np.abs(np.linalg.det(Covbi))))
    #     #print (chi2nomar, chi2mar )
    #     return chi2tot

    
    
    # def compute_bestfit_analytic(self, modelX, Pi, data, onebin):

    #     invcov = data['cov_inv']
    #     invcovdata = data['invcovdata']
        
    #     Pi = np.transpose(Pi, axes=(2, 0, 1))
        
    #     Pi = Pi[0]
        
    #     # print(np.shape(np.dot(invcov, Pi.T)))
        
    #     priormat = np.diagflat(1. / self.eft_priors**2)
        
    #     # Pimult = np.einsum('ab, bcd -> acd', invcov, Pi.T)
    #     # Covbi = np.einsum('abc, cde -> abde', Pi, Pimult)
    #     Covbi = np.dot(Pi, np.dot(invcov, Pi.T)) + priormat
    #     Cinvbi = np.linalg.inv(Covbi)
    #     vectorbi = np.dot(modelX.T, np.dot(invcov, Pi.T)) - np.dot(invcovdata, Pi.T)

    #     return -np.dot(Cinvbi, vectorbi.T)

    # def compute_chi2_marginalised(self, P_model, Pi, data):

    #     Pi = np.transpose(Pi, axes=(2, 0, 1))
    #     Pimult = np.dot(Pi, data["cov_inv"])
    #     Covbi = np.einsum("dpk,dqk->dpq", Pimult, Pi)
    #     Covbi += np.diag(1.0 / np.tile(self.eft_priors, len(data["x_data"]))) ** 2
    #     vectorbi = np.einsum("dpk,kd->dp", Pimult, P_model) - np.dot(Pi, data["invcovdata"])
    #     chi2nomar = (
    #         np.einsum("kd,kd->d", P_model, np.dot(data["cov_inv"], P_model))
    #         - 2.0 * np.dot(data["invcovdata"], P_model)
    #         + data["chi2data"]
    #     )
    #     chi2mar = -np.einsum("dp,dp->d", vectorbi, np.linalg.solve(Covbi, vectorbi)) + np.log(np.linalg.det(Covbi))
    #     chi_squared = chi2nomar + chi2mar

    #     return chi_squared

    # def compute_bestfit_analytic(self, P_model, Pi, data):

    #     Pi = np.transpose(Pi, axes=(2, 0, 1))
    #     Pimult = np.dot(Pi, data["cov_inv"])
    #     Covbi = np.einsum("dpk,dqk->dpq", Pimult, Pi)
    #     Covbi += np.diag(1.0 / np.tile(self.eft_priors, len(data["x_data"]))) ** 2
    #     Cinvbi = np.linalg.inv(Covbi)
    #     vectorbi = np.einsum("dpk,kd->dp", Pimult, P_model) - np.dot(Pi, data["invcovdata"])

    #     return -np.einsum("dpq,dp->qd", Cinvbi, vectorbi)

    def get_components(self, coords, cvals):

        if self.direct:
            plin, ploop = self.compute_model_direct(coords)
        else:
            plin, ploop = self.compute_pk(coords)

        plin0, plin2, plin4 = plin
        ploop0, ploop2, ploop4 = ploop

        b1, b2, b3, b4, cct, cr1, cr2, ce1, cemono, cequad, bnlo = cvals

        # the columns of the Ploop data files.
        cloop = np.array([1, b1, b2, b3, b4, b1 * b1, b1 * b2, b1 * b3, b1 * b4, b2 * b2, b2 * b4, b4 * b4])
        cvalsct = np.array(
            [
                2.0 * b1 * cct / self.k_nl ** 2,
                2.0 * b1 * cr1 / self.k_m ** 2,
                2.0 * b1 * cr2 / self.k_m ** 2,
                2.0 * cct / self.k_nl ** 2,
                2.0 * cr1 / self.k_m ** 2,
                2.0 * cr2 / self.k_m ** 2,
            ]
        )
        cnlo = 2.0 * b1 ** 2 * bnlo / self.k_m ** 4

        P0lin = plin0[0] + b1 * plin0[1] + b1 * b1 * plin0[2]
        P2lin = plin2[0] + b1 * plin2[1] + b1 * b1 * plin2[2]
        P0loop = np.dot(cloop, ploop0[:12, :])
        P2loop = np.dot(cloop, ploop2[:12, :])
        P0ct = np.dot(cvalsct, ploop0[12:-1, :])
        P2ct = np.dot(cvalsct, ploop2[12:-1, :])
        P0nlo = cnlo * ploop0[-1, :]
        P2nlo = cnlo * ploop2[-1, :]
        if self.pardict["do_hex"]:
            P4lin = plin4[0] + b1 * plin4[1] + b1 * b1 * plin4[2]
            P4loop = np.dot(cloop, ploop4[:12, :])
            P4ct = np.dot(cvalsct, ploop4[12:-1, :])
            P4nlo = cnlo * ploop4[-1, :]
            Plin = [P0lin, P2lin, P4lin]
            Ploop = [P0loop + P0nlo, P2loop + P2nlo, P4loop + P4nlo]
            Pct = [P0ct, P2ct, P4ct]
        else:
            Plin = [P0lin, P2lin]
            Ploop = [P0loop + P0nlo, P2loop + P2nlo]
            Pct = [P0ct, P2ct]

        if self.pardict["do_corr"]:
            C0 = np.exp(-self.k_m * self.kin) * self.k_m ** 2 / (4.0 * np.pi * self.kin)
            C1 = -self.k_m ** 2 * np.exp(-self.k_m * self.kin) / (4.0 * np.pi * self.kin ** 2)
            C2 = (
                np.exp(-self.k_m * self.kin)
                * (3.0 + 3.0 * self.k_m * self.kin + self.k_m ** 2 * self.kin ** 2)
                / (4.0 * np.pi * self.kin ** 3)
            )
            P0st = ce1 * C0 + cemono * C1
            P2st = cequad * C2
            P4st = np.zeros(len(self.kin))
        else:
            P0st = ce1 + cemono * self.kin ** 2 / self.k_m ** 2
            P2st = cequad * self.kin ** 2 / self.k_m ** 2
            P4st = np.zeros(len(self.kin))
        if self.pardict["do_hex"]:
            Pst = [P0st, P2st, P4st]
        else:
            Pst = [P0st, P2st]

        return Plin, Ploop, Pct, Pst
    
    # def pk2xi_fun(self, P_model, Ploop, b1, x_data, damping = 0.25, index = None, output = False):
       
    #     # try:
    #     #     plin0, plin2 = plin
    #     #     ploop0, ploop2 = ploop
    #     # except:
    #     #     plin0, plin2, plin4 = plin
    #     #     ploop0, ploop2, ploop4 = ploop
    #     Ploop0, Ploop2, Ploop4 = Ploop
    #     if self.pardict['do_hex'] == 1:
    #         P_model_mono, P_model_quad, P_model_hexa = P_model
    #     else:
    #         P_model_mono, P_model_quad = P_model
            
    #     # Pstl_mono, Pstl_quad, Pstl_hexa = Pstl
        
    #     # print(np.log10(P11l_mono[:, -20:]))
        
    #     # power_0, scale_0 = np.polyfit(np.log10(self.co.k[-20:]), np.log10(P11l_mono[0][-20:]), 1)
    #     # power_1, scale_1 = np.polyfit(np.log10(self.co.k[-20:]), np.log10(P11l_mono[1][-20:]), 1)
    #     # power_2, scale_2 = np.polyfit(np.log10(self.co.k[-20:]), np.log10(P11l_mono[2][-20:]), 1)
        
    #     # print(power_0, power_1, power_2)
    #     # print(scale_0, scale_1, scale_2)
        
    #     P_model_mono = P_model_mono[:, 0]
    #     P_model_quad = P_model_quad[:, 0]
        
    #     Ploop0 = Ploop0[:, :, 0]
    #     Ploop2 = Ploop2[:, :, 0]
    #     Ploop4 = Ploop4[:, :, 0]
    #     # kmode_out_new = np.repeat(self.kmode_out.reshape((len(self.kmode_out),1)), len(b1), axis=1)
        
    #     # P11l_mono_interp = []
    #     # for i in range(len(b1)):
    #     #     power_0, scale_0, r_value_0, p_value_0, std_err = linregress(np.log10(self.kin[-20:]), np.log10(P_model_mono[-20:, i]))
    #     #     print(power_0, scale_0, r_value_0, p_value_0)
    #     #     P11l_mono_interp.append(np.concatenate((interp1d(self.kin, P_model_mono[:, i], fill_value = 'extrapolate', kind = 'cubic', axis = 0)(self.kmode_in), 
    #     #                                 10.0**scale_0*self.kmode_out**power_0), axis = 0))
        
    #     # P11l_mono_interp = np.array(P11l_mono_interp).T
        
    #     # power_1, scale_1, r_value_1, p_value_1, std_err = linregress(np.log10(self.kin[-20:]), np.log10(P_model_quad[-20:]))
    #     # if self.pardict['do_hex'] == 1:
    #     #     power_2, scale_2, r_value_2, p_value_2, std_err = linregress(np.log10(self.kin[-20:]), np.log10(P_model_hexa[-20:]))
        
    #     # print(power_0, scale_0, r_value_0, p_value_0)
    #     # print(power_1, scale_1, r_value_1, p_value_1)
    #     # print(power_2, scale_2, r_value_2, p_value_2)
    #     # np.save('P11l_mono.npy', P11l_mono)
        
        
        
    #     power_0, scale_0, r_value_0, p_value_0, std_err = linregress(np.log10(self.kin[-20:]), np.log10(P_model_mono[-20:]))
        
    #     P11l_mono_interp = np.concatenate((interp1d(self.kin, P_model_mono, fill_value = 'extrapolate', kind = 'cubic', axis = 0)(self.kmode_in), 
    #                                 10.0**scale_0*self.kmode_out**power_0), axis = 0)
    #     # print(x_data)
        
    #     # P11l_mono_interp = interp1d(self.kin, P_model_mono, kind = 'linear', fill_value='extrapolate', axis = 0)(self.kmode)
        
    #     P11l_quad_interp = interp1d(self.kin, P_model_quad, kind = 'linear', fill_value='extrapolate', axis = 0)(self.kmode)
    #     if self.pardict['do_hex'] == 1:
    #         P11l_hexa_interp = interp1d(self.kin, P_model_hexa, kind = 'nearest', fill_value = 'extrapolate', axis = 0)(self.kmode)
            
    #         # P11l_interp = np.array([P11l_mono_interp, P11l_quad_interp, P11l_hexa_interp])
    #         P11l_mono_new = self.pk2xi_0.__call__(self.kmode, P11l_mono_interp, x_data[0], damping=damping)
    #         P11l_quad_new = self.pk2xi_2.__call__(self.kmode, P11l_quad_interp, x_data[1], damping=damping)
    #         P11l_hexa_new = self.pk2xi_4.__call__(self.kmode, P11l_hexa_interp, x_data[2], damping=damping)
    #         # P11l_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, P11l_mono_interp[i], self.co.dist, damping=damping) for i in range(len(b1))]])
    #         # P11l_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, P11l_quad_interp[i], self.co.dist, damping=damping) for i in range(len(b1))]])
    #         # P11l_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, P11l_hexa_interp[i], self.co.dist, damping=damping) for i in range(len(b1))]])
    #     else:
    #         # P11l_interp = np.array([P11l_mono_interp, P11l_quad_interp])
    #         P11l_mono_new = self.pk2xi_0.__call__(self.kmode, P11l_mono_interp, x_data[0], damping=damping)
    #         P11l_quad_new = self.pk2xi_2.__call__(self.kmode, P11l_quad_interp, x_data[1], damping=damping)
    #         # P11l_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, P11l_mono_interp[i], self.co.dist, damping=damping) for i in range(len(b1))]])
    #         # P11l_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, P11l_quad_interp[i], self.co.dist, damping=damping) for i in range(len(b1))]])
        
    #     Ploopl_mono_interp = interp1d(self.kin, Ploop0, kind = 'linear', fill_value = 'extrapolate', axis = 0)(self.kmode)
    #     Ploopl_quad_interp = interp1d(self.kin, Ploop2, kind = 'linear', fill_value = 'extrapolate', axis = 0)(self.kmode)
        
    #     if self.pardict['do_hex'] == 1:
            
    #         Ploopl_hexa_interp = interp1d(self.kin, Ploop4, kind = 'nearest', fill_value = 'extrapolate', axis = 0)(self.kmode)
    #         # Ploopl_interp = np.array([Ploopl_mono_interp, Ploopl_quad_interp, Ploopl_hexa_interp])
        
    #     # start = time.time()
    #     Ploopl_mono_new = []
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 0], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 1], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 2], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 3], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 4], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 5], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 6], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 7], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 8], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 9], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 10], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 11], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 12], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 13], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 14], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 15], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 16], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 17], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 18], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 19], x_data[0], damping=damping))
    #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, 20], x_data[0], damping=damping))
        
    #     Ploopl_quad_new = []
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 0], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 1], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 2], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 3], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 4], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 5], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 6], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 7], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 8], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 9], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 10], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 11], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 12], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 13], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 14], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 15], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 16], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 17], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 18], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 19], x_data[1], damping=damping))
    #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, 20], x_data[1], damping=damping))
        
    #     # Ploopl_mono_new = []
    #     # Ploopl_quad_new = []
    #     # for i in range(21):
    #     #     Ploopl_mono_new.append(self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, i], x_data[0], damping=damping))
    #     #     Ploopl_quad_new.append(self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp[:, i], x_data[1], damping=damping))
        
    #     # end = time.time()
    #     # print(end - start)
        
    #     Ploopl_mono_new = np.array(Ploopl_mono_new).T
    #     Ploopl_quad_new = np.array(Ploopl_quad_new).T
    #     if self.pardict['do_hex'] == 1:
        
    #         Ploopl_hexa_new = []
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 0], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 1], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 2], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 3], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 4], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 5], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 6], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 7], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 8], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 9], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 10], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 11], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 12], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 13], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 14], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 15], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 16], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 17], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 18], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 19], x_data[2], damping=damping))
    #         Ploopl_hexa_new.append(self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp[:, 20], x_data[2], damping=damping))
            
    #         Ploopl_hexa_new = np.array(Ploopl_hexa_new).T
    #         # Ploopl_quad_new = [self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp, x_data, damping=damping)]
    #         # Ploopl_hexa_new = [self.pk2xi_4.__call__(self.kmode, Ploopl_hexa_interp, x_data, damping=damping)]
    #         # Ploopl_mono_new = np.array([[self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp[:, i] self.co.dist, damping=damping) for i in range(len(b1))]])
    #         # Ploopl_quad_new = np.array([[self.pk2xi_2.__call__(self.kmode, Ploopl_interp[1, i], self.co.dist, damping=damping) for i in range(len(b1))]])
    #         # Ploopl_hexa_new = np.array([[self.pk2xi_4.__call__(self.kmode, Ploopl_interp[2, i], self.co.dist, damping=damping) for i in range(len(b1))]])     
    #         return np.concatenate(([P11l_mono_new], [P11l_quad_new], [P11l_hexa_new]), axis = 0), np.concatenate(([Ploopl_mono_new], [Ploopl_quad_new], [Ploopl_hexa_new]), axis = 0)
    #     else:
    #         # Ploopl_interp = np.array([Ploopl_mono_interp, Ploopl_quad_interp])
        
    #         # Ploopl_mono_new = [self.pk2xi_0.__call__(self.kmode, Ploopl_mono_interp, x_data, damping=damping)]
    #         # Ploopl_quad_new = [self.pk2xi_2.__call__(self.kmode, Ploopl_quad_interp, x_data, damping=damping)]
            
    #         return np.concatenate(([P11l_mono_new], [P11l_quad_new]), axis = 0), np.concatenate(([Ploopl_mono_new], [Ploopl_quad_new]), axis = 0)
            
        
        


# Holds all the data in a convenient dictionary
class FittingData:
    def __init__(self, pardict):

        x_data, ndata, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask = self.read_data(pardict)
        winnames = np.loadtxt(pardict["winfile"], dtype=str) if "winfile" in pardict else None

        self.data = {
            "x_data": x_data,
            "fit_data": fit_data,
            "ndata": ndata,
            "cov": cov,
            "cov_inv": cov_inv,
            "chi2data": chi2data,
            "invcovdata": invcovdata,
            "fitmask": fitmask,
            "shot_noise": pardict["shot_noise"],
            "windows": winnames,
        }

        # Check covariance matrix is symmetric and positive-definite by trying to do a cholesky decomposition
        diff = np.abs((self.data["cov"] - self.data["cov"].T) / self.data["cov"])
        if not (np.logical_or(diff <= 1.0e-6, np.isnan(diff))).all():
            print(diff)
            print("Error: Covariance matrix not symmetric!")
            exit(0)
        try:
            cholesky(self.data["cov"])
        except:
            print("Error: Covariance matrix not positive-definite!")
            exit(0)
            
        # self.data = self.apply_hartlap_percival(pardict, self.data)
            
    def percival_factor(self, redindex, pardict, data, onebin, n_sims):
        if int(pardict['do_marg']) == 1:
            nparams = len(pardict['freepar']) + 3 
        else:
            nparams = len(pardict['freepar']) + 10
            
        # print(nparams)
            
        if onebin == False:
            percival_A = 2.0/((n_sims[redindex] - data["ndata"][0]-1.0)*(n_sims[redindex] - data['ndata'][0]-4.0))
            percival_B = percival_A/2.0*(n_sims[redindex] - data['ndata'][0]-2.0)
            percival_m = (1.0+percival_B*(data['ndata'][0] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
        else:
            percival_A = 2.0/((n_sims - data["ndata"][0]-1.0)*(n_sims - data['ndata'][0]-4.0))
            percival_B = percival_A/2.0*(n_sims - data['ndata'][0]-2.0)
            percival_m = (1.0+percival_B*(data['ndata'][0] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
        return percival_m
            
    def apply_hartlap_percival(self, pardict, data):
        n_sims = np.float64(pardict['n_sims'])
        
        nz = len(pardict['z_pk'])
        if nz > 1.5:
            onebin = False
        else:
            onebin = True
        
        if onebin == False:
            hartlap = [(ns - data["ndata"][i] - 2.0) / (ns - 1.0) for i, ns in enumerate(n_sims)]
            
            length_all = []
            for i in range(nz):
                length = len(data["x_data"][i][0]) + len(data["x_data"][i][1])
                if pardict['do_hex'] == True:
                    length += len(data["x_data"][i][2])
                length_all.append(length)
                
            length_start = 0
            length_end = 0
            cov_full = []
            for i in range(nz):
                if i == 0:
                    length_start += 0    
                else:
                    length_start += length_all[i-1]
                length_end += length_all[i]
                
                hartlap = (n_sims[i] - data["ndata"][0] - 2.0) / (n_sims[i] - 1.0)
                
                percival_m = self.percival_factor(i, pardict, data, onebin, n_sims)
                
                cov_part = data['cov'][length_start:length_end, length_start:length_end]*percival_m/hartlap
                
                cov_full.append(cov_part)
                
            cov_new = block_diag(*cov_full)
            
            cov_lu, pivots, cov_new_inv, info = lapack.dgesv(cov_new, np.eye(len(cov_new)))
            
            fitdata = data['fit_data']
            
            chi2data = np.dot(fitdata, np.dot(cov_new_inv, fitdata))
            
            invcovdata = np.dot(fitdata, cov_new_inv)
            
            data['cov'] = cov_new
            data['cov_inv'] = cov_new_inv
            data['chi2data'] = chi2data
            data['invcovdata'] = invcovdata
            
            # cov_new = copy.copy(fittingdata.data['cov'])
            # for i, (nd, ndcum) in enumerate(
            #     zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
            # ):
            #     cov_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= percival[i]
            
            # cov_inv_new = copy.copy(fittingdata.data["cov_inv"])
            # for i, (nd, ndcum) in enumerate(
            #     zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
            # ):
            #     cov_inv_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= hartlap[i]
            
            # nz = len(pardict['z_pk'])
            # keyword = '_all_mean'
        else:
            length_all = []
            for i in range(nz):
                length = len(data["x_data"][i][0]) + len(data["x_data"][i][1])
                if pardict['do_hex'] == True:
                    length += len(data["x_data"][i][2])
                length_all.append(length)
            
            
            length_start = 0
            length_end = 0
            for i in range(1):
                if i == 0:
                    length_start += 0    
                else:
                    length_start += length_all[i-1]
                length_end += length_all[i]   
                
            print(length_start, length_end)
            
            hartlap = (n_sims - data["ndata"][0] - 2.0) / (n_sims - 1.0)
            print(hartlap)
            
            percival_m = self.percival_factor(np.int32(pardict['red_index']), pardict, data, onebin, n_sims)
            print(percival_m)
            
            cov_part = data['cov'][length_start:length_end, length_start:length_end]*percival_m
            fitdata_part = data['fit_data'][length_start:length_end]
            
            cov_lu, pivots, cov_part_inv, info = lapack.dgesv(cov_part, np.eye(len(cov_part)))
            
            cov_part_inv = cov_part_inv*hartlap
            
            chi2data_part = np.dot(fitdata_part, np.dot(cov_part_inv, fitdata_part))
            invcovdata_part = np.dot(fitdata_part, cov_part_inv)
            
            data['cov'] = cov_part
            data['cov_inv'] = cov_part_inv
            data['chi2data'] = chi2data_part
            data['invcovdata'] = invcovdata_part
            data['fit_data'] = fitdata_part
            
            # nz = 1 
            
            # if single_mock == False:
            #     keyword = '_bin_'+str(redindex) + '_mean'
            # else:
            #     keyword = '_bin_'+str(redindex) + '_mock_' + str(mock_num)
            
        return data

    def read_pk(self, inputfile, step_size, skiprows):

        dataframe = pd.read_csv(
            inputfile,
            comment="#",
            skiprows=skiprows,
            delim_whitespace=True,
            names=["k", "pk0", "pk2", "pk4", "nk"],
            header=None,
        )
        k = dataframe["k"].values
        # print(k)
        if step_size == 1:
            k_rebinned = k
            pk0_rebinned = dataframe["pk0"].values
            pk2_rebinned = dataframe["pk2"].values
            pk4_rebinned = dataframe["pk4"].values
            nk_rebinned = dataframe["nk"].values
        else:
            add = k.size % step_size
            weight = dataframe["nk"].values
            if add:
                to_add = step_size - add
                k = np.concatenate((k, [k[-1]] * to_add))
                dataframe["pk0"].values = np.concatenate(
                    (dataframe["pk0"].values, [dataframe["pk0"].values[-1]] * to_add)
                )
                dataframe["pk2"].values = np.concatenate(
                    (dataframe["pk2"].values, [dataframe["pk2"].values[-1]] * to_add)
                )
                dataframe["pk4"].values = np.concatenate(
                    (dataframe["pk4"].values, [dataframe["pk4"].values[-1]] * to_add)
                )
                weight = np.concatenate((weight, [0] * to_add))
            k = k.reshape((-1, step_size))
            pk0 = (dataframe["pk0"].values).reshape((-1, step_size))
            pk2 = (dataframe["pk2"].values).reshape((-1, step_size))
            pk4 = (dataframe["pk4"].values).reshape((-1, step_size))
            weight = weight.reshape((-1, step_size))
            # Take the average of every group of step_size rows to rebin
            k_rebinned = np.average(k, axis=1)
            pk0_rebinned = np.average(pk0, axis=1, weights=weight)
            pk2_rebinned = np.average(pk2, axis=1, weights=weight)
            pk4_rebinned = np.average(pk4, axis=1, weights=weight)

        return np.vstack([k_rebinned, pk0_rebinned, pk2_rebinned, pk4_rebinned]).T

    def read_data(self, pardict):

        # Updated. Now reads files for every redshift bin and stores them consecutively. Also
        # deals with NGC+SGC data, concatenating it for every redshift bin.

        # Read in the data
        datafiles = np.loadtxt(pardict["datafile"], ndmin=1, dtype=str)
        nz = len(pardict["z_pk"])

        print(datafiles)
        all_xdata = []
        all_ndata = []
        all_fitmask = []
        all_fit_data = []
        for i in range(nz):
            x_data, ndata, fitmask, fit_data = self.get_some_data(pardict, datafiles[i])
            all_xdata.append(x_data)
            all_ndata.append(ndata)
            all_fitmask.append(fitmask)
            all_fit_data.append(fit_data)
        fitmask = np.concatenate(all_fitmask)
        
        fit_data = np.concatenate(all_fit_data)

        """# Read in, reshape and mask the covariance matrix
        cov_flat = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
        nin = len(x_data)
        cov_input = cov_flat[:, 2].reshape((3 * nin, 3 * nin))
        nx0, nx2 = len(x_data[0]), len(x_data[1])
        nx4 = len(x_data[2]) if pardict["do_hex"] else 0
        mask0, mask2, mask4 = fitmask[0][:, None], fitmask[1][:, None], fitmask[2][:, None]
        cov = np.zeros((nx0 + nx2 + nx4, nx0 + nx2 + nx4))
        cov[:nx0, :nx0] = cov_input[mask0, mask0.T]
        cov[:nx0, nx0 : nx0 + nx2] = cov_input[mask0, nin + mask2.T]
        cov[nx0 : nx0 + nx2, :nx0] = cov_input[nin + mask2, mask0.T]
        cov[nx0 : nx0 + nx2, nx0 : nx0 + nx2] = cov_input[nin + mask2, nin + mask2.T]
        if pardict["do_hex"]:
            cov[:nx0, nx0 + nx2 :] = cov_input[mask0, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, :nx0] = cov_input[2 * nin + mask4, mask0.T]
            cov[nx0 : nx0 + nx2, nx0 + nx2 :] = cov_input[nin + mask2, 2 * nin + mask4.T]
            cov[nx0 + nx2 :, nx0 : nx0 + nx2] = cov_input[2 * nin + mask4, nin + mask2.T]
            cov[nx0 + nx2 :, nx0 + nx2 :] = cov_input[2 * nin + mask4, 2 * nin + mask4.T]"""

        # Read in, reshape and mask the covariance matrix
        cov_input = np.array(pd.read_csv(pardict["covfile"], delim_whitespace=True, header=None))
        
        # if get_binmat == True:
            
        #     if int(pardict['do_hex']) == 1:
        #         Nl = 3
        #     else:
        #         Nl = 2
                
            
        #     binmat, length_all = self.getxbin_mat(pardict, all_xdata)
        #     if Nl == 3: 
        #         cov = np.matmul(binmat, np.matmul(cov_input, binmat.T))
        #     else:
                
        #         if nz == 1:
        #             mask_cov = np.concatenate([np.full(length_all[0], True), np.full(length_all[0], True), np.full(length_all[0], False)])
                    
        #         else:
        #             mask_cov = []
        #             for i in range(nz):
        #                 mask_cov.append(np.concatenate([np.full(length_all[i], True), np.full(length_all[i], True), np.full(length_all[i], False)]))
                        
        #             mask_cov = np.concatenate(mask_cov)
                    
        #         cov_all = np.delete(np.delete(cov_input, ~mask_cov, axis=0), ~mask_cov, axis=1)
                
        #         cov = np.matmul(binmat, np.matmul(cov_all, binmat.T))
            
        #     cov_inv = np.linalg.inv(cov)
            
        # else:
        #     cov = np.delete(np.delete(cov_input, ~fitmask, axis=0), ~fitmask, axis=1)
        #     # Invert the covariance matrix
        #     cov_lu, pivots, cov_inv, info = lapack.dgesv(cov, np.eye(len(cov)))
        
        # print(np.shape(cov_input), len(fitmask))
        
        cov = np.delete(np.delete(cov_input, ~fitmask, axis=0), ~fitmask, axis=1)
        
        # print(np.shape(cov))
       #     # Invert the covariance matrix
        try:
            cov_lu, pivots, cov_inv, info = lapack.dgesv(cov, np.eye(len(cov)))
        except:
            cov_inv = np.linalg.inv(cov)

        chi2data = np.dot(fit_data, np.dot(cov_inv, fit_data))
        invcovdata = np.dot(fit_data, cov_inv)

        return all_xdata, all_ndata, fit_data, cov, cov_inv, chi2data, invcovdata, fitmask
    
    # def getxbin_mat(self, pardict, all_xdata):
        
    #     nz = len(pardict['z_pk'])
    #     datafiles = np.loadtxt(pardict["datafile"], ndmin=1, dtype=str)
        
    #     if int(pardict['do_hex']) == 1:
    #         Nl = 3
    #     else:
    #         Nl = 2
        
    #     window_cov = []
    #     length_all = []
    #     for k in range(nz):
    #         data = self.read_pk(datafiles[k], 1, 0)
    #         x_data = data[:, 0]
    #         length_all.append(len(x_data))
            
    #         window_cov_z = []
    #         for j in range(Nl):
    #             ks = all_xdata[k][j]
    #             dk = ks[-1] - ks[-2]
    #             ks_input = x_data
                
    #             binmat = np.zeros((len(ks), len(ks_input)))
    #             for ii in range(len(ks_input)):

    #                 # Define basis vector
    #                 pkvec = np.zeros_like(ks_input)
    #                 pkvec[ii] = 1
    #                 # print(pkvec)

    #                 # Define the spline:
    #                 pkvec_spline = splrep(ks_input, pkvec)

    #                 # Now compute binned basis vector:
    #                 tmp = np.zeros_like(ks)
    #                 for i, kk in enumerate(ks):
    #                     kl = kk - dk / 2
    #                     kr = kk + dk / 2
    #                     kin = np.linspace(kl, kr, 100)
    #                     tmp[i] = np.trapz(kin**2 * splev(kin, pkvec_spline, ext=2), x=kin) * 3 / (kr**3 - kl**3)
                        
    #                 binmat[:, ii] = tmp
                    
    #             window_cov_z.append(binmat)
            
    #         window_cov.append(block_diag(*window_cov_z))
            
    #     return block_diag(*window_cov), length_all

    def get_some_data(self, pardict, datafile):

        if pardict["do_corr"] or pardict["corr_convert"]:
            data = np.array(pd.read_csv(datafile, delim_whitespace=True, header=None, comment="#"))
        else:
            data = self.read_pk(datafile, 1, 0)
            
        # print(np.shape(data))

        x_data = data[:, 0]
        
        # print(x_data, pardict['xfit_min'][0], pardict['xfit_max'][0])
        """fitmask = [
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][0], x_data <= pardict["xfit_max"][0]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][1], x_data <= pardict["xfit_max"][1]))[0]).astype(
                int
            ),
            (np.where(np.logical_and(x_data >= pardict["xfit_min"][2], x_data <= pardict["xfit_max"][2]))[0]).astype(
                int
            ),
        ]"""
        ell = 3 if pardict["do_hex"] else 2
        if pardict['do_corr'] or pardict["corr_convert"]:
            fitmask = np.array(
                [np.logical_and(x_data >= np.float64(pardict["sfit_min"][i]), x_data <= np.float64(pardict["sfit_max"][i])) for i in range(ell)]
            )
        else:
            fitmask = np.array(
                [np.logical_and(x_data >= pardict["xfit_min"][i], x_data <= pardict["xfit_max"][i]) for i in range(ell)]
            )
        if not pardict["do_hex"]:
            fitmask = np.concatenate([fitmask, [np.full(len(x_data), False)]])
        x_data = [data[fitmask[i], 0] for i in range(ell)]
        fit_data = np.concatenate([data[fitmask[i], i + 1] for i in range(ell)])

        return x_data, np.sum([len(x_data[i]) for i in range(ell)]), np.concatenate(fitmask), fit_data


def create_plot(pardict, fittingdata, plotindex=0):

    if pardict["do_hex"]:
        x_data = fittingdata.data["x_data"][plotindex]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = fittingdata.data["x_data"][plotindex][:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    fit_data = fittingdata.data["fit_data"]
    cov = fittingdata.data["cov"]

    ndata = nx0 + nx2 + nx4
    plt_data = (
        np.concatenate(x_data) ** 2 * fit_data[plotindex * ndata : (plotindex + 1) * ndata]
        if pardict["do_corr"]
        else np.concatenate(x_data) ** 1.0 * fit_data[plotindex * ndata : (plotindex + 1) * ndata]
    )
    if pardict["do_corr"]:
        plt_err = np.concatenate(x_data) ** 2 * np.sqrt(np.diag(cov)[plotindex * ndata : (plotindex + 1) * ndata])
    else:
        plt_err = np.concatenate(x_data) ** 1.0 * np.sqrt(np.diag(cov)[plotindex * ndata : (plotindex + 1) * ndata])

    plt.errorbar(
        x_data[0],
        plt_data[:nx0],
        yerr=plt_err[:nx0],
        marker="o", 
        markerfacecolor="r",
        markeredgecolor="k",
        color="r",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    plt.errorbar(
        x_data[1],
        plt_data[nx0 : nx0 + nx2],
        yerr=plt_err[nx0 : nx0 + nx2],
        marker="o",
        markerfacecolor="b",
        markeredgecolor="k",
        color="b",
        linestyle="None",
        markeredgewidth=1.3,
        zorder=5,
    )
    if pardict["do_hex"]:
        plt.errorbar(
            x_data[2],
            plt_data[nx0 + nx2 :],
            yerr=plt_err[nx0 + nx2 :],
            marker="o",
            markerfacecolor="g",
            markeredgecolor="k",
            color="g",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )

    plt.xlim(np.amin(pardict["xfit_min"]) * 0.95, np.amax(pardict["xfit_max"]) * 1.05)
    if pardict["do_corr"]:
        plt.xlabel(r"$s\,(h^{-1}\,\mathrm{Mpc})$", fontsize=16)
        plt.ylabel(r"$s^{2}\xi(s)$", fontsize=16, labelpad=5)
    else:
        plt.xlabel(r"$k\,(h\,\mathrm{Mpc}^{-1})$", fontsize=16)
        plt.ylabel(r"$kP(k)\,(h^{-2}\,\mathrm{Mpc}^{2})$", fontsize=16, labelpad=5)
    plt.tick_params(width=1.3)
    plt.tick_params("both", length=10, which="major")
    plt.tick_params("both", length=5, which="minor")
    for axis in ["top", "left", "bottom", "right"]:
        plt.gca().spines[axis].set_linewidth(1.3)
    for tick in plt.gca().xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in plt.gca().yaxis.get_ticklabels():
        tick.set_fontsize(14)
    plt.tight_layout()
    plt.gca().set_autoscale_on(False)
    plt.ion()

    return plt


def update_plot(pardict, x_data, P_model, plt, keep=False, plot_index=0):

    if pardict["do_hex"]:
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = x_data[:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    plt_data = np.concatenate(x_data) ** 2 * P_model if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * P_model

    plt10 = plt.errorbar(
        x_data[0],
        plt_data[:nx0],
        marker="None",
        color="r",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt11 = plt.errorbar(
        x_data[1],
        plt_data[nx0 : nx0 + nx2],
        marker="None",
        color="b",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    if pardict["do_hex"]:
        plt12 = plt.errorbar(
            x_data[2],
            plt_data[nx0 + nx2 :],
            marker="None",
            color="g",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )

    if keep:
        plt.ioff()
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt10 is not None:
            plt10.remove()
        if plt11 is not None:
            plt11.remove()
        if pardict["do_hex"]:
            if plt12 is not None:
                plt12.remove()


def update_plot_lin_loop(pardict, x_data, P_model, P_model_lin, P_model_loop, plt, keep=False, plot_index=0):

    if pardict["do_hex"]:
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), len(x_data[2])
    else:
        x_data = x_data[:2]
        nx0, nx2, nx4 = len(x_data[0]), len(x_data[1]), 0
    plt_data = np.concatenate(x_data) ** 2 * P_model if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * P_model
    plt_data_lin = (
        np.concatenate(x_data) ** 2 * P_model_lin if pardict["do_corr"] else np.concatenate(x_data) ** 1.0 * P_model_lin
    )
    plt_data_loop = (
        np.concatenate(x_data) ** 2 * P_model_loop
        if pardict["do_corr"]
        else np.concatenate(x_data) ** 1.0 * P_model_loop
    )

    plt10 = plt.errorbar(
        x_data[0],
        plt_data[:nx0],
        marker="None",
        color="r",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt20 = plt.errorbar(
        x_data[0],
        plt_data_lin[:nx0],
        marker="None",
        color="r",
        linestyle=":",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt30 = plt.errorbar(
        x_data[0],
        plt_data_lin[:nx0] + plt_data_loop[:nx0],
        marker="None",
        color="r",
        linestyle="--",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt11 = plt.errorbar(
        x_data[1],
        plt_data[nx0 : nx0 + nx2],
        marker="None",
        color="b",
        linestyle="-",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt21 = plt.errorbar(
        x_data[1],
        plt_data_lin[nx0 : nx0 + nx2],
        marker="None",
        color="b",
        linestyle=":",
        markeredgewidth=1.3,
        zorder=0,
    )
    plt31 = plt.errorbar(
        x_data[1],
        plt_data_lin[nx0 : nx0 + nx2] + plt_data_loop[nx0 : nx0 + nx2],
        marker="None",
        color="b",
        linestyle="--",
        markeredgewidth=1.3,
        zorder=0,
    )
    if pardict["do_hex"]:
        plt12 = plt.errorbar(
            x_data[2],
            plt_data[nx0 + nx2 :],
            marker="None",
            color="g",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt22 = plt.errorbar(
            x_data[2],
            plt_data_lin[nx0 + nx2 :],
            marker="None",
            color="g",
            linestyle=":",
            markeredgewidth=1.3,
            zorder=0,
        )
        plt32 = plt.errorbar(
            x_data[2],
            plt_data_lin[nx0 + nx2 :] + plt_data_loop[nx0 + nx2 :],
            marker="None",
            color="g",
            linestyle="--",
            markeredgewidth=1.3,
            zorder=0,
        )

    if keep:
        plt.ioff()
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt10 is not None:
            plt10.remove()
            plt20.remove()
            plt30.remove()
        if plt11 is not None:
            plt11.remove()
            plt21.remove()
            plt31.remove()
        if pardict["do_hex"]:
            if plt12 is not None:
                plt12.remove()
                plt22.remove()
                plt32.remove()


def update_plot_components(pardict, kin, P_components, plt, keep=False, comp_list=(True, True, True, True)):

    ls = [":", "-.", "--", "-"]
    labels = ["Linear", "Linear+Loop", "Linear+Loop+Counter", "Linear+Loop+Counter+Stoch"]
    kinfac = kin ** 2 if pardict["do_corr"] else kin ** 1.0

    part_comp = [np.zeros(len(kin)), np.zeros(len(kin)), np.zeros(len(kin))]
    for (line, comp, add, label) in zip(ls, P_components, comp_list, labels):
        for i, c in enumerate(comp):
            part_comp[i] += c
        if add:
            plt10 = plt.errorbar(
                kin,
                kinfac * part_comp[0],
                marker="None",
                color="r",
                linestyle=line,
                markeredgewidth=1.3,
                zorder=0,
                label=label,
            )
            plt11 = plt.errorbar(
                kin,
                kinfac * part_comp[1],
                marker="None",
                color="b",
                linestyle=line,
                markeredgewidth=1.3,
                zorder=0,
            )
            if pardict["do_hex"]:
                plt12 = plt.errorbar(
                    kin,
                    kinfac * part_comp[2],
                    marker="None",
                    color="g",
                    linestyle=line,
                    markeredgewidth=1.3,
                    zorder=0,
                )
    plt.legend()

    if keep:
        plt.ioff()
        plt.show()
    if not keep:
        plt.pause(0.005)
        if plt10 is not None:
            plt10.remove()
        if plt11 is not None:
            plt11.remove()
        if pardict["do_hex"]:
            if plt12 is not None:
                plt12.remove()


def format_pardict(pardict):

    pardict["do_corr"] = int(pardict["do_corr"])
    pardict["do_marg"] = int(pardict["do_marg"])
    pardict["do_hex"] = int(pardict["do_hex"])
    pardict["taylor_order"] = int(pardict["taylor_order"])
    pardict["xfit_min"] = np.array(pardict["xfit_min"]).astype(float)
    pardict["xfit_max"] = np.array(pardict["xfit_max"]).astype(float)
    pardict["order"] = int(pardict["order"])
    # pardict["scale_independent"] = True if pardict["scale_independent"].lower() is "true" else False
    pardict["scale_independent"] = True if pardict["scale_independent"].lower() == "true" else False
    pardict["z_pk"] = np.array(pardict["z_pk"], dtype=float)
    if not any(np.shape(pardict["z_pk"])):
        pardict["z_pk"] = [float(pardict["z_pk"])]
        
    pardict['corr_convert'] = int(pardict['corr_convert'])

    return pardict


def do_optimization(func, start):

    from scipy.optimize import basinhopping, minimize
    
    result = basinhopping(
        func,
        start,
        niter_success=10,
        niter=100,
        stepsize=0.005,
        minimizer_kwargs={
            "method": "Nelder-Mead",
            # "method": "Powell",
            "tol": 1.0e-4,
            "options": {"maxiter": 40000, "xatol": 1.0e-4, "fatol": 1.0e-4},
        },
    )
    
    # result = minimize(func, start, method='Powell', tol=1e-6)
    # from scipy.optimize import differential_evolution, shgo

    # # result = differential_evolution(func, bounds=((2.5, 3.5), (0.50, 0.75), (0.07, 0.16), (0.0, 0.04), (0.0, 4.0), (-1000.0, 1000.0), 
    # #                                               (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0),
    # #                                               (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)), tol=1.0e-6)
    
    # result = shgo(func, bounds=((2.5, 3.5), (0.50, 0.75), (0.07, 0.16), (0.0, 0.04), (0.0, 4.0), (-1000.0, 1000.0), 
    #                                               (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0),
    #                                               (-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)))
    
    print("#-------------- Best-fit----------------")
    print(result)

    

    return result


def read_chain(chainfile, burnlimitlow=5000, burnlimitup=None):

    # Read in the samples
    walkers = []
    samples = []
    like = []
    infile = open(chainfile, "r")
    for line in infile:
        ln = line.split()
        samples.append(list(map(float, ln[1:-1])))
        walkers.append(int(ln[0]))
        like.append(float(ln[-1]))
    infile.close()

    like = np.array(like)
    walkers = np.array(walkers)
    samples = np.array(samples)
    nwalkers = max(walkers)

    if burnlimitup is None:
        bestid = np.argmax(like)
    else:
        bestid = np.argmax(like[: np.amax(walkers) * burnlimitup])

    burntin = []
    burntlike = []
    nburntin = 0

    for i in range(nwalkers + 1):
        ind = np.where(walkers == i)[0]
        if len(ind) == 0:
            continue
        x = [j for j in range(len(ind))]
        if burnlimitup is None:
            ind2 = np.where(np.asarray(x) >= burnlimitlow)[0]
        else:
            ind2 = np.where(np.logical_and(np.asarray(x) >= burnlimitlow, np.asarray(x) <= burnlimitup))[0]
        for k in range(len(ind2 + 1)):
            burntin.append(samples[ind[ind2[k]]])
            burntlike.append(like[ind[ind2[k]]])
        nburntin += len(ind2)
    burntin = np.array(burntin)
    burntlike = np.array(burntlike)

    return burntin, samples[bestid], burntlike


def read_chain_backend(chainfile):

    import copy
    import emcee

    reader = emcee.backends.HDFBackend(chainfile)

    tau = reader.get_autocorr_time()
    thin = int(0.5 * np.min(tau))
    # tau = 400.0
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples


def get_Planck(filename, nfiles, usecols=(6, 29, 3, 2), raw=False):

    weights = []
    chain = []
    for i in range(1, nfiles):
        strin = str("%s_%d.txt" % (filename, i))
        data = np.array(pd.read_csv(strin, delim_whitespace=True, header=None))
        weights.append(data[:, 0])
        chain.append(data[:, list(usecols)])
    weights = np.concatenate(weights)
    chain = np.concatenate(chain)

    # Need to convert H0 to h if requested
    index = np.where(np.array(usecols) == 29)[0]
    if len(index) > 0:
        chain[:, index] /= 100.0

    if raw:
        return weights, chain
    else:
        cov = np.cov(chain, rowvar=False, aweights=weights)
        return np.average(chain, axis=0, weights=weights), cov, np.linalg.inv(cov)


class PowerToCorrelation(ABC):
    """Generic class for converting power spectra to correlation functions
    Using a class based method as there might be multiple implementations and
    some of the implementations have state.
    """

    def __init__(self, ell=0):
        self.ell = ell

    def __call__(self, ks, pk, ss):
        """Generates the correlation function
        Parameters
        ----------
        ks : np.ndarray
            The k values for the power spectrum data. *Assumed to be in log space*
        pk : np.ndarray
            The P(k) values
        ss : np.nparray
            The distances to calculate xi(s) at.
        Returns
        -------
        xi : np.ndarray
            The correlation function at the specified distances
        """
        raise NotImplementedError()
    
class PowerToCorrelationSphericalBessel(PowerToCorrelation):
    def __init__(self, qs=None, ell=15, low_ring=True, fourier=True):

        """
        From Stephen Chen. Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
        the untransformed coordinate, up to a given order L in bessel functions (j_l for l
        less than or equal to L. The point is to save time by evaluating the Mellin transforms
        u_m in advance.
        Does not use fftw as in spherical_bessel_transform_fftw.py, which makes it convenient
        to evaluate the generalized correlation functions in qfuncfft, as there aren't as many
        ffts as in LPT modules so time saved by fftw is minimal when accounting for the
        startup time of pyFFTW.
        Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
        with the above modifications.
        Taken from velocileptors.
        """

        if qs is None:
            qs = np.logspace(-4, np.log(5.0), 2000)

        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2 * np.pi**2)

        self.q = qs
        self.ell = ell

        self.Nx = len(qs)
        self.Delta = np.log(qs[-1] / qs[0]) / (self.Nx - 1)

        self.N = 2 ** (int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        self.pads = np.zeros((self.N - self.Nx) // 2)
        self.pad_iis = np.arange(self.Npad - self.Npad // 2, self.N - self.Npad // 2)
        
        # print(np.shape(self.pads))
        # print(np.shape(self.pad_iis))

        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = np.arange(0, self.N // 2 + 1)
        self.ydict = {}
        self.udict = {}
        self.qdict = {}

        if low_ring:
            for ll in range(self.ell + 1):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta / np.pi * np.angle(self.UK(ll, q + 1j * np.pi / self.Delta))  # ln(xmin*ymax)
                ys = np.exp(lnxy - self.Delta) * qs / (qs[0] * qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

                self.ydict[ll] = ys
                self.udict[ll] = us
                self.qdict[ll] = q

        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(self.ell + 1):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0] * qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)

                self.ydict[ll] = ys
                self.udict[ll] = us
                self.qdict[ll] = q

    def __call__(self, ks, fq, ss, damping=0.25, nu=None):
        """
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        """
        if nu is None:
            nu = self.ell

        fq = fq * np.exp(-(ks**2) * damping**2)

        q = self.qdict[nu]
        y = self.ydict[nu]
        f = np.concatenate((self.pads, self.q ** (3 - q) * fq, self.pads))

        fks = np.fft.rfft(f)
        gks = self.udict[nu] * fks
        gs = np.fft.hfft(gks) / self.N

        return np.real((1j) ** nu * splev(ss, splrep(y, y ** (-q) * gs[self.pad_iis])))

    def UK(self, nu, z):
        """
        The Mellin transform of the spherical bessel transform.
        """
        return self.sqrtpi * np.exp(np.log(2) * (z - 2) + loggamma(0.5 * (nu + z)) - loggamma(0.5 * (3 + nu - z)))

    def update_tilt(self, nu, tilt):
        """
        Update the tilt for a particular nu. Assume low ring coordinates.
        """
        q = tilt
        ll = nu

        ms = np.arange(0, self.N // 2 + 1)
        lnxy = self.Delta / np.pi * np.angle(self.UK(ll, q + 1j * np.pi / self.Delta))  # ln(xmin*ymax)
        ys = np.exp(lnxy - self.Delta) * self.q / (self.q[0] * self.q[-1])
        us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

        self.ydict[ll] = ys
        self.udict[ll] = us
        self.qdict[ll] = q

    def loginterp(
        x,
        y,
        yint=None,
        side="both",
        lorder=9,
        rorder=9,
        lp=1,
        rp=-2,
        ldx=1e-6,
        rdx=1e-6,
        interp_min=-12,
        interp_max=12,
        Nint=10**5,
        verbose=False,
        option="B",
    ):
        """
        Extrapolate function by evaluating a log-index of left & right side.
        From Chirag Modi's CLEFT code at
        https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
        The warning for divergent power laws on both ends is turned off. To turn back on uncomment lines 26-33.
        """

        if yint is None:
            yint = InterpolatedUnivariateSpline(x, y, k=5)
        if side == "both":
            side = "lr"

        # Make sure there is no zero crossing between the edge points
        # If so assume there can't be another crossing nearby

        if np.sign(y[lp]) == np.sign(y[lp - 1]) and np.sign(y[lp]) == np.sign(y[lp + 1]):
            l = lp
        else:
            l = lp + 2

        if np.sign(y[rp]) == np.sign(y[rp - 1]) and np.sign(y[rp]) == np.sign(y[rp + 1]):
            r = rp
        else:
            r = rp - 2

        lneff = derivative(yint, x[l], dx=x[l] * ldx, order=lorder) * x[l] / y[l]
        rneff = derivative(yint, x[r], dx=x[r] * rdx, order=rorder) * x[r] / y[r]

        # print(lneff, rneff)

        # uncomment if you like warnings.
        # if verbose:
        #    if lneff < 0:
        #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        #        print('WARNING: Runaway index on left side, bad interpolation. Left index = %0.3e at %0.3e'%(lneff, x[l]))
        #    if rneff > 0:
        #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        #        print('WARNING: Runaway index on right side, bad interpolation. Reft index = %0.3e at %0.3e'%(rneff, x[r]))

        if option == "A":

            xl = np.logspace(interp_min, np.log10(x[l]), Nint)
            xr = np.logspace(np.log10(x[r]), interp_max, Nint)
            yl = y[l] * (xl / x[l]) ** lneff
            yr = y[r] * (xr / x[r]) ** rneff
            # print(xr/x[r])

            xint = x[l + 1 : r].copy()
            yint = y[l + 1 : r].copy()
            if side.find("l") > -1:
                xint = np.concatenate((xl, xint))
                yint = np.concatenate((yl, yint))
            if side.find("r") > -1:
                xint = np.concatenate((xint, xr))
                yint = np.concatenate((yint, yr))
            yint2 = InterpolatedUnivariateSpline(xint, yint, k=5, ext=3)

        else:
            # nan_to_numb is to prevent (xx/x[l/r])^lneff to go to nan on the other side
            # since this value should be zero on the wrong side anyway
            # yint2 = lambda xx: (xx <= x[l]) * y[l]*(xx/x[l])**lneff \
            #                 + (xx >= x[r]) * y[r]*(xx/x[r])**rneff \
            #                 + (xx > x[l]) * (xx < x[r]) * interpolate(x, y, k = 5, ext=3)(xx)
            yint2 = (
                lambda xx: (xx <= x[l]) * y[l] * np.nan_to_num((xx / x[l]) ** lneff)
                + (xx >= x[r]) * y[r] * np.nan_to_num((xx / x[r]) ** rneff)
                + (xx > x[l]) * (xx < x[r]) * InterpolatedUnivariateSpline(x, y, k=5, ext=3)(xx)
            )

        return yint2
    
