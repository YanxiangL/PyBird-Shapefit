# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:57:34 2022

@author: s4479813
"""

import numpy as np
import sys
from configobj import ConfigObj
from classy import Class
import scipy.constants as conts
from scipy import interpolate, signal
import scipy as sp
import copy
from nbodykit import cosmology
from findiff import FinDiff
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect

# sys.path.append("../")
# from tbird.Grid import run_camb, run_class
# from fitting_codes.fitting_utils import format_pardict, read_chain_backend, BirdModel, do_optimization, get_Planck, FittingData

def format_pardict(pardict):
    pardict["do_corr"] = int(pardict["do_corr"])
    pardict["do_marg"] = int(pardict["do_marg"])
    pardict["do_hex"] = int(pardict["do_hex"])
    pardict["taylor_order"] = int(pardict["taylor_order"])
    pardict["xfit_min"] = np.array(pardict["xfit_min"]).astype(float)
    pardict["xfit_max"] = np.array(pardict["xfit_max"]).astype(float)
    pardict["order"] = int(pardict["order"])
    pardict["scale_independent"] = True if pardict["scale_independent"].lower() is "true" else False
    pardict["z_pk"] = np.array(pardict["z_pk"], dtype=float)
    if not any(np.shape(pardict["z_pk"])):
        pardict["z_pk"] = [float(pardict["z_pk"])]
    return pardict

# def grid_params(pardict, params):
#     ln10As, h, omega_cdm, omega_b = params[:4]
    
#     A_s = np.exp(ln10As)/1.0e10
#     H_0 = 100.0*h
    
#     M = Class()
    
#     z_data = float(pardict["z_pk"][0])
    
#     M.set(
#         {
#             "A_s": A_s,
#             "n_s": float(pardict["n_s"]),
#             "H0": H_0,
#             "omega_b": omega_b,
#             "omega_cdm": omega_cdm,
#             "N_ur": float(pardict["N_ur"]),
#             "N_ncdm": int(pardict["N_ncdm"]),
#             "m_ncdm": pardict["m_ncdm"],
#             "Omega_k": float(pardict["Omega_k"]),
#             "tau_reio": float(pardict["tau_reio"]),
#         }
#     )
#     M.set(
#         {
#             "output": "mPk, dTk",
#             "P_k_max_1/Mpc": float(pardict["P_k_max_h/Mpc"]),
#             "z_max_pk": z_data,
#         }
#     )
#     M.compute()
    
#     DM_at_z = M.angular_distance(z_data) * (1. + z_data)
#     H_at_z = M.Hubble(z_data) * conts.c / 1000.0
#     rd = M.rs_drag()
    
#     theo_fAmp = M.scale_independent_growth_factor_f(z_data)*np.sqrt(M.pk_lin(kmpiv*h_fid*r_d_fid/(rd),z_data)*(h_fid*r_d_fid/(rd))**3.)/Amp_fid
    
#     theo_aperp = (DM_at_z) / DM_fid / rd * r_d_fid
#     theo_apara = H_z_fid/ (H_at_z) / rd * r_d_fid
    
#     # EHpk = EH98(kvec*h_fid*r_d_fid/(rd*h), z_data ,1.0, cosmo = M)
#     EHpk = EH98(kvec*h_fid*r_d_fid/(rd*h), z_data ,1.0, cosmo=M)*h**3*(r_d_fid/rd)**3
#     #Add in the primordial power spectrum calculation. 
#     # P_pk = Primordial(kvec*h_fid, A_s, float(pardict["n_s"]))
    
#     f = M.scale_independent_growth_factor_f(z_data)
#     sigma8 = M.sigma(8.0 / M.h(), z_data)
    
#     Pk_ratio = EHpk
    
#     Pkshape_ratio_prime = slope_at_x(np.log(kvec),np.log(Pk_ratio/Pk_ratio_fid))
#     # Pkshape_ratio_prime = slope_at_x(np.log(kvec),np.log(EHpk/EHpk_fid))
#     theo_mslope = np.interp(kmpiv, kvec, Pkshape_ratio_prime)
    
#     d_dk = FinDiff(0, np.log(kvec), 1, acc=10)
    
#     transfer_new = EH98_transfer(kvec*h_fid*r_d_fid/(rd*h), z_data ,1.0, cosmo=M)*(r_d_fid/rd)**3
    
#     ratio_dash_dash = d_dk(np.log(transfer_new/transfer_fid))
#     mslope_dash = interpolate.interp1d(kvec, ratio_dash_dash, kind='cubic')(kmpiv)
    
#     # print(theo_mslope, mslope_dash, np.abs((mslope_dash - theo_mslope)/theo_mslope))
    
#     d_dk_scale = FinDiff(0, np.log(kvec*h_fid), 1, acc=10)
    
#     # transfer_scale = interpolate.interp1d(kvec*h_fid*r_d_fid/rd, transfer_new, kind='cubic', fill_value='extrapolate')(kvec*h_fid)
    
#     # ratio_dash = d_dk_scale(np.log(transfer_scale/transfer_fid))
    
#     transfer_class_new = M.get_transfer(z=z_data)
#     k_new = transfer_class_new['k (h/Mpc)']
#     transfer_class = interpolate.interp1d(k_new, transfer_class_new['d_tot']*(r_d_fid/rd)**3, kind='cubic')(kvec*h_fid*r_d_fid/(rd*h))
    
#     # ratio_dash = d_dk_scale(np.log(transfer_new/transfer_fid))
    
#     transfer_ratio_new = transfer_class/transfer_new
#     smooth_new = smooth(transfer_ratio_new/transfer_ratio)
    
#     transfer_smooth = smooth_new*transfer_new*transfer_ratio
    
#     pk_class = []
#     for i in range(len(kvec)):
#         pk_class.append(M.pk_lin(k=kvec[i]*h_fid*r_d_fid/rd, z=z_data)*(r_d_fid/rd)**3*h**3)
#     pk_class = np.array(pk_class)
    
#     pk_ratio_new = pk_class/EHpk
#     smooth_pk_new = smooth(pk_ratio_new/pk_ratio_fid)
    
#     pk_smooth = smooth_pk_new*EHpk*pk_ratio_fid
    
#     # pardict_new = copy.deepcopy(pardict)
    
#     # pardict_new['ln10^{10}A_s'] = ln10As
#     # pardict_new['h'] = h
#     # pardict_new['omega_cdm'] = omega_cdm
#     # pardict_new['omega_b'] = omega_b
    
#     # fittingdata_new = FittingData(pardict_new)
#     # birdmodel = BirdModel(pardict_new, template=True, direct=True, fittingdata=fittingdata_new, window = str(fittingdata_new.data["windows"]), Shapefit=True)
    
#     # P11l = birdmodel.correlator.bird.P11l
#     # P11l_new = interpolate.interp1d(birdmodel.correlator.co.k, P11l, kind='cubic', axis=-1, fill_value = 'extrapolate')(kvec*h_fid*r_d_fid/(rd*h))*(r_d_fid/rd)**3
#     # P11l_new = P11l_new.reshape((P11l_new.shape[0]*P11l_new.shape[1], P11l_new.shape[2]))
    
#     # Pctl = birdmodel.correlator.bird.Pctl
#     # Pctl_new = interpolate.interp1d(birdmodel.correlator.co.k, Pctl, kind='cubic', axis=-1, fill_value='extrapolate')(kvec*h_fid*r_d_fid/(rd*h))*(r_d_fid/rd)**3
#     # Pctl_new = Pctl_new.reshape((Pctl_new.shape[0]*Pctl_new.shape[1], Pctl_new.shape[2]))
    
#     # Ploopl = birdmodel.correlator.bird.Ploopl
#     # Ploopl_new = interpolate.interp1d(birdmodel.correlator.co.k, Ploopl, kind='cubic', axis=-1, fill_value='extrapolate')(kvec*h_fid*r_d_fid/(rd*h))*(r_d_fid/rd)**3
#     # Ploopl_new = Ploopl_new.reshape((Ploopl_new.shape[0]*Ploopl_new.shape[1], Ploopl_new.shape[2]))
    
#     # deriv_lin = []
#     # for i in range(P11l_new.shape[0]):
#     #     if np.any(i == skip_lin):
#     #         continue
#     #     ratio_lin = d_dk(np.log(np.abs(P11l_new[i]/P11l_fid[i])))
#     #     deriv_lin.append(interpolate.interp1d(kvec*h_fid, ratio_lin, kind='cubic')(kmpiv*h_fid))
#     # deriv_lin = np.array(deriv_lin)
    
#     # deriv_ct = []
#     # for i in range(Pctl_new.shape[0]):
#     #     if np.any(i == skip_ct):
#     #         continue
#     #     ratio_ct = d_dk(np.log(np.abs(Pctl_new[i]/Pctl_fid[i])))
#     #     deriv_ct.append(interpolate.interp1d(kvec*h_fid, ratio_ct, kind='cubic')(kmpiv*h_fid))
#     # deriv_ct = np.array(deriv_ct)
    
#     # deriv_loop = []
#     # for i in range(Ploopl_new.shape[0]):
#     #     if np.any(i == skip_loop):
#     #         continue
#     #     ratio_loop = d_dk(np.log(np.abs(Ploopl_new[i]/Ploopl_fid[i])))/2.0
#     #     deriv_loop.append(interpolate.interp1d(kvec*h_fid, ratio_loop, kind='cubic')(kmpiv*h_fid))
#     # deriv_loop = np.array(deriv_loop)
    
#     # mslope_new = (np.sum(deriv_lin) + np.sum(deriv_ct) + np.sum(deriv_loop))/(len(deriv_lin) + len(deriv_ct) + len(deriv_loop))
    
#     # ratio_dash = d_dk(np.log(transfer_class/transfer_class_fid))
    
#     # ratio_dash = d_dk(np.log(np.abs(transfer_smooth/transfer_smooth_fid)))
    
#     ratio_dash = d_dk(np.log(pk_class/pk_class_fid))    
    
#     mslope_new = interpolate.interp1d(kvec*h_fid, ratio_dash, kind='cubic')(kmpiv*h_fid)
    
#     print(mslope_dash, mslope_new, np.abs((mslope_new - mslope_dash)/mslope_dash))
    
#     # print(theo_fAmp*(h/h_fid)**1.5*fsigma8_fid, f*sigma8, np.abs((theo_fAmp*(h/h_fid)**1.5*fsigma8_fid - f*sigma8)/(f*sigma8)))
    
#     return np.array([theo_aperp, theo_apara, theo_fAmp, theo_mslope, f*sigma8, theo_fAmp*(h/h_fid)**1.5, mslope_dash, mslope_new])

def dewiggle(k_lin, Pk_lin, n = 10, kmin = 0.01, kmax = 1.0, return_k = False):
    
    from scipy.fftpack import dst, idst
    from findiff import FinDiff
    from scipy.signal import find_peaks
    
    Pk = CubicSpline(np.log(k_lin), np.log(Pk_lin))
    length = 2**n

    k = np.linspace(kmin, kmax, length)
    logkPk = Pk(np.log(k)) + np.log(k)
    
    Pk_dst = dst(logkPk, norm='ortho')

    Pk_dst_even = Pk_dst[0:][::2]

    Pk_dst_odd = Pk_dst[1:][::2]
    
    n_range = np.arange(0, length)

    n_even = n_range[0:][::2]

    n_odd = n_range[1:][::2]
    
    Pk_spline_even = CubicSpline(n_even, Pk_dst_even)(np.linspace(n_even[0], n_even[-1], 2*length))
    Pk_spline_odd = CubicSpline(n_odd, Pk_dst_odd)(np.linspace(n_odd[0], n_odd[-1], 2*length))
    
    # dn = 2
    
    dn_even = np.linspace(n_even[0], n_even[-1], 2*length)[1] - np.linspace(n_even[0], n_even[-1], 2*length)[0]
    dn_odd = np.linspace(n_odd[0], n_odd[-1], 2*length)[1] - np.linspace(n_odd[0], n_odd[-1], 2*length)[0]
    
    d2_dn2_even = FinDiff(0, dn_even, 2, acc=10)
    d2_dn2_odd = FinDiff(0, dn_odd, 2, acc=10)
    
    dpkeven_dn = np.interp(n_even, np.linspace(n_even[0], n_even[-1], 2*length), d2_dn2_even(Pk_spline_even))
    dpkodd_dn = np.interp(n_odd, np.linspace(n_odd[0], n_odd[-1], 2*length), d2_dn2_odd(Pk_spline_odd))
    
    # d2_dn2 = FinDiff(0, dn, 2, acc = 10)
    
    # dpkeven_dn = d2_dn2(Pk_dst_even)

    # dpkodd_dn = d2_dn2(Pk_dst_odd)
    
    peaks_even = find_peaks(dpkeven_dn)[0]

    peaks_odd = find_peaks(dpkodd_dn)[0]
    
    trough_even = find_peaks(-dpkeven_dn)[0]

    trough_odd = find_peaks(-dpkodd_dn)[0]
    
    i_min_even = trough_even[0] - 3

    i_min_odd = trough_odd[0] - 3
    
    i_max_even = peaks_even[1] + 10

    i_max_odd = peaks_odd[1] + 20
    
    Pk_dst_even_scale = (n_even + 1)**2*Pk_dst_even

    Pk_dst_odd_scale = (n_odd + 1)**2*Pk_dst_odd

    remove_even = np.arange(i_min_even, i_max_even+1)

    remove_odd = np.arange(i_min_odd, i_max_odd+1)
    
    Pk_dst_even_scale_del = np.delete(Pk_dst_even_scale, remove_even)

    Pk_dst_odd_scale_del = np.delete(Pk_dst_odd_scale, remove_odd)

    n_even_del = np.delete(n_even, remove_even)

    n_odd_del = np.delete(n_odd, remove_odd)
    
    pk_even = CubicSpline(n_even_del, Pk_dst_even_scale_del)

    pk_odd = CubicSpline(n_odd_del, Pk_dst_odd_scale_del)

    even_pk = pk_even(n_even)/(n_even+1)**2

    odd_pk = pk_odd(n_odd)/(n_odd+1)**2

    dst_pk = np.zeros(length)

    dst_pk[n_even] = even_pk

    dst_pk[n_odd] = odd_pk

    nw_logkpk = idst(dst_pk, norm = 'ortho')

    Pk_nw = np.exp(nw_logkpk - np.log(k))
    
    if return_k == False:
        return Pk_nw
    else:
        return Pk_nw, k
    
def smooth_wallisch2018(ks, pk, ii_l=None, ii_r=None, extrap_min=1e-5, extrap_max=10, N=16):
    """Implement the wiggle/no-wiggle split procedure from Benjamin Wallisch's thesis (arXiv:1810.02800)"""
    
    from scipy.fftpack import dst, idst
    from scipy.ndimage import gaussian_filter
    from scipy.signal import argrelmin, argrelmax

    # put onto a linear grid
    kgrid = np.linspace(extrap_min, extrap_max, 2**N)
    lnps = interpolate.InterpolatedUnivariateSpline(ks, np.log(ks * pk), ext=0)(kgrid)

    # sine transform
    dst_ps = dst(lnps)
    dst_odd = dst_ps[1::2]
    dst_even = dst_ps[0::2]

    # find the BAO regions
    if ii_l is None or ii_r is None:
        d2_even = np.gradient(np.gradient(dst_even))
        ii_l = argrelmin(gaussian_filter(d2_even, 4))[0][0]
        ii_r = argrelmax(gaussian_filter(d2_even, 4))[0][1]

        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis)
        iis_div[0] = 1.0
        cutiis_even = (iis > (ii_l - 3)) * (iis < (ii_r + 10))

        d2_odd = np.gradient(np.gradient(dst_odd))
        ii_l = argrelmin(gaussian_filter(d2_odd, 4))[0][0]
        ii_r = argrelmax(gaussian_filter(d2_odd, 4))[0][1]

        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis)
        iis_div[0] = 1.0
        cutiis_odd = (iis > (ii_l - 3)) * (iis < (ii_r + 20))

    else:
        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis)
        iis_div[0] = 1.0
        cutiis_odd = (iis > ii_l) * (iis < ii_r)
        cutiis_even = (iis > ii_l) * (iis < ii_r)

    # ... and interpolate over them
    interp_odd = interpolate.interp1d(iis[~cutiis_odd], (iis**2 * dst_odd)[~cutiis_odd], kind="cubic")(iis) / iis_div**2
    interp_odd[0] = dst_odd[0]

    interp_even = interpolate.interp1d(iis[~cutiis_even], (iis**2 * dst_even)[~cutiis_even], kind="cubic")(iis) / iis_div**2
    interp_even[0] = dst_even[0]

    # Transform back
    interp = np.zeros_like(dst_ps)
    interp[0::2] = interp_even
    interp[1::2] = interp_odd

    lnps_nw = idst(interp) / 2**17

    return interpolate.InterpolatedUnivariateSpline(kgrid, np.exp(lnps_nw) / kgrid, ext=1)(ks)
    
def smooth_hinton2017(ks, pk, degree=13, sigma=1, weight=0.5, **kwargs):
    """Smooth power spectrum based on Hinton 2017 polynomial method"""
    # logging.debug("Smoothing spectrum using Hinton 2017 method")
    log_ks = np.log(ks)
    log_pk = np.log(pk)
    index = np.argmax(pk)
    maxk2 = log_ks[index]
    if sigma < 0.001:
        gauss = 0.0
    else:
        gauss = np.exp(-0.5 * np.power(((log_ks - maxk2) / sigma), 2))
    w = np.ones(pk.size) - weight * gauss
    z = np.polyfit(log_ks, log_pk, degree, w=w)
    p = np.poly1d(z)
    polyval = p(log_ks)
    pk_smoothed = np.exp(polyval)
    return pk_smoothed

def grid_params(pardict, params):
    if with_w0 == True:
        ln10As, h, omega_cdm, omega_b, w = params[:5]
    elif with_w0_wa == True:
        ln10As, h, omega_cdm, omega_b, w, wa = params[:6]
    elif with_omegak == True:
        ln10As, h, omega_cdm, omega_b, omegak = params[:5]
    else:
        ln10As, h, omega_cdm, omega_b = params[:4]
        
    print(params)
    
    A_s = np.exp(ln10As)/1.0e10
    H_0 = 100.0*h
    
    M = Class()
    
    z_data = float(pardict["z_pk"][0])
    
    if with_w0 == True:
        cosmo = cosmology.cosmology.Cosmology(h = h, Omega0_b = omega_b/h**2, Omega0_cdm = omega_cdm/h**2, 
                                              N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
                                              A_s = A_s, N_ncdm = int(pardict["N_ncdm"]), Omega_k = float(pardict["Omega_k"]), 
                                              tau_reio = float(pardict["tau_reio"]), P_z_max = z_data, Omega0_lambda = 0.0, w0_fld = w)
                
    elif with_w0_wa == True:
        cosmo = cosmology.cosmology.Cosmology(h = h, Omega0_b = omega_b/h**2, Omega0_cdm = omega_cdm/h**2, 
                                              N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
                                              A_s = A_s, N_ncdm = int(pardict["N_ncdm"]), Omega_k = float(pardict["Omega_k"]), 
                                              tau_reio = float(pardict["tau_reio"]), P_z_max = z_data, Omega0_lambda = 0.0, w0_fid = w, wa_fld = wa)
        
    elif with_omegak == True:
        cosmo = cosmology.cosmology.Cosmology(h = h, Omega0_b = omega_b/h**2, Omega0_cdm = omega_cdm/h**2, 
                                              N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
                                              A_s = A_s, N_ncdm = int(pardict["N_ncdm"]), Omega_k = omegak, 
                                              tau_reio = float(pardict["tau_reio"]), P_z_max = z_data)
    else:
        # from nbodykit import cosmology
        cosmo = cosmology.cosmology.Cosmology(h = h, Omega0_b = omega_b/h**2, Omega0_cdm = omega_cdm/h**2, 
                                              N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
                                              A_s = A_s, N_ncdm = int(pardict["N_ncdm"]), Omega_k = float(pardict["Omega_k"]), 
                                              tau_reio = float(pardict["tau_reio"]), P_z_max = z_data)
    # EHpk = (r_d_fid/rd)**3*cosmology.power.linear.LinearPower(cosmo, z_data, transfer='NoWiggleEisensteinHu')(kvec*h_fid*r_d_fid/(rd*h))
    
    # cosmo = cosmology.cosmology.Cosmology(h = 0.6736, Omega0_b = 0.02237/0.6736**2, Omega0_cdm = 0.12/0.6736**2, 
    #                                       N_ur = 2.0328, P_k_max = 100.0, n_s = 0.9649, 
    #                                       A_s = 2.083e-9, N_ncdm = 1, Omega_k = 0.0, 
    #                                       tau_reio = 0.0544, P_z_max = 0.8)
    
    M.set(
        {
            "A_s": A_s,
            "n_s": float(pardict["n_s"]),
            "H0": H_0,
            "omega_b": omega_b,
            "omega_cdm": omega_cdm,
            "N_ur": float(pardict["N_ur"]),
            "N_ncdm": int(pardict["N_ncdm"]),
            "m_ncdm": pardict["m_ncdm"],
            "tau_reio": float(pardict["tau_reio"]),
        }
    )
    
    if "w" in pardict.keys():
        M.set(
            {"Omega_Lambda": 0.0,
             "w0_fld": w
                })
        if "wa" in pardict.keys():
            M.set(
                {"wa_fld": wa
                })
    if with_omegak == True:
        M.set({
        "Omega_k": omegak})
        
    else:
        M.set({
        "Omega_k": float(pardict["Omega_k"])})
            
            
    M.set(
        {
            "output": "mPk, dTk",
            "P_k_max_1/Mpc": float(pardict["P_k_max_h/Mpc"]),
            "z_max_pk": z_data,
        }
    )
    M.compute()
    
    DM_at_z = M.angular_distance(z_data) * (1. + z_data)
    H_at_z = M.Hubble(z_data) * conts.c / 1000.0
    rd = M.rs_drag()*h
    
    # EH98_new  = (r_d_fid/rd)**3*cosmology.power.transfers.NoWiggleEisensteinHu(cosmo, z_data)(kvec*r_d_fid/(rd))**2
    # kmpiv_new = kmpiv*h_fid*r_d_fid/rd/h
    # kmpiv_CLASS_new = kmpiv_CLASS*h_fid*r_d_fid/rd/h
    # kmpiv_EH98_new = kmpiv_EH98*h_fid*r_d_fid/rd/h
    
    kmpiv_CLASS_new = kmpiv_CLASS*r_d_fid/rd
    # kmpiv_EH98_new = kmpiv_EH98*r_d_fid/rd
    # print(kmpiv_CLASS_new, kmpiv_EH98_new)
    
    # kmpiv_new = bisect(CubicSpline(kvec, FinDiff(0, np.log(kvec), 2, acc=10)(np.log(EH98_new/EH98_fid))), 0.01, 0.1)
    # print(kmpiv_new, 0.02*h_fid*r_d_fid/rd/h)
    # kmpiv_dash = 0.02*h_fid*r_d_fid/rd/h
    
    # print(M.scale_independent_growth_factor(z_data), cosmo.scale_independent_growth_factor(z_data))
    
    theo_fAmp_dash = M.scale_independent_growth_factor_f(z_data)*np.sqrt(M.pk_lin(kmpiv_CLASS_new*h_fid,z_data)*(h*r_d_fid/(rd))**3.)/Amp_fid
    # theo_fAmp = M.scale_independent_growth_factor_f(z_data)*np.sqrt(M.pk_lin(kmpiv_EH98_new*h,z_data)*(h_fid*r_d_fid/(rd))**3.)/Amp_fid
    
    theo_fAmp = M.scale_independent_growth_factor_f(z_data)*np.sqrt(M.pk_lin(kmpiv_CLASS_new*h,z_data)*(h*r_d_fid/(rd))**3.)/Amp_fid

    
    # theo_fAmp1 = M.scale_independent_growth_factor_f(z_data)*np.sqrt(np.array([M.pk_lin(ratio_i*kmpiv_new*h,z_data) for ratio_i in ratio])*(h*r_d_fid/(rd))**3.)/Amp_fid_1
    # theo_fAmp2 = M.scale_independent_growth_factor_f(z_data)*np.sqrt(np.array([M.pk_lin(ratio_i*kmpiv_new*h,z_data) for ratio_i in ratio_flip])*(h*r_d_fid/(rd))**3.)/Amp_fid_2
 
    # theo_fAmp3 = np.mean((theo_fAmp1*theo_fAmp2)**(1/2.0))
    
    # print(theo_fAmp3)
    
    # theo_fAmp_prime = M.scale_independent_growth_factor_f(z_data)*np.sqrt(EH98(kmpiv/h_fid*h*r_d_fid/rd, z_data, 1.0, M)*(r_d_fid/rd)**3)/Amp_fid_prime
    
    # theo_fAmp = M.scale_independent_growth_factor_f(z_data)*np.sqrt((r_d_fid/rd)**3*cosmology.power.linear.LinearPower(cosmo, z_data, 
    #                                                                                                     transfer='NoWiggleEisensteinHu')(kmpiv_CLASS_new))/Amp_fid
    # print(kmpiv/h_fid*h*r_d_fid/rd, (r_d_fid/rd)**3, np.sqrt(cosmology.power.linear.LinearPower(cosmo, z_data, 
    #                                                                                                    transfer='NoWiggleEisensteinHu')(kmpiv/h_fid*h*r_d_fid/rd)))
    
    theo_aperp = (DM_at_z) / DM_fid / (rd/h) * (r_d_fid/h_fid)
    theo_apara = H_z_fid/ (H_at_z) / (rd/h) * (r_d_fid/h_fid)
    
    # print((r_d_fid/h_fid)/(rd/h), DM_at_z/DM_fid, H_z_fid/H_at_z)

    
    f = M.scale_independent_growth_factor_f(z_data)
    sigma8 = M.sigma(8.0 / M.h(), z_data)
    sigma_s8 = M.sigma(rd/r_d_fid*8.0/M.h(), z_data)
    
    # EHpk = (r_d_fid/rd)**3*cosmology.power.linear.LinearPower(cosmo, z_data, transfer='NoWiggleEisensteinHu')(kvec*h_fid*r_d_fid/(rd*h))
    EHpk = (r_d_fid/rd)**3*cosmology.power.transfers.NoWiggleEisensteinHu(cosmo, z_data)(kvec*r_d_fid/(rd))**2
    if int(pardict["N_ncdm"]) > 0:
        Plin = (r_d_fid/rd)**3*np.array([M.pk_cb_lin(ki * M.h(), z_data) * M.h() ** 3 for ki in kvec*r_d_fid/rd])
    else:
        Plin = (r_d_fid/rd)**3*np.array([M.pk_lin(ki * M.h(), z_data) * M.h() ** 3 for ki in kvec*r_d_fid/rd])
    
    # Plin = (r_d_fid/rd)**3*np.array([M.pk_lin(ki * M.h(), z_data) * M.h() ** 3 for ki in kvec*r_d_fid/rd])
        
    # np.save('Plin_w_' + str(w) + '.npy', Plin)
    EHpk_prime = smooth_wallisch2018(kvec*r_d_fid/rd, Plin)
    EHpk_dash = smooth_hinton2017(kvec*r_d_fid/rd, Plin)
    # EHpk = (r_d_fid/rd)**3*cosmology.power.linear.LinearPower(cosmo, z_data, transfer='NoWiggleEisensteinHu')(kvec*r_d_fid/(rd))

    
    # EHpk_dash = (r_d_fid/rd)**3*cosmology.power.transfers.NoWiggleEisensteinHu(cosmo, z_data)(kvec*h_fid*r_d_fid/(rd*h))
    # EHpk_dash = (r_d_fid/rd)**3*np.array([M.pk_lin(kveci*h_fid*r_d_fid/rd, z_data) for kveci in kvec])*(h)**3
    # EHpk_dash, k_new = dewiggle(kvec*h_fid/h*r_d_fid/rd, np.array([M.pk_lin(kveci*h_fid*r_d_fid/rd, z_data) for kveci in kvec])*h**3, n=12, return_k=True)
    # EHpk_dash = (r_d_fid/rd)**3*EHpk_dash/Primordial(k_new, A_s, float(pardict["n_s"]))
    # EHpk_dash = (r_d_fid/rd)**3*EHpk_dash*(h_fid/h)**3
    # print(len(EHpk_dash)
    
    kstart = np.argmin(np.abs(kvec-0.01))
    kend = np.argmin(np.abs(kvec-0.1))
    Pk_ratio = EHpk
    
    Pkshape_ratio = slope_at_x(np.log(kvec[kstart:kend]),np.log(Pk_ratio[kstart:kend]/Pk_ratio_fid[kstart:kend]))
    # Pkshape_ratio_prime = slope_at_x(kvec,np.log(Pk_ratio/Pk_ratio_fid))
    theo_mslope = np.interp(kmpiv_CLASS, kvec[kstart:kend], Pkshape_ratio)
    
    # Pkshape_ratio_dash = slope_at_x(np.log(kvec), np.log(EHpk_dash/transfer_fid_dash))
    # Pkshape_ratio_dash = slope_at_x(np.log(k_new), np.log(EHpk_dash/transfer_fid_dash))
    # Pkshape_ratio_dash = slope_at_x(np.log(kvec), np.log((r_d_fid*h_fid/rd/h)**3*cosmology.power.transfers.NoWiggleEisensteinHu(cosmo, z_data)(kvec*h_fid*r_d_fid/(rd*h))**2/transfer_fid_dash))
    # theo_mslope_dash = np.interp(kmpiv_EH98, kvec, Pkshape_ratio_dash)
    # theo_mslope_dash = np.interp(kmpiv_CLASS_new, k_new, Pkshape_ratio_dash)

    # d_dx = FinDiff(0, (kvec)[1] - (kvec)[0], 1, acc=10)
    # ratio_prime = d_dx(np.log(Pk_ratio/Pk_ratio_fid))
    # theo_mslope_new = np.interp(kmpiv_CLASS, kvec, ratio_prime)*kmpiv_CLASS
    # np.save('Plin_no_wiggle_w_' + str(np.int32(np.floor(-(w+0.7)/0.074))) + '.npy', EHpk_dash/EH98_dash_fid)
    Pkshape_ratio_dash = slope_at_x(np.log(kvec[kstart:kend]), np.log(EHpk_dash[kstart:kend]/EH98_dash_fid[kstart:kend]))
    theo_mslope_new = np.interp(kmpiv_CLASS, kvec[kstart:kend], Pkshape_ratio_dash)
    
    Pkshape_ratio_prime = slope_at_x(np.log(kvec[kstart:kend]), np.log(np.divide(EHpk_prime[kstart:kend], EH98_prime_fid[kstart:kend], out = np.ones_like(EHpk_prime[kstart:kend]), where = EH98_prime_fid[kstart:kend]!=0)))
    theo_mslope_prime = np.interp(kmpiv_CLASS, kvec[kstart:kend], Pkshape_ratio_prime)

    # print(np.min(Pk_ratio), np.min(EHpk_dash), np.min(EHpk_prime))
    
    # np.save('test.npy', [Pk_ratio, EHpk_dash, EHpk_prime])
    
    # Pkshape_ratio_new = slope_at_x(np.log(kvec), np.log(EH98_new/EH98_fid))
    
    # theo_mslope_new = np.interp(kmpiv_CLASS, kvec, Pkshape_ratio_new)
    
    # print(theo_aperp, theo_apara, theo_fAmp, theo_mslope, f*sigma8/fsigma8_fid, theo_mslope_dash, theo_fAmp_dash, theo_fAmp_prime, theo_mslope_new)
    
    # return np.array([theo_aperp, theo_apara, theo_fAmp, theo_mslope, f*sigma8, theo_mslope_dash, theo_fAmp_dash, theo_fAmp_prime ,theo_mslope_new])
    
    # factor = -0.2269*(rd/r_d_fid)**2 + 0.6676*(rd/r_d_fid) + 0.5595
    
    # factor = (rd/r_d_fid)**0.22144
    # sigma8_sq = (theo_fAmp*factor*fsigma8_fid/M.scale_independent_growth_factor_f(z_data)/sigma8_fid)**2
    
    # m = -0.6*np.log(Pk_ratio[682]/Pk_ratio_fid[682]/sigma8_sq)
    
    print(theo_aperp, theo_apara, theo_fAmp, theo_mslope, theo_mslope_new, theo_mslope_prime, f*sigma8/fsigma8_fid, theo_fAmp_dash)
    
    return np.array([theo_aperp, theo_apara, theo_fAmp, theo_mslope, theo_mslope_new, theo_mslope_prime, f*sigma8/fsigma8_fid, theo_fAmp_dash])
    
    # print(theo_aperp, theo_apara, theo_fAmp, theo_mslope, f*sigma8, theo_mslope_dash, theo_fAmp_dash, theo_mslope_new)
    
    # return np.array([theo_aperp, theo_apara, theo_fAmp, theo_mslope, f*sigma8, theo_mslope_dash, theo_fAmp_dash, theo_mslope_new])
    
# def find_new_kmpiv(gamma_eff, kvec):
#     ktest = np.linspace(0.001, 1.0, 8192)
#     model = CubicSpline(kvec, gamma_eff)(ktest)
#     result = FinDiff(0, ktest[1] - ktest[0], 1, acc=10)(model)
#     return ktest[np.argmin(result)]

def EH98_kmpiv(cosmo):
    h = cosmo.h()
    Obh2 = cosmo.Omega_b()*h**2
    Omh2 = cosmo.Omega_m()*h**2
    sound_horizon = h * 44.5 * np.log(9.83/Omh2) / \
    np.sqrt(1 + 10 * Obh2** 0.75) # in Mpc/h
    
    return np.sqrt(np.sqrt(15.0)/5.0)*100.0/43.0/sound_horizon

def CLASS_kmpiv(cosmo):
    rd = cosmo.rs_drag()*cosmo.h()
    return np.sqrt(np.sqrt(15.0)/5.0)*100.0/43.0/rd
    

def EH98(kvector, redshift, scaling_factor, cosmo = None):
    #This code calculate the EH98 no-wiggle power spectrum using the same formula in Nbodykit but with the sould horizon and equality from CLASS. 
    h = cosmo.h()
    Obh2 = cosmo.Omega_b()*h**2
    Omh2 = cosmo.Omega_m()*h**2
    f_baryon = Obh2/Omh2
    
    #Uncomment this block will mean you calculate the sound horizon and equality k vector with the approximation in Nbodykit. It would give you the same result
    #as with the Nbodykit. 
    #--------------------------------------------------------------------------
    # theta_cmb = cosmo.T_cmb() / 2.7
    # # print(theta_cmb)

    # # wavenumber of equality
    # k_eq = 0.0746 * Omh2 * theta_cmb ** (-2) # units of 1/Mpc

    # sound_horizon = h * 44.5 * np.log(9.83/Omh2) / \
    #                     np.sqrt(1 + 10 * Obh2** 0.75) # in Mpc/h
    # alpha_gamma = 1 - 0.328 * np.log(431*Omh2) * f_baryon + \
    #                     0.38* np.log(22.3*Omh2) * f_baryon ** 2
                        
    # rd = sound_horizon
    #--------------------------------------------------------------------------
                        
    
    
    
    # print(norm, cosmo.sigma(8.0/h, 0.0))
    
    #We obtain the sound horizon and equality k vector from CLASS. Comment this block if you want to use the same approximation as in the Nbodykit.  
    #--------------------------------------------------------------------------
    k_eq = cosmo.k_eq()*scaling_factor
    rd = cosmo.rs_drag()*h/scaling_factor
    alpha_gamma = 1 - 0.328 * np.log(431*Omh2) * f_baryon + 0.38* np.log(22.3*Omh2) * f_baryon ** 2
    #--------------------------------------------------------------------------
    
    #This block calculates the no-wiggle transfer function from the EH98 paper. These are also the same formulae in Nbodykit. 
    #--------------------------------------------------------------------------
    k = kvector*h
    ks = k*rd/h
    q = k/(13.41*k_eq)
    
    gamma_eff = Omh2 * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
    # print(find_new_kmpiv(gamma_eff, kvector), np.sqrt(np.sqrt(15.0)/5.0)*100.0/43.0/rd)
    q_eff = q * Omh2 / gamma_eff
    L0 = np.log(2*np.e + 1.8 * q_eff)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)

    T = L0/(L0+C0*q_eff**2)
    #--------------------------------------------------------------------------
    
    #Scale the transfer function by the growth factor. I am setting z = 0 here since we will scale it by with sigma8 at z = 0. 
    transfer = T*cosmo.scale_independent_growth_factor(0.0)
    
    ns = cosmo.n_s()
    
    #The unnormalized power spectrum. 
    Pk_unnorm = transfer**2*kvector**ns
    
    #Calculate the normalization. 
    norm = (cosmo.sigma(8.0/h, 0.0)/sigma_r(Pk_unnorm, kvector, 8.0))**2
    # ns = cosmo.n_s()
    
    # NWTD2 = (T*cosmo.scale_independent_growth_factor(redshift))**2
    
    # # NWTD2 = (T*cosmo.scale_independent_growth_factor(redshift))**2 * kvector**(ns)*(cosmo.sigma(8.0/h, redshift)/cosmo.sigma(8.0/h, 0.0))**2
    
    # Pk_unnorm = NWTD2*kvector**ns
    
    # norm = (cosmo.sigma(8.0/h, 0.0)/sigma_r(Pk_unnorm, kvector, 8.0))**2
    
    # # deltah = 1.94e-5 * cosmo.Omega_m()**(-0.785-0.05*np.log(cosmo.Omega_m()))*np.exp(-0.95*(ns-1)-0.169*(ns-1)**2.)
    # # # deltah = 1.94e-5 * cosmo.Omega_m()**(-0.785-0.05*np.log(cosmo.Omega_m()))*np.exp((ns-1) + 1.97*(ns-1)**2)
    
    # # Pk = 2*np.pi**2. * deltah**2. * (kvector)**(ns) * NWTD2 /cosmo.Hubble(0)**(3.+ns)
    
    #Multiply the normalization to the unnormalized power spectrum at the correct redshift. 
    Pk = norm*kvector**ns*(T*cosmo.scale_independent_growth_factor(redshift))**2
    
    # print(sigma_r(Pk_unnorm, kvector, 8.0))
    
    # print(norm, Pk[0])
    
    return Pk
    
    # return (T*cosmo.scale_independent_growth_factor(redshift))**2
    # return T**2
                            
    
def sigma_r(Pk, k, r):
    #This is copy and paste from Nbodykit with a small modification to the input arguments. 
    r"""
    The mass fluctuation within a sphere of radius ``r``, in
    units of :math:`h^{-1} Mpc` at ``redshift``.

    This returns :math:`\sigma`, where

    .. math::

        \sigma^2 = \int_0^\infty \frac{k^3 P(k,z)}{2\pi^2} W^2_T(kr) \frac{dk}{k},

    where :math:`W_T(x) = 3/x^3 (\mathrm{sin}x - x\mathrm{cos}x)` is
    a top-hat filter in Fourier space.

    The value of this function with ``r=8`` returns
    :attr:`sigma8`, within numerical precision.

    Parameters
    ----------
    Pk : float, array_like
          The unnormlaized power spectrum 
    k : float, array_like
        The k array in h/Mpc for the input power spectrum. 
    r : float
        the radius of the sphere, in units of :math:`\mathrm{Mpc/h}`
    """
    import mcfit
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    R, sigmasq = mcfit.TophatVar(k, lowring=True)(Pk, extrap=True)

    return spline(R, sigmasq)(r)**0.5
    

# def EH98(kvector, redshift, scaling_factor, cosmo = None):
#     #This is the link to the paper: https://arxiv.org/pdf/astro-ph/9710252.pdf
#     #The input kvector should be in unit of h/Mpc after rescaling by rd. 
    
#     if cosmo is not None:
#         cdict = cosmo.get_current_derived_parameters(['z_d'])
#         h = cosmo.h()
#         H_at_z = cosmo.Hubble(redshift) * conts.c /1000. /(100.*h)
#         Omm = cosmo.Omega_m()
#         Omb = cosmo.Omega_b()
#         #Cannot find the following function. 
#         # Omc = cosmo.omegach2()/h**2.
#         Omc = cosmo.Omega0_cdm()
#         Omm_at_z = Omm*(1.+redshift)**3./H_at_z**2.
#         # print(Omm, Omm_at_z)
#         OmLambda_at_z = 1.-Omm_at_z
#         # OmLambda_at_z = (1.0-Omm)/(Omm*(1+redshift)**3 + (1.0-Omm))
#         ns = cosmo.n_s()
#         rs = cosmo.rs_drag()*h/scaling_factor
#         # rs = 44.5*np.log(9.83/Omm/h**2)/np.sqrt(1.0 + 10*(Omb*h**2)**(0.75))*cosmo.h()
        
#         # theta_cmb = cosmo.T_cmb()/2.7
#         # A_N0_Nm = (2.32 + 0.56*cosmo.Neff())**(-1.0)
        
#         Omnu = Omm-Omb-Omc
#         fnu = Omnu/Omm
#         fb = Omb/Omm
#         fnub = (Omb+Omnu)/Omm
#         fc = Omc/Omm
#         fcb = (Omc+Omb)/Omm
#         pc = 1./4.*(5-np.sqrt(1+24*fc))
#         pcb = 1./4.*(5-np.sqrt(1+24*fcb))
#         # Neff = cosmo.Neff()
#         # Omg = cosmo.Omega_g()
#         # Omr = Omg * (1. + Neff * (7./8.)*(4./11.)**(4./3.))
#         # aeq = Omr/(Omb+Omc)/(1-fnu)
#         # zeq = 1./aeq -1.
#         # zeq = 2.50*10**4*Omm*h**2*theta_cmb**(-4.0) - 1.0
#         # zeq = A_N0_Nm*10**5*Omm*h**2*theta_cmb**(-4.0) - 1.0
        
        
#         # Heq = cosmo.Hubble(zeq)
#         # keq = aeq*Heq*scaling_factor   
        
#         zeq = cosmo.z_eq()
#         keq = cosmo.k_eq()
        
#         # print(zeq, keq)
        
#         # print(keq, 7.46*10**(-2)*Omm*h**2*theta_cmb**(-2.0))
#         # keq = 7.46*10**(-2)*Omm*h**2*theta_cmb**(-2.0)
#         # keq = 0.1492*Omm*h**2*theta_cmb**(-2.0)*np.sqrt(A_N0_Nm)

#         zd = cdict['z_d']
#         # Ommh = Omm*h**2
#         # b1 = 0.313*(Ommh)**(-0.419)*(1.0+0.607*Ommh**0.674)
#         # b2 = 0.238*(Ommh)**0.223
#         # zd = 1291.0*(Ommh)**0.251/(1.0+0.659*(Ommh)**0.828)*(1.0+b1*Ommh**b2)
#         # print(cdict['z_d'], zd)
#         yd = (1.+zeq)/(1.+zd)
#         growth = cosmo.scale_independent_growth_factor(redshift)
        
#     # else:
#     #     H_at_z = Hz/(100.0*h)
#     #     Omm = Omm
#     #     Omb = Omb/h**2
#     #     Omc = Omc/h**2
#     #     Omm_at_z = Omm*(1.+redshift)**3./H_at_z**2.
#     #     OmLambda_at_z = 1.-Omm_at_z
#     #     ns = n_s
#     #     rs = rd*h/scaling_factor
#     #     Omnu = Omm-Omb-Omc
#     #     fnu = Omnu/Omm
#     #     fb = Omb/Omm
#     #     fnub = (Omb+Omnu)/Omm
#     #     fc = Omc/Omm
#     #     fcb = (Omc+Omb)/Omm
#     #     pc = 1./4.*(5-np.sqrt(1+24*fc))
#     #     pcb = 1./4.*(5-np.sqrt(1+24*fcb))
#     #     zeq = zeq
#     #     keq = keq
#     #     zd = z_d
#     #     yd = (1.+zeq)/(1.+zd)
#     #     growth = growth
        
#     if (fnu==0):
#         Nnu = 0.
#     else:
#         Nnu = 1.
#     #alpha_gamma = 1 - 0.328*np.log(431*Omm*h**2)*Omb/Omm + 0.38*np.log(22.3*Omm*h**2)*(Omb/Omm)**2
    
#     #There seems to be a mistake in this equation. 
#     alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu)*Nnu**0.2 + 0.169*fnu) \
#                 *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
    
#     # alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1.0-0.553*fnub + 0.126*fnub**3.) / (1.0 - 0.193*np.sqrt(fnu*Nnu) + 0.169*fnu*(Nnu)**0.2) \
#     #             *(1.+yd)**(pcb-pc) * (1.0+(pc-pcb)/2.0*(1.+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
                
#     #eff_shape = (alpha_gamma + (1.-alpha_gamma)/(1+(0.43*kvector*rs)**4.))
#     eff_shape = (np.sqrt(alpha_nu) + (1.-np.sqrt(alpha_nu))/(1.+(0.43*kvector*rs)**4.))
    
#     #add in q_test. 
#     # q_test = kvector/keq*7.46e-2
#     q_test = kvector/(keq*13.41)*h
#     # q0 = kvector/(keq/7.46e-2)/eff_shape
#     q0 = q_test/eff_shape
#     # q0 = kvector*theta_cmb**2/eff_shape
#     betac = (1.-0.949*fnub)**(-1.)
#     L0 = np.log(np.exp(1.)+1.84*np.sqrt(alpha_nu)*betac*q0)
#     C0 = 14.4 + 325./(1+60.5*q0**1.08)
#     # C0 = 14.4 + 325./(1+60.5*q0**1.11)
#     T0 = L0/(L0+C0*q0**2.)
#     if (fnu==0):
#         yfs=0.
#         qnu=3.92*q_test
#     else:
#         yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6.))*(Nnu*q_test/fnu)**2.
#         qnu = 3.92*q_test*np.sqrt(Nnu/fnu)
#     #The original code seems to be missing a factor of 5. 
#     D1 = (1.+zeq)/(1.+redshift)*5*Omm_at_z/2./(Omm_at_z**(4./7.) - OmLambda_at_z + (1.+Omm_at_z/2.)*(1.+OmLambda_at_z/70.))
#     Dcbnu = (fcb**(0.7/pcb)+(D1/(1.+yfs))**0.7)**(pcb/0.7) * D1**(1.-pcb)
#     Bk = 1. + 1.24*fnu**(0.64)*Nnu**(0.3+0.6*fnu)/(qnu**(-1.6)+qnu**0.8)
#     #
#     Tcbnu = T0*Dcbnu/D1*Bk
#     # Tcbnu = T0*Bk
#     # deltah = 1.94e-5 * Omm**(-0.785-0.05*np.log(Omm))*np.exp(-0.95*(ns-1)-0.169*(ns-1)**2.)
    
#     # growth = D1**2/((1.+zeq)*5.*Omm/2./(Omm**(4./7.)-(1.0-Omm)+(1.+Omm/2.0)*((1.0 + 1.0 - Omm)/70.)))**2
    
#     #The output power spectrum will be in the unit of (Mpc/h)^3. 
#     if cosmo is not None:
#         # Pk = 2*np.pi**2. * deltah**2. * (kvector)**(ns) * Tcbnu**2. * growth**2. /cosmo.Hubble(0)**(3.+ns)
#         Pk = Tcbnu**2
#         # Pk = Tcbnu**2. * growth**2
#     # else:
#     #     Pk = 2*np.pi**2. * deltah**2. * (kvector)**(ns) * Tcbnu**2. * growth**2. /Hubble**(3.+ns)
#     return Pk


def EH98_transfer(kvector, redshift, scaling_factor, cosmo = None, rd = None, z_d = None, Omm = None, zeq = None, keq = None, Hz = None, 
         h=None, Omb = None, Omc = None, n_s = None, growth = None, Hubble = None):
    
    #This is the link to the paper: https://arxiv.org/pdf/astro-ph/9710252.pdf
    #The input kvector should be in unit of h/Mpc after rescaling by rd. 
    
    if cosmo is not None:
        cdict = cosmo.get_current_derived_parameters(['z_d'])
        h = cosmo.h()
        H_at_z = cosmo.Hubble(redshift) * conts.c /1000. /(100.*h)
        Omm = cosmo.Omega_m()
        Omb = cosmo.Omega_b()
        #Cannot find the following function. 
        # Omc = cosmo.omegach2()/h**2.
        Omc = cosmo.Omega0_cdm()
        Omm_at_z = Omm*(1.+redshift)**3./H_at_z**2.
        OmLambda_at_z = 1.-Omm_at_z
        ns = cosmo.n_s()
        rs = cosmo.rs_drag()*h/scaling_factor
        Omnu = Omm-Omb-Omc
        fnu = Omnu/Omm
        fb = Omb/Omm
        fnub = (Omb+Omnu)/Omm
        fc = Omc/Omm
        fcb = (Omc+Omb)/Omm
        pc = 1./4.*(5-np.sqrt(1+24*fc))
        pcb = 1./4.*(5-np.sqrt(1+24*fcb))
        Neff = cosmo.Neff()
        Omg = cosmo.Omega_g()
        Omr = Omg * (1. + Neff * (7./8.)*(4./11.)**(4./3.))
        aeq = Omr/(Omb+Omc)/(1-fnu)
        zeq = 1./aeq -1.
        Heq = cosmo.Hubble(zeq)/h
        keq = aeq*Heq*scaling_factor   
        zd = cdict['z_d']
        yd = (1.+zeq)/(1.+zd)
        growth = cosmo.scale_independent_growth_factor(redshift)
        
    else:
        H_at_z = Hz/(100.0*h)
        Omm = Omm
        Omb = Omb/h**2
        Omc = Omc/h**2
        Omm_at_z = Omm*(1.+redshift)**3./H_at_z**2.
        OmLambda_at_z = 1.-Omm_at_z
        ns = n_s
        rs = rd*h/scaling_factor
        Omnu = Omm-Omb-Omc
        fnu = Omnu/Omm
        fb = Omb/Omm
        fnub = (Omb+Omnu)/Omm
        fc = Omc/Omm
        fcb = (Omc+Omb)/Omm
        pc = 1./4.*(5-np.sqrt(1+24*fc))
        pcb = 1./4.*(5-np.sqrt(1+24*fcb))
        zeq = zeq
        keq = keq
        zd = z_d
        yd = (1.+zeq)/(1.+zd)
        growth = growth
        
    if (fnu==0):
        Nnu = 0.
    else:
        Nnu = 1.
    #alpha_gamma = 1 - 0.328*np.log(431*Omm*h**2)*Omb/Omm + 0.38*np.log(22.3*Omm*h**2)*(Omb/Omm)**2
    
    #There seems to be a mistake in this equation. 
    # alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu)*Nnu**0.2 + 0.169*fnu) \
    #             *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
    
    alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu*Nnu) + 0.169*fnu*(Nnu)**0.2) \
                *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2.0*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
                
    #eff_shape = (alpha_gamma + (1.-alpha_gamma)/(1+(0.43*kvector*rs)**4.))
    eff_shape = (np.sqrt(alpha_nu) + (1.-np.sqrt(alpha_nu))/(1+(0.43*kvector*rs)**4.))
    
    #add in q_test. 
    q_test = kvector/keq*7.46e-2
    q0 = kvector/(keq/7.46e-2)/eff_shape
    betac = (1.-0.949*fnub)**(-1.)
    L0 = np.log(np.exp(1.)+1.84*np.sqrt(alpha_nu)*betac*q0)
    # C0 = 14.4 + 325./(1+60.5*q0**1.08)
    C0 = 14.4 + 325./(1+60.5*q0**1.11)
    T0 = L0/(L0+C0*q0**2.)
    if (fnu==0):
        yfs=0.
        qnu=3.92*q_test
        # qnu=3.92*q0
    else:
        yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6.))*(Nnu*q_test/fnu)**2.
        qnu = 3.92*q_test*np.sqrt(Nnu/fnu)
        # yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6.))*(Nnu*q0/fnu)**2.
        # qnu = 3.92*q0*np.sqrt(Nnu/fnu)
        
    #The original code seems to be missing a factor of 5. 
    D1 = (1.+zeq)/(1.+redshift)*5*Omm_at_z/2./(Omm_at_z**(4./7.) - OmLambda_at_z + (1.+Omm_at_z/2.)*(1.+OmLambda_at_z/70.))
    Dcbnu = (fcb**(0.7/pcb)+(D1/(1.+yfs))**0.7)**(pcb/0.7) * D1**(1.-pcb)
    Bk = 1. + 1.24*fnu**(0.64)*Nnu**(0.3+0.6*fnu)/(qnu**(-1.6)+qnu**0.8)
    #
    Tcbnu = T0*Dcbnu/D1*Bk
    
    return Tcbnu
    

def Primordial(k, A_s, n_s, k_p = 0.05):
    #k_p is in the unit of 1/Mpc, so the input power spectrum also need to be in 1/Mpc. 
    P_R = A_s*(k/k_p)**(n_s - 1.0)
    return P_R
    

def slope_at_x(xvector,yvector):
    diff = np.diff(yvector)/np.diff(xvector)
    diff = np.append(diff,diff[-1])
    return diff

def smooth(transfer_ratio):
    gradient_transfer = np.gradient(transfer_ratio, kvec)
    
    maxima_index = signal.argrelextrema(gradient_transfer, np.greater)[0]
    
    minima_index = signal.argrelextrema(gradient_transfer, np.less)[0]
    
    maxima = []
    maxima_k = []
    minima = []
    minima_k = []
    
    for i in range(len(maxima_index)):
        maxima.append(gradient_transfer[maxima_index[i]])
        maxima_k.append(kvec[maxima_index[i]])
        
    maxima = np.array(maxima)
    maxima_k = np.array(maxima_k)
    
    for i in range(len(minima_index)):
        minima.append(gradient_transfer[minima_index[i]])
        minima_k.append(kvec[minima_index[i]])
        
    minima = np.array(minima)
    minima_k = np.array(minima_k)
    
    smooth_maxima = interpolate.interp1d(maxima_k, maxima, kind='quadratic')(kvec[maxima_index[0]:np.int32(maxima_index[-1]+1)])
    smooth_minima = interpolate.interp1d(minima_k, minima, kind='quadratic')(kvec[minima_index[0]:np.int32(minima_index[-1]+1)])
    
    smooth_maxima_all = np.concatenate((gradient_transfer[:maxima_index[0]], smooth_maxima, gradient_transfer[np.int32(maxima_index[-1]+1):]))
    smooth_minima_all = np.concatenate((gradient_transfer[:minima_index[0]], smooth_minima, gradient_transfer[np.int32(minima_index[-1]+1):]))
    
    smooth_all = (smooth_maxima_all + smooth_minima_all)/2.0
    
    return smooth_all

if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    #The job number. 
    job_num = int(sys.argv[2])
    #Total number of parallel jobs. 
    job_total_num = int(sys.argv[3])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    
    # kvec = np.logspace(-5.0, 1.0, 1024)
    kvec = np.linspace(1e-5, 10.0, 1024*8)
    
    z_data = np.float64(pardict["z_pk"])[0]
    
    job_total = np.int32((2.0*float(pardict['order']) + 1.0)**(len(pardict['dx'])))
    
    length = np.int32(job_total/job_total_num)
    
    print(job_total, length)
    
    start = job_num*length
    
    nparams = len(pardict["dx"])
    
    order = int(pardict['order'])
    #The step size of each parameter dx must be written in the order of ln_10^10_As, h, omega_cdm, omega_b
    dx = np.float64(pardict['dx'])
    ln_10_10_As = float(pardict["ln10^{10}A_s"])
    h = float(pardict['h'])
    omega_cdm = float(pardict['omega_cdm'])
    omega_b = float(pardict['omega_b'])
    ln_10_10_As_all = np.linspace(ln_10_10_As - order*dx[0], ln_10_10_As + order*dx[0], 2*order+1)
    h_all = np.linspace(h - order*dx[1], h + order*dx[1], 2*order+1)
    omega_cdm_all = np.linspace(omega_cdm - order*dx[2], omega_cdm + order*dx[2], 2*order+1)
    omega_b_all = np.linspace(omega_b - order*dx[3], omega_b + order*dx[3], 2*order+1)
    
    with_w0 = False
    with_w0_wa = False
    with_omegak = False
    if "w" in pardict.keys():
        w = float(pardict['w'])
        with_w0 = True
        keyword = '_w0'
        w_all = np.linspace(w - order*dx[4], w + order*dx[4], 2*order+1)
        if "wa" in pardict.keys():
            wa = float(pardict['wa'])
            with_w0 = False
            with_w0_wa = True
            keyword = '_w0_wa'
            wa_all = np.linspace(wa - order*dx[5], wa + order*dx[5], 2*order+1)
            
    elif pardict['freepar'][-1] == 'Omega_k':
        with_omegak = True
        omegak = float(pardict['Omega_k'])
        keyword = '_omegak'
        omegak_all = np.linspace(omegak - order*dx[4], omegak + order*dx[4], 2*order+1)
            
    
    
    template = Class()
    
    template.set({
    "A_s": np.exp(float(pardict["ln10^{10}A_s"]))/1e10,
    "n_s": float(pardict["n_s"]),
    "H0": 100.0*float(pardict['h']),
    "omega_b": float(pardict['omega_b']),
    "omega_cdm": float(pardict['omega_cdm']),
    "N_ur": float(pardict["N_ur"]),
    "N_ncdm": int(pardict["N_ncdm"]),
    "m_ncdm": float(pardict["m_ncdm"]),
    "Omega_k": float(pardict["Omega_k"]),
    "tau_reio": float(pardict["tau_reio"]),
     })
    
    if "w" in pardict.keys():
        template.set(
            {"Omega_Lambda": 0.0,
             "w0_fld": float(pardict["w"])
                })
        if "wa" in pardict.keys():
            template.set(
                {"wa_fld": float(pardict["wa"])
                })
    
    template.set({
        "output": "mPk, dTk",
        "P_k_max_1/Mpc": float(pardict["P_k_max_h/Mpc"]),
        "z_max_pk": np.float64(pardict["z_pk"])[0],
    })
    
    template.compute()
    
    h_fid = template.h()
    H_z_fid = template.Hubble(z_data)*conts.c/1000.0
    r_d_fid = template.rs_drag()*h_fid
    DM_fid = template.angular_distance(z_data)*(1.0+z_data)
    
    #This is in h/Mpc. 
    # kmpiv = 0.03
    # kmpiv_test = 0.02
    # kmpiv = np.pi/r_d_fid/h_fid
    # kmpiv = 0.02
    # kmpiv_EH98 = EH98_kmpiv(template)
    # kmpiv_EH98 = 0.036
    kmpiv_CLASS = float(pardict['factor_kp'])
    # kmpiv_CLASS = CLASS_kmpiv(template)
    # kmpiv_EH98 = 0.030
    # kmpiv_CLASS = 0.030
    # print(kmpiv_EH98, kmpiv_CLASS)
    
    # ratio = np.linspace(0.9, 1.0, 11)
    # ratio_flip = ratio**-1
    
    Amp_fid = template.scale_independent_growth_factor_f(z_data)*np.sqrt(template.pk_lin(kmpiv_CLASS*h_fid,z_data)*h_fid**3)
    
    # Amp_fid_1 = template.scale_independent_growth_factor_f(z_data)*np.sqrt(np.array([template.pk_lin(ratio_i*kmpiv*h_fid,z_data) for ratio_i in ratio])*h_fid**3)
    # Amp_fid_2 = template.scale_independent_growth_factor_f(z_data)*np.sqrt(np.array([template.pk_lin(ratio_i*kmpiv*h_fid,z_data) for ratio_i in ratio_flip])*h_fid**3)
    
    if with_w0 == True:
        cosmo_fid = cosmology.cosmology.Cosmology(h = float(pardict['h']), Omega0_b = float(pardict['omega_b'])/float(pardict['h'])**2, 
                                              Omega0_cdm = float(pardict['omega_cdm'])/float(pardict['h'])**2, 
                                              N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
                                              A_s = np.exp(float(pardict["ln10^{10}A_s"]))/1e10, N_ncdm = int(pardict["N_ncdm"]), 
                                              Omega_k = float(pardict["Omega_k"]), tau_reio = float(pardict["tau_reio"]), P_z_max = z_data, Omega0_lambda = 0.0, 
                                              w0_fld = float(pardict['w']))
    elif with_w0_wa == True:
        cosmo_fid = cosmology.cosmology.Cosmology(h = float(pardict['h']), Omega0_b = float(pardict['omega_b'])/float(pardict['h'])**2, 
                                              Omega0_cdm = float(pardict['omega_cdm'])/float(pardict['h'])**2, 
                                              N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
                                              A_s = np.exp(float(pardict["ln10^{10}A_s"]))/1e10, N_ncdm = int(pardict["N_ncdm"]), 
                                              Omega_k = float(pardict["Omega_k"]), tau_reio = float(pardict["tau_reio"]), P_z_max = z_data, Omega0_lambda = 0.0,
                                              w0_fld = float(pardict['w']), wa_fld = float(pardict['wa']))
    else:
        cosmo_fid = cosmology.cosmology.Cosmology(h = float(pardict['h']), Omega0_b = float(pardict['omega_b'])/float(pardict['h'])**2, 
                                          Omega0_cdm = float(pardict['omega_cdm'])/float(pardict['h'])**2, 
                                          N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
                                          A_s = np.exp(float(pardict["ln10^{10}A_s"]))/1e10, N_ncdm = int(pardict["N_ncdm"]), 
                                          Omega_k = float(pardict["Omega_k"]), tau_reio = float(pardict["tau_reio"]), P_z_max = z_data)
    
    # Amp_fid = template.scale_independent_growth_factor_f(z_data)*np.sqrt(cosmology.power.linear.LinearPower(cosmo_fid, np.float64(pardict['z_pk'])[0], 
    #                                                                                                         transfer='NoWiggleEisensteinHu')(kmpiv_CLASS))
    
    # Amp_fid_prime = template.scale_independent_growth_factor_f(z_data)*np.sqrt(EH98(kmpiv, np.float64(pardict["z_pk"])[0], 1.0, template))
    # print(np.sqrt(cosmology.power.linear.LinearPower(cosmo_fid, np.float64(pardict['z_pk'])[0], 
    #                                                                                                         transfer='NoWiggleEisensteinHu')(kvec)))
    # P_pk_fid = Primordial(kvec*h_fid, np.exp(float(pardict["ln10^{10}A_s"]))/1e10, float(pardict["n_s"]))
    # EHpk_fid = EH98(kvec, z_data, 1.0, cosmo=template)*h_fid**3
    
    # pk_class_fid = []
    # for i in range(len(kvec)):
    #     pk_class_fid.append(template.pk_lin(k=kvec[i]*h_fid, z = z_data)*h_fid**3)
    # pk_class_fid = np.array(pk_class_fid)
    
    # np.save('diff.npy', np.abs(pk_class_fid/EHpk_fid -1.0))
    
    # np.save('EHpk.npy', EHpk_fid)
    # np.save('Class_pk.npy', pk_class_fid)
    
    # transfer_fid = EH98_transfer(kvec, z_data, 1.0, cosmo = template)
    # transfer_fid = cosmology.power.linear.LinearPower(cosmo_fid, np.float64(pardict["z_pk"])[0], transfer='NoWiggleEisensteinHu')(kvec)
    transfer_fid = cosmology.power.transfers.NoWiggleEisensteinHu(cosmo_fid, z_data)(kvec)**2
    # transfer_fid_dash = cosmology.power.transfers.NoWiggleEisensteinHu(cosmo_fid, np.float64(pardict["z_pk"])[0])(kvec)
    # transfer_fid_dash = np.array([template.pk_lin(kveci*h_fid, z_data) for kveci in kvec])*(h)**3
    # transfer_fid_dash, k_new_fid = dewiggle(kvec, np.array([template.pk_lin(kveci*h, z_data)*(h)**3 for kveci in kvec]), n=12, return_k=True)
    # transfer_fid_dash = transfer_fid_dash/Primordial(k_new_fid*h, np.exp(float(pardict["ln10^{10}A_s"]))/1e10, float(pardict["n_s"]))
    transfer_fid_dash = transfer_fid
    # print(len(transfer_fid_dash))
    
    # EH98_fid = EH98(kvec, z_data, 1.0, cosmo=template)
    EH98_fid = cosmology.power.transfers.NoWiggleEisensteinHu(cosmo_fid, z_data)(kvec)**2
    if int(pardict["N_ncdm"]) > 0:
        Plin = np.array([template.pk_cb_lin(ki * template.h(), z_data) * template.h() ** 3 for ki in kvec])
    else:
        Plin = np.array([template.pk_lin(ki * template.h(), z_data) * template.h() ** 3 for ki in kvec])
    # Plin = np.array([template.pk_lin(ki * template.h(), z_data) * template.h() ** 3 for ki in kvec])
        
    EH98_prime_fid = smooth_wallisch2018(kvec, Plin)
    EH98_dash_fid = smooth_hinton2017(kvec, Plin)

    
    # Amp_fid_prime = template.scale_independent_growth_factor_f(z_data)*np.sqrt(CubicSpline(kvec, EH98_fid)(kmpiv_CLASS))
    # Amp_fid_prime = template.scale_independent_growth_factor_f(z_data)*np.sqrt(template.pk_lin(kmpiv_CLASS*h_fid, z_data)*h_fid**3)
    
    # Amp_fid_dash = template.scale_independent_growth_factor_f(z_data)*np.sqrt(CubicSpline(k_new_fid, transfer_fid_dash)(kmpiv_CLASS))

    # Amp_fid_dash = template.scale_independent_growth_factor_f(z_data)*np.sqrt(template.pk_lin(kmpiv_CLASS*h_fid, z_data)*h_fid**3)
    
    # transfer_scale_fid = interpolate.interp1d(kvec, transfer_fid, kind='cubic', fill_value='extrapolate')(kvec*h_fid)
    
    fsigma8_fid = template.scale_independent_growth_factor_f(z_data)*template.sigma(8.0/template.h(), z_data)
    sigma8_fid = template.sigma(8.0/template.h(), z_data)
    
    # transfer_class_all = template.get_transfer(z=z_data)
    # k_class = transfer_class_all['k (h/Mpc)']
    # transfer_class_fid = interpolate.interp1d(k_class, transfer_class_all['d_tot'], kind='cubic')(kvec)
    
    # transfer_ratio = transfer_class_fid/transfer_fid
    
    # smooth_fid = smooth(transfer_ratio)
    
    # transfer_smooth_fid = transfer_fid*smooth_fid
    
    # pk_ratio_fid = EHpk_fid/pk_class_fid
    
    # smooth_pk_fid = smooth(pk_ratio_fid)
    
    # pk_smooth_fid = smooth_pk_fid*EHpk_fid
    
    # transfer_ratio_smooth = transfer_ratio[0] + 2 * np.c_[np.r_[0, smooth_all[1:-1:2].cumsum()], smooth_all[::2].cumsum() - smooth_all[0] / 2].ravel()[:len(smooth_all)]
    
    Pk_ratio_fid = transfer_fid
    # print(Amp_fid, fsigma8_fid)
    print(fsigma8_fid)
    # print(cosmology.power.linear.LinearPower(cosmo_fid, np.float64(pardict["z_pk"])[0], transfer='NoWiggleEisensteinHu')._sigma8/np.sqrt(cosmology.power.linear.LinearPower(cosmo_fid, np.float64(pardict["z_pk"])[0], transfer='NoWiggleEisensteinHu')._norm))
    
    # np.save('test_AP.npy', [kvec, Plin/EH98_dash_fid])
    # print(cosmology.power.linear.LinearPower(cosmo_fid, np.float64(pardict["z_pk"])[0], transfer='NoWiggleEisensteinHu')._norm, cosmo_fid.sigma8)

    # np.save('test.npy', [transfer_fid, EH98_fid])
    # print(Amp_fid_prime, Amp_fid, Amp_fid_dash, np.pi/r_d_fid/h_fid)
    
    # np.save('dew_plin.npy', [kvec, transfer_fid_dash])
    
    # birdmodel_fid = BirdModel(pardict, template=True, direct=True, fittingdata=fittingdata, window = str(fittingdata.data["windows"]), Shapefit=True)
    
    # P11l = birdmodel_fid.correlator.bird.P11l
    # P11l_fid = interpolate.interp1d(birdmodel_fid.correlator.co.k, P11l, kind='cubic', axis=-1, fill_value='extrapolate')(kvec)
    # P11l_fid = P11l_fid.reshape((P11l_fid.shape[0]*P11l_fid.shape[1], P11l_fid.shape[2]))
    
    # Pctl = birdmodel_fid.correlator.bird.Pctl
    # Pctl_fid = interpolate.interp1d(birdmodel_fid.correlator.co.k, Pctl, kind='cubic', axis=-1, fill_value='extrapolate')(kvec)
    # Pctl_fid = Pctl_fid.reshape((Pctl_fid.shape[0]*Pctl_fid.shape[1], Pctl_fid.shape[2]))
    
    # Ploopl = birdmodel_fid.correlator.bird.Ploopl
    # Ploopl_fid = interpolate.interp1d(birdmodel_fid.correlator.co.k, Ploopl, kind='cubic', axis=-1, fill_value='extrapolate')(kvec)
    # Ploopl_fid = Ploopl_fid.reshape((Ploopl_fid.shape[0]*Ploopl_fid.shape[1], Ploopl_fid.shape[2]))
    
    # skip_lin = np.array([3])
    # skip_ct = np.array([6])
    # skip_loop = np.array([33, 36, 38, 39, 41, 42, 43])
    
    # np.save('P11l.npy', P11l_fid)
    # np.save('Pctl.npy', Pctl_fid)
    # np.save('Ploopl.npy', Ploopl_fid)
    
    # all_cosmo_params = []
    # for i in range(6561):
    #     index = i

    #     index_As, remainder = divmod(index, np.int32((2*order+1)**(nparams - 1)))
    #     index_h, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-2)))
    #     index_cdm, index_b = divmod(remainder, np.int32((2*order+1)**(nparams-3)))

    #     params = np.array([ln_10_10_As_all[index_As], h_all[index_h], omega_cdm_all[index_cdm], omega_b_all[index_b]])
    #     all_cosmo_params.append(params)
    # all_cosmo_params = np.array(all_cosmo_params)
    
    # data = []
    # for i in range(81):
    #     index = 40 + 81*i
    #     index_As, remainder = divmod(index, np.int32((2*order+1)**(nparams - 1)))
    #     index_h, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-2)))
    #     index_cdm, index_b = divmod(remainder, np.int32((2*order+1)**(nparams-3)))
        
    #     params = np.array([ln_10_10_As_all[index_As], h_all[index_h], omega_cdm_all[index_cdm], omega_b_all[index_b]])
        
    #     print(params)
        
    #     output = grid_params(pardict, params)
    #     data.append(output)
    # np.save('output.npy', data)
    
    # np.save('test.npy', [Pk_ratio_fid, EH98_dash_fid, EH98_prime_fid])
    
    all_params = []
    
    for i in range(length):
        index = start + i
    
        index_As, remainder = divmod(index, np.int32((2*order+1)**(nparams - 1)))
        index_h, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-2)))
        
        if with_w0 == True:
            index_cdm, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-3)))
            index_b, index_w = divmod(remainder, np.int32((2*order+1)**(nparams-4)))
            params = np.array([ln_10_10_As_all[index_As], h_all[index_h], omega_cdm_all[index_cdm], omega_b_all[index_b], w_all[index_w]])
        elif with_w0_wa == True:
            index_cdm, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-3)))
            index_b, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-4)))
            index_w, index_wa = divmod(remainder, np.int32((2*order+1)**(nparams-5)))
            params = np.array([ln_10_10_As_all[index_As], h_all[index_h], omega_cdm_all[index_cdm], omega_b_all[index_b], w_all[index_w], wa_all[index_wa]])
        elif with_omegak == True:
            index_cdm, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-3)))
            index_b, index_omegak = divmod(remainder, np.int32((2*order+1)**(nparams-4)))
            params = np.array([ln_10_10_As_all[index_As], h_all[index_h], omega_cdm_all[index_cdm], omega_b_all[index_b], omegak_all[index_omegak]])
        else:
            index_cdm, index_b = divmod(remainder, np.int32((2*order+1)**(nparams-3)))
            params = np.array([ln_10_10_As_all[index_As], h_all[index_h], omega_cdm_all[index_cdm], omega_b_all[index_b]])
        
        # print(params)
        
        output = grid_params(pardict, params)
        
        all_params.append(output)

    all_params = np.array(all_params) 
    
    if with_w0 == True or with_w0_wa == True or with_omegak == True:
        np.save("Shapefit_Grid_" + str(job_num) + "_" + str(job_total_num) + keyword + 'bin_' + str(pardict['red_index']) + ".npy", all_params)
    else:
        np.save("Shapefit_Grid_" + str(job_num) + "_" + str(job_total_num) + 'bin_' + str(pardict['red_index']) + ".npy", all_params)