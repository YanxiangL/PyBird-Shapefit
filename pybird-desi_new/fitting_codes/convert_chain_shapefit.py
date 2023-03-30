# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:48:25 2022

@author: s4479813
"""

import numpy as np
import sys
from configobj import ConfigObj
from multiprocessing import Pool
# from classy import Class
import scipy.constants as conts
import time
import scipy as sp
from scipy import interpolate, linalg
from findiff import FinDiff

sys.path.append('../')
from pybird_dev.greenfunction import GreenFunction

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

# def read_chain_backend(chainfile):

#     import copy
#     import emcee

#     reader = emcee.backends.HDFBackend(chainfile)

#     tau = reader.get_autocorr_time()
#     thin = int(0.5 * np.min(tau))
#     # tau = 400.0
#     burnin = int(2 * np.max(tau))
#     samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
#     log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
#     bestid = np.argmax(log_prob_samples)

#     return samples, copy.copy(samples[bestid]), log_prob_samples

def read_chain_backend(chainfile):
    import copy
    import emcee
    
    #Read the MCMC chain
    reader = emcee.backends.HDFBackend(chainfile)
    
    #Find the autocorrelation time. 
    tau = reader.get_autocorr_time()
    #Using the autocorrelation time to figure out the burn-in. 
    burnin = int(2 * np.max(tau))
    #Retriving the chain and discard the burn-in, 
    samples = reader.get_chain(discard=burnin, flat=True)
    #The the log-posterior of the chain. 
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    #Find the best-fit parameters. 
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples


def read_chain(pardict, Shapefit, z_bin, with_zeus = True):

    # Reads in a chain containing template fit or Shapefit parameters, and returns the mean data vector and
    # inverse covariance matrix for fitting with cosmological parameters. Assumes the data
    # are suitably Gaussian.
    
    if resum == True:
        marg_str = "marg" if pardict["do_marg"] else "all"
        hex_str = "hex" if pardict["do_hex"] else "nohex"
        dat_str = "xi" if pardict["do_corr"] else "pk"
        if Shapefit:
            fmt_str = (
                "%s_%s_%2dhex%2d_%s_%s_%s_Shapefit_mock_%s_bin_"+str(z_bin)
                if pardict["do_corr"]
                else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_Shapefit_mock_%s_bin_"+str(z_bin)
            )
        else:
            fmt_str = (
                "%s_%s_%2dhex%2d_%s_%s_%s_Templatefit_mock_%s_bin_"+str(z_bin)
                if pardict["do_corr"]
                else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_Templatefit_mock_%s_bin_"+str(z_bin)
            )
        fitlim = pardict["xfit_min"][0] if pardict["do_corr"] else pardict["xfit_max"][0]
        fitlimhex = pardict["xfit_min"][2] if pardict["do_corr"] else pardict["xfit_max"][2]
        
        if with_zeus:
            fmt_str = fmt_str + '.npy'
        else:
            fmt_str = fmt_str + '.hdf5'
    
        taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
        chainfile = str(
            fmt_str
            % (
                pardict["fitfile"],
                dat_str,
                fitlim,
                fitlimhex,
                taylor_strs[pardict["taylor_order"]],
                hex_str,
                marg_str,
                keyword
            )
        )
    else:
        chainfile = '../../data/DESI_KP4_LRG_ELG_QSO_pk_0.20hex0.20_3order_nohex_marg_noresum_bin_' + str(z_bin) + '.hdf5'
    #This is the name of the chainfile. 
    print(chainfile)
    #The next line returns the chains, best-fit parameters and the likelihood for each iteration. 
    if with_zeus == False:
        burntin, bestfit, like = read_chain_backend(chainfile)
    else:
        # burntin = np.load(chainfile).T
        burntin = np.loadtxt('../../data/shapefit_chains_test.txt')
        # print(burntin)
        
        burntin = np.array([burntin[:, 2], burntin[:, 1], burntin[:, 0], burntin[:, 3]]).T
    
    # index = np.where(like > 0.0)[0]
    
    # burntin = np.delete(burntin, index, axis = 0)
    # like = np.delete(like, index, axis = 0)
    # bestfit = burntin[np.argmax(like)]
    
    #Shapefit has 4 compressed parameters while templete fit has 3. 
    # burntin = burntin[:, :4] if Shapefit else burntin[:, :3]
    #Compute the inverse covariance matrix
    cov_inv = np.linalg.inv(np.cov(burntin, rowvar=False))
    
    # print(np.shape(cov_inv))
    
    # n = len(burntin)
    # p = len(cov_inv)
    
    # cov_inv = (n-p-2.0)/(n-1.0)*cov_inv
    # if Shapefit is True:
    #     cov_inv_out = cov_inv[:4, :4]
    # else:
    #     cov_inv_out = cov_inv[:3, :3]

    # return np.mean(burntin, axis=0), cov_inv_out
    
    return np.mean(burntin, axis = 0), cov_inv
    # return bestfit[:4], cov_inv_out


def do_emcee(func, start):
    
    #This is the code to run MCMC. 

    import emcee

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    #Set the number of walkers. 
    nwalkers = nparams * 8
    
    #Set up some random starting position. 
    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    if Shapefit:
        fmt_str = (
            "%s_%s_%2dhex%2d_%s_%s_%s_Shapefit_planck"
            if pardict["do_corr"]
            else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_Shapefit_planck"
        )
    else:
        fmt_str = (
            "%s_%s_%2dhex%2d_%s_%s_%s_template_planck"
            if pardict["do_corr"]
            else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_template_planck"
        )
    fitlim = pardict["xfit_min"][0] if pardict["do_corr"] else pardict["xfit_max"][0]
    fitlimhex = pardict["xfit_min"][2] if pardict["do_corr"] else pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            pardict["fitfile"],
            dat_str,
            fitlim,
            fitlimhex,
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
    )
    
    #Construct the chainfile. 
    chainfile = chainfile + "_mock_" + keyword
    if FullShape is True:
        chainfile = chainfile + "FullShape"
    oldfile = chainfile + ".hdf5"
    newfile = chainfile + ".dat"
    print(oldfile)
    
    from multiprocessing import cpu_count

    ncpu = cpu_count()
    print("{0} CPUs".format(ncpu))

    # Set up the backend
    backend = emcee.backends.HDFBackend(oldfile)
    backend.reset(nwalkers, nparams)

    with Pool() as pool:

        # Initialize the sampler
        #We use multiprocessing here to speed up the MCMC.
        sampler = emcee.EnsembleSampler(nwalkers, nparams, func, pool=pool, backend=backend)

        # Run the sampler for a max of 30000 iterations. We check convergence every 100 steps and stop if
        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
        max_iter = 30000
        index = 0
        old_tau = np.inf
        autocorr = np.empty(max_iter)
        counter = 0
        for sample in sampler.sample(begin, iterations=max_iter, progress=True):

            # Only check convergence every 1000 steps
            if sampler.iteration % 1000:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.max(tau)
            counter += 1000
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print("Max Auto-Correlation time: {0:.3f}".format(autocorr[index]))

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            
            # converged = np.all(tau * 50 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
                break
            old_tau = tau
            index += 1

    # burntin, bestfit, like = read_chain_backend(oldfile)

    # lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    # upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    # # Loop over the parameters in the chain and use the grids to compute the derived parameters
    # chainvals = []
    # for i, (vals, loglike) in enumerate(zip(burntin, like)):
    #     if i % 1000 == 0:
    #         print(i)
    #     ln10As, h, omega_cdm, omega_b = vals[:4]
    #     # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm
    #     if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
    #         np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
    #     ):
    #         continue
    #     Om, Da, Hz, f, sigma8, sigma8_0, sigma12, r_d = birdmodel.compute_params([ln10As, h, omega_cdm, omega_b])
    #     alpha_perp = (Da / h) * (float(pardict["h"]) / Da_fid) * (r_d_fid / (r_d))
    #     alpha_par = (float(pardict["h"]) * Hz_fid) / (h * Hz) * (r_d_fid / (r_d))
    #     chainvals.append(
    #         (
    #             ln10As,
    #             100.0 * h,
    #             omega_cdm,
    #             omega_b,
    #             alpha_perp,
    #             alpha_par,
    #             Om,
    #             2997.92458 * Da / h,
    #             100.0 * h * Hz,
    #             f,
    #             sigma8,
    #             sigma8_0,
    #             sigma12,
    #             loglike,
    #         )
    #     )

    # np.savetxt(newfile, np.array(chainvals))


def lnpost(params):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params)
    return prior + like


def lnprior(params):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors

    ln10As, h, omega_cdm, omega_b = params[:4]
    # ln10As, h, Omega_m = params[:3]
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    valueref = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b']]))
    delta = np.float64(pardict['dx'])
    # lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    # upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta
    
    lower_bounds = valueref - np.float64(pardict["order"]) * delta
    upper_bounds = valueref + np.float64(pardict["order"]) * delta

    # Flat priors for cosmological parameters
    if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
        np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
    ):
        return -np.inf
    
    # if np.any(np.less([ln10As, h, Omega_m], np.array([lower_bounds[0], lower_bounds[1], (lower_bounds[2] + lower_bounds[3])/upper_bounds[1]**2]))) or np.any(
    #     np.greater([ln10As, h, Omega_m], np.array([upper_bounds[0], upper_bounds[1], (upper_bounds[2] + upper_bounds[3])/lower_bounds[1]**2]))
    # ):
    #     return -np.inf
    
    # if np.any(np.less([ln10As, h, omega_cdm, omega_b], [1.5, 0.2, 0.0, 0.0])) or np.any(np.greater([ln10As, h, omega_cdm, omega_b], 
    #    [4.5, 1.0, 1.0, 1.0])):
    #     return -np.inf

    # BBN (D/H) inspired prior on omega_b
    # omega_b_prior = -0.5 * (omega_b - birdmodel.valueref[3]) ** 2 / 0.00037 ** 2
    omega_b_prior = -0.5 * (omega_b - float(pardict['omega_b'])) ** 2 / 0.00037 ** 2
    # omega_b_prior = 0.0
    
    b1, bp, bd = params[4:]
    
    if b1 < 0.0 or b1 > 4.0:
        return -np.inf
    
    if bp < 0.0 or bp > 5.0:
        return -np.inf
    
    if bd < -20.0 or bd > 20.0:
        return -np.inf

    return omega_b_prior
    # return 0.0


# def lnlike(params):

#     ln10As, h, omega_cdm, omega_b = params[:4]
#     # ln10As, h, omega_cdm = params[:3]
#     # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

#     Om, Da, Hz, f, sigma8, sigma8_0, sigma12, r_d = birdmodel.compute_params([ln10As, h, omega_cdm, omega_b])

#     # Factors of little h required as Da and Hz in Grid.py have little h scaled out.
#     alpha_perp = (Da / h) * (float(pardict["h"]) / Da_fid) * (r_d_fid / (r_d))
#     alpha_par = (float(pardict["h"]) * Hz_fid) / (h * Hz) * (r_d_fid / (r_d))

#     model = np.array([alpha_perp, alpha_par, f * sigma8])
#     if hybrid:
#         model = np.concatenate([model, [omega_cdm + omega_b + birdmodel.omega_nu]])

#     chi_squared = (fitdata - model) @ fitcov @ (fitdata - model)

#     return -0.5 * chi_squared

# def lnlike(params):
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
#         }
#     )
#     M.set(
#         {
#             "output": "mPk",
#             "P_k_max_1/Mpc": float(pardict["P_k_max_h/Mpc"]),
#             "z_max_pk": z_data,
#         }
#     )
#     M.compute()
    
#     DM_at_z = M.angular_distance(z_data) * (1. + z_data)
#     H_at_z = M.Hubble(z_data) * conts.c / 1000.0
#     rd = M.rs_drag()
    
#     theo_fAmp = M.scale_independent_growth_factor_f(z_data)*np.sqrt(M.pk_lin(kmpiv*h_fid*h_fid*r_d_fid/(rd*h),z_data)*(h_fid*r_d_fid/(rd*h))**3.)/Amp_fid
    
#     theo_aperp = DM_at_z / DM_fid / rd * r_d_fid
#     theo_apara = H_z_fid/ H_at_z / rd * r_d_fid
    
#     EHpk = EH98(M, kvec*h_fid*r_d_fid/(rd*h), z_data ,1.0)
#     P_pk = Primordial(kvec*h_fid, A_s, float(pardict["n_s"]))
    
#     Pk_ratio = EHpk/P_pk
    
#     Pkshape_ratio_prime = slope_at_x(np.log(kvec),np.log(Pk_ratio/Pk_ratio_fid))
#     theo_mslope = np.interp(np.log(kmpiv), np.log(kvec), Pkshape_ratio_prime)
    
#     diff = np.array([theo_aperp - fitdata[0], theo_apara - fitdata[1], (theo_fAmp - 1.0)*fitdata[2], theo_mslope - fitdata[3]])
    
#     chi_squared = diff @ fitcov @ diff
    
#     return chi_squared

# def lnlike(params):
    
#     ln10As, h, omega_cdm, omega_b = params[:4]
    
#     if Shapefit is True:
    
#         # A_s = np.exp(ln10As)/1.0e10
        
#         # theo_aperp, theo_apara, theo_fAmp, theo_mslope = interpolation_function(params)[0]
#         theo_aperp, theo_apara, theo_fAmp, theo_mslope, fsigma8, theo_fAmp_new, mslope_dash, mslope_new = interpolation_function(params)[0]
        
#         # #Convert the angular diameter distance in Mpc/h. 
#         # theo_aperp = (DM_at_z) / DM_fid / rd * r_d_fid
#         # theo_apara = H_z_fid/ (H_at_z) / rd * r_d_fid
        
#         # # EHpk = EH98(kvec*h_fid*r_d_fid/(rd*h), z_data ,1.0, cosmo = M)
#         # EHpk = EH98(kvec*h_fid*r_d_fid/(rd*h), z_data ,1.0, rd = rd, z_d = z_d, Omm = Omm, zeq = zeq, keq = keq, Hz = H_at_z, 
#         #          h=h, Omb = omega_b, Omc = omega_cdm, n_s = float(pardict["n_s"]), growth = growth, Hubble=Hubble)
#         # #Add in the primordial power spectrum calculation. 
#         # P_pk = Primordial(kvec*h_fid, A_s, float(pardict["n_s"]))
        
#         # Pk_ratio = EHpk/P_pk
        
#         # Pkshape_ratio_prime = slope_at_x(np.log(kvec),np.log(Pk_ratio/Pk_ratio_fid))
#         # # Pkshape_ratio_prime = slope_at_x(np.log(kvec),np.log(EHpk/EHpk_fid))
#         # theo_mslope = np.interp(kmpiv, kvec, Pkshape_ratio_prime)
        
#         diff = np.array([theo_aperp - fitdata[0], theo_apara - fitdata[1], theo_fAmp_new*fsigma8_fid - fitdata[2], mslope_dash - fitdata[3]])
#         # diff = np.array([theo_aperp - fitdata[0], theo_apara - fitdata[1], theo_fAmp*fsigma8_fid - fitdata[2], mslope - fitdata[3]])

        
#         chi_squared = diff @ fitcov @ diff
#     else:
#             if FullShape is False:
#                 theo_aperp, theo_apara, theo_fAmp, theo_mslope, fsigma8, theo_fAmp_new, mslope_dash, mslope_new = interpolation_function(params)[0]
    
#                 # # Factors of little h required as Da and Hz in Grid.py have little h scaled out.
#                 # alpha_perp = (Da ) * (1.0 / Da_fid) * (r_d_fid / (r_d))
#                 # alpha_par = (Hz_fid) / (Hz) * (r_d_fid / (r_d))
                
#                 # # print(alpha_perp, alpha_par, f*sigma8)
    
#                 # diff = np.array([theo_aperp - fitdata[0], theo_apara - fitdata[1], theo_fAmp_new*fsigma8_fid - fitdata[2]])
#                 diff = np.array([theo_aperp - fitdata[0], theo_apara - fitdata[1], fsigma8 - fitdata[2]])
#             else:
#                 Om, Da, Hz, Dn, f, sigma8, sigma8_0, sigma12, r_d = interpolation_function2(params)[0]
#                 theo_aperp, theo_apara, theo_fAmp, theo_mslope = interpolation_function(params)[0]
#             # factor = float(pardict['h'])/h
#                 diff = np.array([theo_aperp - fitdata[0], theo_apara - fitdata[1], f*sigma8 - fitdata[2]])

#             chi_squared = diff @ fitcov @ diff    
#     return -0.5*chi_squared

# def lnlike(params):
#     #This is the likelihood function
    
#     #Read in the cosmological parameters. 
#     ln10As, h, omega_cdm, omega_b = params[:4]
    
#     # ln10As, h, Omega_m = params[:3]
    
#     A_s = np.exp(ln10As)/1.0e10
#     H_0 = 100.0*h
    
#     #Set up Class. 
#     M = Class()
    
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
#         }
#     )
#     # M.set(
#     #     {
#     #         "A_s": A_s,
#     #         "n_s": float(pardict["n_s"]),
#     #         "H0": H_0,
#     #         "Omega_m": Omega_m,
#     #         "N_ur": float(pardict["N_ur"]),
#     #         "N_ncdm": int(pardict["N_ncdm"]),
#     #         "m_ncdm": pardict["m_ncdm"],
#     #         "Omega_k": float(pardict["Omega_k"]),
#     #     }
#     # )
    
#     M.set(
#         {
#             "output": "mPk",
#             "P_k_max_1/Mpc": float(pardict["P_k_max_h/Mpc"]),
#             "z_max_pk": np.max(z_data_all) + 0.5,
#         }
#     )
#     M.compute()
    
#     model_all = []
#     for i in range(nz):
        
#         #The fiducial values. 
#         h_fid, H_z_fid, r_d_fid, DM_fid, fAmp_fid, transfer_fid, fsigma8_fid = fiducial_all[i]
#         #The redshift. 
#         z_data = float(pardict["z_pk"][i])
        
#         #Compute DM, H(z) and rd with the new cosmology.
#         DM_at_z = M.angular_distance(z_data) * (1. + z_data)
#         H_at_z = M.Hubble(z_data) * conts.c / 1000.0
#         rd = M.rs_drag()
        
#         #Compute the ratio of fAmp in the Shapefit paper in order to find the corresponding fsigma8 parameter. 
#         theo_fAmp = M.scale_independent_growth_factor_f(z_data)*np.sqrt(M.pk_lin(kmpiv*h_fid*r_d_fid/(rd),z_data)*(h*r_d_fid/(rd))**3.)/fAmp_fid
        
#         theo_aperp = (DM_at_z) / DM_fid / rd * r_d_fid
#         theo_apara = H_z_fid/ (H_at_z) / rd * r_d_fid
    
#         if Shapefit is True:
            
#             # d_dk = FinDiff(0, np.log(kvec), 1, acc=10)
            
#             #Compute the transfer function with the EH98 formula. 
#             transfer_new = EH98_transfer(kvec*h_fid*r_d_fid/(rd*h), z_data ,1.0, cosmo=M)*(r_d_fid/rd)**3
            
#             # ratio_transfer = d_dk(np.log(transfer_new/transfer_fid))
            
#             #Find the slope at the pivot scale.  
#             ratio_transfer = slope_at_x(np.log(kvec), np.log(transfer_new/transfer_fid))
#             theo_mslope = interpolate.interp1d(kvec, ratio_transfer, kind='cubic')(kmpiv)
            
#             #Find the model a_perp, a_par fsigma8 and m.             
#             model_all.append([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope])
#         else:
#                 #This is for the template fit. 
#                 if FullShape is False:
        
#                     model_all.append([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid])
#                 #Ignore this. 
#                 else:
#                     Om, Da, Hz, Dn, f, sigma8, sigma8_0, sigma12, r_d = interpolation_function2(params)[0]
#                     theo_aperp, theo_apara, theo_fAmp, theo_mslope = interpolation_function(params)[0]
                    
#     model = np.concatenate(model_all)
    
#     #Calculate the difference between the model and "data" (from shapefit) compressed parameter and then calculate chi-squared. 
#     chi_squared = np.dot(np.dot(model-data_all, cov_all), model-data_all)
    
#     if np.random.rand() < 0.01:
#         print(params[:4], chi_squared)
      
#     return -0.5*chi_squared

def lnlike(params):
    theo_aperp, theo_apara, theo_fAmp, theo_mslope, fsigma8, theo_mslope_dash, theo_fAmp_dash, theo_fAmp_prime, theo_mslope_prime = interpolation_function(params[:4])[0]
    
    # theo_fAmp = theo_fAmp_dash
    theo_fAmp = theo_fAmp_prime
    
    # theo_mslope = theo_mslope_dash
    
    theo_mslope = theo_mslope_prime
    
    
    # GF = GreenFunction(Omega0_m = (params[2] + params[3])/params[2]**2)
    
    # f = GF.fplus(1.0/(1.0 + np.int16(pardict['z_pk'])))
    
    # sigma8 = 0.5369920683243679
    
    # fsigma8_new = f*sigma8
    
    b1, bp, bd = params[4:]
    
    model = np.array([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope, b1, bp, bd])
    
    # model = np.array([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope])
    
    chi_squared = np.dot(np.dot(model-data_all, cov_all), model-data_all)
    if np.random.rand() < 0.001:
        print(params[:4], chi_squared)
        
    return -0.5*chi_squared

def EH98(kvector, redshift, scaling_factor, cosmo=None):
    #This is the link to the paper: https://arxiv.org/pdf/astro-ph/9710252.pdf
    #The input kvector should be in unit of h/Mpc after rescaling by rd. 
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
        
    if (fnu==0):
        Nnu = 0.
    else:
        Nnu = 1.
    #alpha_gamma = 1 - 0.328*np.log(431*Omm*h**2)*Omb/Omm + 0.38*np.log(22.3*Omm*h**2)*(Omb/Omm)**2
    
    #There seems to be a mistake in this equation. 
    # alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu)*Nnu**0.2 + 0.169*fnu) \
    #             *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
    
    alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu*Nnu) + 0.169*fnu*(Nnu)**0.2) \
                *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
                
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
    else:
        yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6.))*(Nnu*q_test/fnu)**2.
        qnu = 3.92*q_test*np.sqrt(Nnu/fnu)
    D1 = (1.+zeq)/(1.+redshift)*5*Omm_at_z/2./(Omm_at_z**(4./7.) - OmLambda_at_z + (1.+Omm_at_z/2.)*(1.+OmLambda_at_z/70.))
    Dcbnu = (fcb**(0.7/pcb)+(D1/(1.+yfs))**0.7)**(pcb/0.7) * D1**(1.-pcb)
    Bk = 1. + 1.24*fnu**(0.64)*Nnu**(0.3+0.6*fnu)/(qnu**(-1.6)+qnu**0.8)
    #
    Tcbnu = T0*Dcbnu/D1*Bk
    deltah = 1.94e-5 * Omm**(-0.785-0.05*np.log(Omm))*np.exp(-0.95*(ns-1)-0.169*(ns-1)**2.)
    
    #The output power spectrum will be in the unit of (Mpc/h)^3. 
    Pk = 2*np.pi**2. * deltah**2. * (kvector)**(ns) * Tcbnu**2. * growth**2. /cosmo.Hubble(0)**(3.+ns)
    return Pk
    
def EH98_transfer(kvector, redshift, scaling_factor, cosmo=None):
    
    #This is the link to the paper: https://arxiv.org/pdf/astro-ph/9710252.pdf
    #The input kvector should be in unit of h/Mpc after rescaling by rd. 
    
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
    #find the slope
    diff = np.diff(yvector)/np.diff(xvector)
    diff = np.append(diff,diff[-1])
    return diff
    


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    Shapefit = bool(int(sys.argv[2]))
    FullShape = bool(int(sys.argv[3]))
    resum = bool(int(sys.argv[4]))
    job_total_num = int(sys.argv[5])
    
    try:
        #Do single mock. 
        mock_num = int(sys.argv[6])
        mean = False
    except:
        #Do mean of the mock. 
        mean = True
    
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    
    if resum == False:
        keyword = '_noresum'
    
    #Read in the mock files. 
    if Shapefit == True:
        if mean == False:
            keyword = str(mock_num)
            
            datafiles = np.loadtxt(pardict['datafile'], dtype=str)
            mockfile = str(datafiles) + str(mock_num) + '.dat'
            newfile = '../config/data_mock_' + str(mock_num) + '.txt'
            text_file = open(newfile, "w")
            n = text_file.write(mockfile)
            text_file.close()
            pardict['datafile'] = newfile
        else:
            keyword = str("mean")
            pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
            pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
    else:
        if mean == False:
            keyword = str(mock_num)
            
            datafiles = np.loadtxt(pardict['datafile'], dtype=str)
            mockfile = str(datafiles) + str(mock_num) + '.dat'
            newfile = '../config/data_mock_' + str(mock_num) + '.txt'
            text_file = open(newfile, "w")
            n = text_file.write(mockfile)
            text_file.close()
            pardict['datafile'] = newfile
        else:
            keyword = str("mean")
            pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
            pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
            
    
    # fittingdata = FittingData(pardict)

    # Set up the BirdModel
    if Shapefit:
        print("Using Shapefit")
        
    else:
        print("Using template fit")
        
        # grid_all = np.load('TableParams_class-redbin0_flat_BOSS.npy')
        # # order = float(pardict["order"])
        # order = 4
        # delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
        # valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
        # truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
        # # shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
        # shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(4) + 1)])
        # grid_all = grid_all.reshape((*shapecrd[1:], np.shape(grid_all)[1]))
        # interpolation_function2 = sp.interpolate.RegularGridInterpolator(truecrd, grid_all, bounds_error=False)
        
        # test = lnlike(np.array([3.01, 0.67, 0.12, 0.023]))
    
    #This stores the input parameters. 
    # birdmodel = BirdModel(pardict, template=True, direct=True, fittingdata=fittingdata, window = None, Shapefit=Shapefit)
    
    grid_all = []
    for i in range(job_total_num):
        grid = np.load("Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + ".npy")
        grid_all.append(grid)
    
    grid_all = np.vstack(grid_all)
    order = float(pardict["order"])
    delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
    valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
    shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
    grid_all = grid_all.reshape((*shapecrd[1:], np.shape(grid_all)[1]))
    interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
    # interpolation_function2 = sp.interpolate.RegularGridInterpolator(grid_all, truecrd)
    
    z_data_all = np.float64(pardict["z_pk"])
    nz = len(z_data_all)
    
    # #This is in h/Mpc. 
    # kmpiv = 0.03
    
    # kvec = np.logspace(-2.0, 0.0, 300)
    
    # fiducial_all = []
    
    fit_data_all = []
    
    fit_cov_all = []
    
    fsigma8_fid = 0.450144
    
    # #Compute the compressed parameters based on the template (true) cosmology. 
    # template = Class()
    
    # template.set({
    # "A_s": np.exp(float(pardict["ln10^{10}A_s"]))/1e10,
    # "n_s": float(pardict["n_s"]),
    # "H0": 100.0*float(pardict['h']),
    # "omega_b": float(pardict['omega_b']),
    # "omega_cdm": float(pardict['omega_cdm']),
    # "N_ur": float(pardict["N_ur"]),
    # "N_ncdm": int(pardict["N_ncdm"]),
    # "m_ncdm": float(pardict["m_ncdm"]),
    # "Omega_k": float(pardict["Omega_k"]),
    #  })
    
    # template.set({
    #     "output": "mPk",
    #     "P_k_max_1/Mpc": float(pardict["P_k_max_h/Mpc"]),
    #     "z_max_pk": np.max(z_data_all) + 0.5,
    # })
    
    # template.compute()
    
    for i in range(nz):
    
    #     z_data = z_data_all[i]
        
    #     # h_fid = template.h()
    #     # H_z_fid = template.Hubble(z_data)*conts.c/1000.0
    #     # r_d_fid = template.rs_drag()
    #     # DM_fid = template.angular_distance(z_data)*(1.0+z_data)
        
    #     # Amp_fid = template.scale_independent_growth_factor_f(z_data)*np.sqrt(template.pk_lin(kmpiv*h_fid,z_data)*h_fid**3)
        
    #     #Find fiducial a_perp, a_par, fsigma8 and m. 
    #     h_fid = template.h()
    #     H_z_fid = template.Hubble(z_data)*conts.c/1000.0
    #     r_d_fid = template.rs_drag()
    #     DM_fid = template.angular_distance(z_data)*(1.0+z_data)
        
    #     fAmp_fid = template.scale_independent_growth_factor_f(z_data)*np.sqrt(template.pk_lin(kmpiv*h_fid,z_data)*h_fid**3)
        
    #     # P_pk_fid = Primordial(kvec*h_fid, np.exp(float(pardict["ln10^{10}A_s"]))/1e10, float(pardict["n_s"]))
    #     EHpk_fid = EH98(kvec, z_data, 1.0, cosmo=template)*h_fid**3
        
    #     transfer_fid = EH98_transfer(kvec, z_data, 1.0, cosmo = template)
        
    #     # f = M.scale_independent_growth_factor_f(z_data)
    #     # sigma8 = M.sigma(8.0 / M.h(), z_data)
        
    #     fsigma8_fid = template.scale_independent_growth_factor_f(z_data)*template.sigma(8.0/template.h(), z_data)
        
    #     Pk_ratio_fid = EHpk_fid
        
    #     print(z_data, fsigma8_fid)
        
    #     fiducial_all.append([h_fid, H_z_fid, r_d_fid, DM_fid, fAmp_fid, transfer_fid, fsigma8_fid])

        # Read in the chain and get the template parameters
        if FullShape == True:
            print("Using FullShape fit")
            useful = np.load("FullShape_template.npy")
            fitdata = np.mean(useful, axis=0)
            fitcov = np.linalg.inv(np.cov(useful, rowvar=True))
        else:
            # fitdata, fitcov = read_chain(pardict, Shapefit, i, with_zeus=True)
            fitdata, fitcov = read_chain(pardict, Shapefit, i, with_zeus=False)
            
        fit_data_all.append(fitdata)
        fit_cov_all.append(fitcov)
    
    cov_all = sp.linalg.block_diag(*fit_cov_all)
    data_all = np.concatenate(fit_data_all)
    # np.save("fit_data.npy", fitdata)
    # np.save("fit_cov.npy", fitcov)
    # np.save("Pk_ratio_fid.npy", Pk_ratio_fid)
    print("The mean posterior is ", data_all)

    #We start MCMC with the True cosmological parameters. 
    # start = np.array([birdmodel.valueref[0], birdmodel.valueref[1], birdmodel.valueref[2], birdmodel.valueref[3]])
    # start = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b']]))
    start = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], 2.0, 1.0, 1.0]))
    # start = np.array([birdmodel.valueref[0], birdmodel.valueref[1], (birdmodel.valueref[2] + birdmodel.valueref[3])/birdmodel.valueref[1]**2])
    
    # print(cov_all, np.shape(cov_all))
    
    # print(template.Neff())

    # Does an optimization
    # result = do_optimization(lambda *args: -lnpost(*args), start)

    # Does an MCMC and then post-processes to get some derived params
    do_emcee(lnpost, start)
    
    # nparams = len(pardict["dx"])
    
    # order = int(pardict['order'])
    # #The step size of each parameter dx must be written in the order of ln_10^10_As, h, omega_cdm, omega_b
    # dx = np.float64(pardict['dx'])
    # ln_10_10_As = float(pardict["ln10^{10}A_s"])
    # h = float(pardict['h'])
    # omega_cdm = float(pardict['omega_cdm'])
    # omega_b = float(pardict['omega_b'])
    # ln_10_10_As_all = np.linspace(ln_10_10_As - order*dx[0], ln_10_10_As + order*dx[0], 2*order+1)
    # h_all = np.linspace(h - order*dx[1], h + order*dx[1], 2*order+1)
    # omega_cdm_all = np.linspace(omega_cdm - order*dx[2], omega_cdm + order*dx[2], 2*order+1)
    # omega_b_all = np.linspace(omega_b - order*dx[3], omega_b + order*dx[3], 2*order+1)
    
    
    # output_all = []
    # params_all = []
    # for i in range(np.int32((2*order+1)**nparams)):
    #     index = i
    
    #     index_As, remainder = divmod(index, np.int32((2*order+1)**(nparams - 1)))
    #     index_h, remainder = divmod(remainder, np.int32((2*order+1)**(nparams-2)))
    #     index_cdm, index_b = divmod(remainder, np.int32((2*order+1)**(nparams-3)))

        
    #     params = np.array([ln_10_10_As_all[index_As], h_all[index_h], omega_cdm_all[index_cdm], omega_b_all[index_b]])
        
    #     output = lnlike(params)
        
    #     params_all.append(params)
    #     output_all.append(output)
        
    # np.save("params_all.npy", params_all)
    # np.save("output_all.npy", output_all)
        
    