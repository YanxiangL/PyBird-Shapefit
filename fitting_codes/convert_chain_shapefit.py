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
from chainconsumer import ChainConsumer
import findiff
from itertools import combinations

# sys.path.append('../')
# from pybird_dev.greenfunction import GreenFunction

# sys.path.append("../")
# from tbird.Grid import run_camb, run_class
# from fitting_codes.fitting_utils import format_pardict, read_chain_backend, BirdModel, do_optimization, get_Planck, FittingData

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
    tau = reader.get_autocorr_time(tol=0)
    #Using the autocorrelation time to figure out the burn-in. 
    burnin = int(2 * np.max(tau))
    #Retriving the chain and discard the burn-in, 
    samples = reader.get_chain(discard=burnin, flat=True)
    #The the log-posterior of the chain. 
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    #Find the best-fit parameters. 
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples


def read_chain(pardict, Shapefit, z_bin, with_zeus = True, index = None):

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
        
        if nz == 1:
            chainfile = str(
                fmt_str
                % (
                    pardict["shapefitfile"],
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
            chainfile = str(
                fmt_str
                % (
                    pardict["shapefitfile"][index],
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
    # chainfile = '../../data/check_prior_LCDM.hdf5'
    print(chainfile)
    #The next line returns the chains, best-fit parameters and the likelihood for each iteration. 
    c = ChainConsumer()
    if with_zeus == False:
        burntin, bestfit, like = read_chain_backend(chainfile)
        
        # burntin = interpolation_function(burntin)
        # burntin[:, 2] *= fsigma8_fid
    else:
        # burntin = np.load(chainfile).T
        # burntin = np.loadtxt('../../data/shapefit_chains_test.txt')
        burntin = np.loadtxt('../../data/Mark_LRG_SF_0p18.txt')
        print('../../data/Mark_LRG_SF_0p18.txt')
        
        # burntin = np.array([burntin[:, 2], burntin[:, 1], burntin[:, 0], burntin[:, 3]]).T
    
    if Approx_Gaussian == True:
        if Shapefit == True:
            c.add_chain(burntin[:, :4], parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$"], name = 'Shapefit')
            data = c.analysis.get_summary()
            # mean = np.array([data[r"$\alpha_{\perp}$"][1], data[r"$\alpha_{\parallel}$"][1], data[r'$f\sigma_8$'][1], data[r'$m$'][1]])
            # mean = bestfit[:4]
            mean = np.mean(burntin, axis=0)[:4]
            # mean = np.median(burntin, axis=0)[:4]
        else:
            c.add_chain(burntin[:, :3], parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$"], name = 'Shapefit')
            data = c.analysis.get_summary()
            # mean = np.array([data[r"$\alpha_{\perp}$"][1], data[r"$\alpha_{\parallel}$"][1], data[r'$f\sigma_8$'][1]])
            mean = np.mean(burntin, axis=0)[:3]
            # mean = bestfit[:3]
            
        cov = c.analysis.get_covariance()[1]
        
        return mean, cov
    else:
        if Shapefit == True:
            start = time.time()
            interpolator = sp.interpolate.LinearNDInterpolator(burntin[:, :4], like, fill_value=-1e30)
            end = time.time()
            print(end - start)
        else:
            start = time.time()
            interpolator = sp.interpolate.LinearNDInterpolator(burntin[:, :3], like, fill_value=-1e30)
            end = time.time()
            print(end - start)
            
        return interpolator

# def read_chain(pardict, Shapefit, z_bin, with_zeus = True):

#     # Reads in a chain containing template fit or Shapefit parameters, and returns the mean data vector and
#     # inverse covariance matrix for fitting with cosmological parameters. Assumes the data
#     # are suitably Gaussian.
    
#     if resum == True:
#         marg_str = "marg" if pardict["do_marg"] else "all"
#         hex_str = "hex" if pardict["do_hex"] else "nohex"
#         dat_str = "xi" if pardict["do_corr"] else "pk"
#         if Shapefit:
#             fmt_str = (
#                 "%s_%s_%2dhex%2d_%s_%s_%s_Shapefit_mock_%s_bin_"+str(z_bin)
#                 if pardict["do_corr"]
#                 else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_Shapefit_mock_%s_bin_"+str(z_bin)
#             )
#         else:
#             fmt_str = (
#                 "%s_%s_%2dhex%2d_%s_%s_%s_Templatefit_mock_%s_bin_"+str(z_bin)
#                 if pardict["do_corr"]
#                 else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_Templatefit_mock_%s_bin_"+str(z_bin)
#             )
#         fitlim = pardict["xfit_min"][0] if pardict["do_corr"] else pardict["xfit_max"][0]
#         fitlimhex = pardict["xfit_min"][2] if pardict["do_corr"] else pardict["xfit_max"][2]
        
#         if with_zeus:
#             fmt_str = fmt_str + '.npy'
#         else:
#             fmt_str = fmt_str + '_IS.hdf5'
    
#         taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
#         chainfile = str(
#             fmt_str
#             % (
#                 pardict["fitfile"],
#                 dat_str,
#                 fitlim,
#                 fitlimhex,
#                 taylor_strs[pardict["taylor_order"]],
#                 hex_str,
#                 marg_str,
#                 keyword
#             )
#         )
#     else:
#         chainfile = '../../data/DESI_KP4_LRG_ELG_QSO_pk_0.20hex0.20_3order_nohex_marg_noresum_bin_' + str(z_bin) + '.hdf5'
#     #This is the name of the chainfile. 
#     print(chainfile)
#     #The next line returns the chains, best-fit parameters and the likelihood for each iteration. 
#     # sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     if with_zeus == False:
#         burntin, bestfit, like = read_chain_backend(chainfile)
#         sample_new = burntin
#         sample_new = interpolation_function(sample_new)
#         sample_new[:, 2] *= fsigma8_fid
#     else:
#         # burntin = np.load(chainfile).T
#         # burntin = np.loadtxt('../../data/shapefit_chains_test.txt')
#         burntin = np.loadtxt('../../data/DESI_KP4_LRG_pk_0.20hex0.20_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mean_converted_bin_0_0.dat')
#         # print(burntin)
        
#         burntin = np.array([burntin[:, 2], burntin[:, 1], burntin[:, 0], burntin[:, 3]]).T
        
#         sample_new = burntin
        
#     bin_num = 40
#     freq, bins = np.histogramdd(sample_new[:, :4], bin_num, density = True)
#     bin_new = np.array([[(bins[i][j] + bins[i][j+1])/2.0 for j in range(bin_num)] for i in range(cosmo_num)])
#     interpolation_lnlike = sp.interpolate.RegularGridInterpolator(bin_new, freq, bounds_error=False, fill_value = -1.0e-30, method = 'linear')
    
#     return interpolation_lnlike


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
    if FullShape is True:
        chainfile = chainfile + "_FullShape"
    else:
        chainfile = chainfile + "_mock_" + keyword
        
    if method == 1:
        NWM = '_EH98'
    elif method == 2:
        NWM = '_Hinton2017'
    elif method == 3:
        NWM = '_Wallisch2018'
    else:
        raise ValueError('Incorrect method for de-wiggle power spectrum. Enter 1 for EH98, 2 for Hinton2017 and 3 for Wallisch2018.')
        
    # if method_fsigma8 == 1:
    #     NWM += '_default'
    # elif method_fsigma8 == 2:
    #     NWM += '_fsigma8'
    # elif method_fsigma8 == 3:
    #     NWM += '_fsigmas8'
    # else:
    #     raise ValueError('Incorrect method for de-wiggle power spectrum. Enter 1 for EH98, 2 for Hinton2017 and 3 for Wallisch2018.')
    
    oldfile = chainfile + NWM + ".hdf5"
    newfile = chainfile + NWM + ".dat"
    
    # oldfile = '../../data/check_prior_wCDM_converted.hdf5'
    
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
        max_iter = 50000
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
    if with_w0 == True:
        w = params[4]
    elif with_w0_wa == True:
        wa = params[5]
    elif with_omegak == True:
        omegak = params[4]
    # ln10As, h, Omega_m = params[:3]
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm
    if with_w0 == True:
        valueref = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], pardict['w']]))
    elif with_w0_wa == True:
        valueref = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], pardict['w'], pardict['wa']]))
    elif with_omegak == True:
        valueref = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], pardict['Omega_k']]))
    else:
        valueref = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b']]))
    delta = np.float64(pardict['dx'])
    # lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    # upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta
    
    lower_bounds = valueref - np.float64(pardict["order"]) * delta
    upper_bounds = valueref + np.float64(pardict["order"]) * delta

    # Flat priors for cosmological parameters
    
    if with_w0 == True:
        if np.any(np.less([ln10As, h, omega_cdm, omega_b, w], lower_bounds)) or np.any(
            np.greater([ln10As, h, omega_cdm, omega_b, w], upper_bounds)
        ):
            return -np.inf
    elif with_w0_wa == True:
        if np.any(np.less([ln10As, h, omega_cdm, omega_b, w, wa], lower_bounds)) or np.any(
            np.greater([ln10As, h, omega_cdm, omega_b, w, wa], upper_bounds)
        ):
            return -np.inf
    elif with_omegak == True:
        if np.any(np.less([ln10As, h, omega_cdm, omega_b, omegak], lower_bounds)) or np.any(
            np.greater([ln10As, h, omega_cdm, omega_b, omegak], upper_bounds)
        ):
            return -np.inf
    else:
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
    # if FullShape == False:
    #     omega_b_prior = -0.5 * (omega_b - float(pardict['omega_b'])) ** 2 / 0.00037 ** 2
    # else:
    #     omega_b_prior = 0.0
    # omega_b_prior = 0
    # omega_b_prior = -0.5 * (omega_b - float(pardict['omega_b'])) ** 2 / 0.00037 ** 2
    omega_b_prior = -0.5 * (omega_b - 0.02237) ** 2 / 0.00037 ** 2

    
    # if (omega_b < 0.02163) or (omega_b > 0.02311):
    #     return -np.inf
    # omega_b_prior = 0.0
    
    # b1, bp, bd = params[4:]
    
    # if b1 < 0.0 or b1 > 4.0:
    #     return -np.inf
    
    # if bp < 0.0 or bp > 5.0:
    #     return -np.inf
    
    # if bd < -20.0 or bd > 20.0:
    #     return -np.inf

    return omega_b_prior
    # return 0.0

def lnlike(params):
    # theo_aperp, theo_apara, theo_fAmp, theo_mslope, fsigma8, theo_mslope_dash, theo_fAmp_dash, theo_fAmp_prime, theo_mslope_prime = interpolation_function(params[:4])[0]
    
    # theo_aperp, theo_apara, theo_fAmp, theo_mslope, fsigma8, error, ratio = interpolation_function(params[:4])[0]
    
    if FullShape == True:
        model = np.tile(params, nz)
        
        chi_squared = np.dot(np.dot(model-data_all, cov_inv_all), model-data_all)
        
        return -0.5*chi_squared
    else:
        model = []
        for i in range(nz):
            # theo_aperp, theo_apara, theo_fAmp, theo_EH98, theo_hinton, theo_wallisch, fsigma8_ratio, fsigmas8_ratio = interpolation_functions[i](params)[0] 
            
            # if with_w0 == True or with_omegak == True:
            #     theo_aperp, theo_apara, theo_fAmp, theo_EH98, theo_hinton, theo_wallisch, fsigma8_ratio, fsigmas8_ratio = interpolation_functions[i](params[:5])[0]
            # elif with_w0_wa == True:
            #     theo_aperp, theo_apara, theo_fAmp, theo_EH98, theo_hinton, theo_wallisch, fsigma8_ratio, fsigmas8_ratio = interpolation_functions[i](params[:6])[0]
            # else:
            #     theo_aperp, theo_apara, theo_fAmp, theo_EH98, theo_hinton, theo_wallisch, fsigma8_ratio, fsigmas8_ratio = interpolation_functions[i](params[:4])[0]
            
            dtheta = params - valueref
            theo_aperp, theo_apara, theo_fAmp, theo_EH98, theo_hinton, theo_wallisch, fsigma8_ratio, fsigmas8_ratio = get_ParamsTaylor(dtheta, interpolation_functions[i], pardict["taylor_order"])
            
            # print(theo_aperp, theo_apara, theo_fAmp, theo_EH98)
            
            # theo_aperp = fsigma8_ratio
            # theo_apara = fsigmas8_ratio
            
            # print(theo_aperp, theo_apara, theo_fAmp, theo_EH98)
            
            if method == 1:
                theo_mslope = theo_EH98
            elif method == 2:
                theo_mslope = theo_hinton
            elif method == 3:
                theo_mslope = theo_wallisch
            else:
                raise ValueError('Incorrect method for de-wiggle power spectrum. Enter 1 for EH98, 2 for Hinton2017 and 3 for Wallisch2018.')
                
            # theo_fAmp = fsigma8_ratio
            # if method_fsigma8 == 1:
            #     theo_fAmp = theo_fAmp
            # elif method_fsigma8 == 2:
            #     theo_fAmp = fsigma8_ratio
            # elif method_fsigma8 == 3:
            #     theo_fAmp = fsigmas8_ratio
            # else:
            #     raise ValueError('Incorrect method for de-wiggle power spectrum. Enter 1 for EH98, 2 for Hinton2017 and 3 for Wallisch2018.')
            
            # theo_fAmp = fsigmas8_ratio
            # model = np.array([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope, b1, bp, bd])
            
            
            if Approx_Gaussian == True:
                if nz == 1:
                    if Shapefit == True:
                        model = np.array([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope])
                    else:
                        model = np.array([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid])
                else:
                    if Shapefit == True:
                        model.append([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid[i], theo_mslope])
                    else:
                        model.append([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid[i]])
            else:
                if nz == 1:
                    if Shapefit == True:
                        model = interpolator_all[i]([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope])
                    else:
                        model = interpolator_all[i]([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid])
                else:
                    if Shapefit == True:
                        model.append(interpolator_all[i]([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid[i], theo_mslope]))
                    else:
                        model.append(interpolator_all[i]([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid[i]]))
            
            # model = np.array([theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope])
            
            # print(i, fsigma8_fid[i])
        
        if Approx_Gaussian == True:
            model = np.array(model)
            
            if nz > 1.5:
                model = model.flatten()
                # print(model)
                # print(data_all)
                
            chi_squared = np.dot(np.dot(model-data_all, cov_inv_all), model-data_all)
            # if np.random.rand() < 0.001:
            #     print(params[:5], chi_squared)
                
            return -0.5*chi_squared
        else:
            lnlike = np.sum(model)
            
            return lnlike



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
    
def get_pder_lin(parref, pi, dx, filename = None, template=False):
    """Calculates the derivative aroud the Grid.valueref points. Do this only once.
    gridshape is 2 * order + 1, times the number of free parameters
    pi is of shape gridshape, n multipoles, k length, P columns (zeroth being k's)"""
    # Findiff syntax is Findiff((axis, delta of uniform grid along the axis, order of derivative, accuracy))
    t0 = time.time()

    if template:
        lenpar = 4
        idx = int(parref["template_order"]) + 1
    else:
        lenpar = len(parref["freepar"])
        idx = int(parref["order"]) + 1

    p0 = pi[(idx,) * lenpar]
    # t1 = time.time()
    # print("Done p0 in %s sec" % str(t1 - t0))

    dpdx = np.array([findiff.FinDiff((i, dx[i], 1), acc=4)(pi)[(idx,) * lenpar] for i in range(lenpar)])
    # print(findiff.coefficients(deriv=1, acc=4))
    # t0 = time.time()
    # print("Done dpdx in %s sec" % str(t0 - t1))

    # Second derivatives
    d2pdx2 = np.array([findiff.FinDiff((i, dx[i], 2), acc=2)(pi)[(idx,) * lenpar] for i in range(lenpar)])
    # t1 = time.time()
    # print("Done d2pdx2 in %s sec" % str(t1 - t0))

    d2pdxdy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), acc=2)(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    # t0 = time.time()
    # print("Done d2pdxdy in %s sec" % str(t0 - t1))

    # Third derivatives: we only need it for A_s, so I do this by hand
    d3pdx3 = np.array([findiff.FinDiff((i, dx[i], 3))(pi)[(idx,) * lenpar] for i in range(lenpar)])
    # t1 = time.time()
    # print("Done d3pdx3 in %s sec" % str(t1 - t0))

    d3pdx2dy = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1))(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    # t0 = time.time()
    # print("Done d3pdx2dy in %s sec" % str(t0 - t1))

    d3pdxdy2 = np.array(
        [
            [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2))(pi)[(idx,) * lenpar]]
            for (i, j) in combinations(range(lenpar), 2)
        ]
    )
    # t1 = time.time()
    # print("Done d3pdxdy2 in %s sec" % str(t1 - t0))

    d3pdxdydz = np.array(
        [
            [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1))(pi)[(idx,) * lenpar]]
            for (i, j, k) in combinations(range(lenpar), 3)
        ]
    )
    # t0 = time.time()
    # print("Done d3pdxdydz in %s sec" % str(t0 - t1))

    # d4pdx4 = np.array([findiff.FinDiff((i, dx[i], 4))(pi)[(idx,) * lenpar] for i in range(lenpar)])
    # t1 = time.time()
    # print("Done d4pdx4 in %s sec" % str(t1 - t0))

    # d4pdx3dy = np.array(
    #     [
    #         [i, j, findiff.FinDiff((i, dx[i], 3), (j, dx[j], 1))(pi)[(idx,) * lenpar]]
    #         for (i, j) in combinations(range(lenpar), 2)
    #     ]
    # )
    # t0 = time.time()
    # print("Done d4pdx3dy in %s sec" % str(t0 - t1))

    # d4pdxdy3 = np.array(
    #     [
    #         [i, j, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 3))(pi)[(idx,) * lenpar]]
    #         for (i, j) in combinations(range(lenpar), 2)
    #     ]
    # )
    # t1 = time.time()
    # print("Done d4pdxdy3 in %s sec" % str(t1 - t0))

    # d4pdx2dydz = np.array(
    #     [
    #         [i, j, k, findiff.FinDiff((i, dx[i], 2), (j, dx[j], 1), (k, dx[k], 1))(pi)[(idx,) * lenpar]]
    #         for (i, j, k) in combinations(range(lenpar), 3)
    #     ]
    # )
    # t0 = time.time()
    # print("Done d4pdx2dydz in %s sec" % str(t0 - t1))

    # d4pdxdy2dz = np.array(
    #     [
    #         [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 2), (k, dx[k], 1))(pi)[(idx,) * lenpar]]
    #         for (i, j, k) in combinations(range(lenpar), 3)
    #     ]
    # )
    # t1 = time.time()
    # print("Done d4pdxdy2dz in %s sec" % str(t1 - t0))

    # d4pdxdydz2 = np.array(
    #     [
    #         [i, j, k, findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 2))(pi)[(idx,) * lenpar]]
    #         for (i, j, k) in combinations(range(lenpar), 3)
    #     ]
    # )
    # t0 = time.time()
    # print("Done d4pdxdydz2 in %s sec" % str(t0 - t1))

    # d4pdxdydzdzm = np.array(
    #     [
    #         [
    #             i,
    #             j,
    #             k,
    #             m,
    #             findiff.FinDiff((i, dx[i], 1), (j, dx[j], 1), (k, dx[k], 1), (m, dx[m], 1))(pi)[(idx,) * lenpar],
    #         ]
    #         for (i, j, k, m) in combinations(range(lenpar), 4)
    #     ]
    # )
    # t1 = time.time()
    # print("Done d4pdxdydzdm in %s sec" % str(t1 - t0))

    allder = (
        p0,
        dpdx,
        d2pdx2,
        d2pdxdy,
        d3pdx3,
        d3pdx2dy,
        d3pdxdy2,
        d3pdxdydz,
        # d4pdx4,
        # d4pdx3dy,
        # d4pdxdy3,
        # d4pdx2dydz,
        # d4pdxdy2dz,
        # d4pdxdydz2,
        # d4pdxdydzdzm,
    )
    # np.save(filename, allder)
    return allder

def get_ParamsTaylor(dtheta, derivatives, taylor_order):
    # Shape of dtheta: number of free parameters
    # Shape of derivatives: tuple up to third derivative where each element has shape (num free par, multipoles, lenk, columns)
    t1 = np.einsum("p,pm->m", dtheta, derivatives[1])
    t2diag = np.einsum("p,pm->m", dtheta ** 2, derivatives[2])
    t2nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * d[2] for d in derivatives[3]], axis=0)
    t3diag = np.einsum("p,pm->m", dtheta ** 3, derivatives[4])
    t3semidiagx = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * d[2] for d in derivatives[5]], axis=0)
    t3semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * d[2] for d in derivatives[6]], axis=0)
    t3nondiag = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[7]], axis=0)
    # t4diag = np.einsum("p,pm->m", dtheta ** 4, derivatives[8])
    # t4semidiagx = np.sum([dtheta[d[0]] ** 3 * dtheta[d[1]] * d[2] for d in derivatives[9]], axis=0)
    # t4semidiagy = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 3 * d[2] for d in derivatives[10]], axis=0)
    # t4semidiagx2 = np.sum([dtheta[d[0]] ** 2 * dtheta[d[1]] * dtheta[d[2]] * d[3] for d in derivatives[11]], axis=0)
    # t4semidiagy2 = np.sum([dtheta[d[0]] * dtheta[d[1]] ** 2 * dtheta[d[2]] * d[3] for d in derivatives[12]], axis=0)
    # t4semidiagz2 = np.sum([dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] ** 2 * d[3] for d in derivatives[13]], axis=0)
    # t4nondiag = np.sum(
    #     [dtheta[d[0]] * dtheta[d[1]] * dtheta[d[2]] * dtheta[d[3]] * d[4] for d in derivatives[14]], axis=0
    # )
    allPS = derivatives[0] + t1
    if taylor_order > 1:
        allPS += 0.5 * t2diag + t2nondiag
        if taylor_order > 2:
            allPS += t3diag / 6.0 + t3semidiagx / 2.0 + t3semidiagy / 2.0 + t3nondiag
            # if taylor_order > 3:
            #     allPS += (
            #         t4diag / 24.0
            #         + t4semidiagx / 6.0
            #         + t4semidiagy / 6.0
            #         + t4semidiagx2 / 2.0
            #         + t4semidiagy2 / 2.0
            #         + t4semidiagz2 / 2.0
            #         + t4nondiag
            #     )

    return allPS

if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    Shapefit = bool(int(sys.argv[2]))
    FullShape = bool(int(sys.argv[3]))
    # resum = bool(int(sys.argv[4]))
    resum = True
    # fixedbias = bool(int(sys.argv[4]))
    job_total_num = int(sys.argv[4])
    method = int(sys.argv[5]) #Enter 1 for EH98, 2 for Hinton2017 and 3 for Wallisch2018. 
    # method_fsigma8 = int(sys.argv[7])
    # method_fsigma8 = 3
    Approx_Gaussian = bool(int(sys.argv[6]))
    # flatprior = int(sys.argv[8])
    
    # print(method_fsigma8)
    
    try:
        #Do single mock. 
        mock_num = int(sys.argv[7])
        mean = False
    except:
        #Do mean of the mock. 
        mean = True
    
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    
    z_data_all = np.float64(pardict["z_pk"])
    nz = len(z_data_all)
    
    if resum == False:
        keyword = '_noresum'
    
    #Read in the mock files. 
    if Shapefit == True:
        if mean == False:
            keyword = str(mock_num)
            
            # datafiles = np.loadtxt(pardict['datafile'], dtype=str)
            # mockfile = str(datafiles) + str(mock_num) + '.dat'
            # newfile = '../config/data_mock_' + str(mock_num) + '.txt'
            # text_file = open(newfile, "w")
            # n = text_file.write(mockfile)
            # text_file.close()
            # pardict['datafile'] = newfile
        else:
            keyword = str("mean")
            # pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
            # pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
    else:
        if mean == False:
            keyword = str(mock_num)
            
            # datafiles = np.loadtxt(pardict['datafile'], dtype=str)
            # mockfile = str(datafiles) + str(mock_num) + '.dat'
            # newfile = '../config/data_mock_' + str(mock_num) + '.txt'
            # text_file = open(newfile, "w")
            # n = text_file.write(mockfile)
            # text_file.close()
            # pardict['datafile'] = newfile
        else:
            keyword = str("mean")
            # pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
            # pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
            
    
    # fittingdata = FittingData(pardict)
    
    # cosmo_num = 4

    # Set up the BirdModel
    if Shapefit:
        print("Using Shapefit")
        
    else:
        if FullShape == False:
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
    interpolation_functions = []
    for j in range(nz):
        with_w0 = False
        with_w0_wa = False
        with_omegak = False
        grid_all = []
        for i in range(job_total_num):
            if "w" in pardict.keys():
                filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_w0' + 'bin_' + str(pardict['red_index'][j]) + ".npy"
                with_w0 = True
                if "wa" in pardict.keys():
                    filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_w0_wa' + 'bin_' + str(pardict['red_index'][j]) + ".npy"
                    with_w0 = False
                    with_w0_wa = True
            elif pardict['freepar'][-1] == 'Omega_k':
                filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_omegak' + 'bin_' + str(pardict['red_index'][j]) + ".npy"
                with_omegak = True
            else:
                filename = "Shapefit_Grid_sigma_" + str(i) + "_" + str(job_total_num) + 'bin_' + str(pardict['red_index'][j]) + ".npy"
                
            
            grid = np.load(filename)
            grid_all.append(grid)
        
        print(filename)
        grid_all = np.vstack(grid_all)
        
        order = np.int32(pardict["order"])
        delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
        valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
        truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
        shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
        padshape = [(1, 1)] * (len(shapecrd) - 1)
        grid_all_new = grid_all.reshape((*shapecrd[1:], np.shape(grid_all)[1]))
        grid_all_new = np.pad(grid_all_new, padshape + [(0, 0)], "constant", constant_values=0)
        interpolation_function = get_pder_lin(pardict, grid_all_new, delta)
        
        # if with_w0 == True:
        #     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        # elif with_w0_wa == True:
        #     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        # elif with_omegak == True:
        #     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        # else:
        #     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        
        # interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all_new)
        
        # np.save('Grid_' + str(i) + '.npy', interpolation_function)
        
        # raise ValueError('Test complete')
        # interpolation_function2 = sp.interpolate.RegularGridInterpolator(grid_all, truecrd)
        interpolation_functions.append(interpolation_function)
    
    # #This is in h/Mpc. 
    # kmpiv = 0.03
    
    # kvec = np.logspace(-2.0, 0.0, 300)
    
    # fiducial_all = []
    
    fit_data_all = []
    
    fit_cov_all = []
    
    interpolator_all = []
    
    fsigma8_fid = np.float64(pardict['fsigma8'])
    
    
    # # template.compute()
    # if fixedbias == True:
    #     keyword = keyword + '_fixedbias'
        
    # # keyword = keyword + '_best'
    # if flatprior == True:   
    #     keyword += '_flat'
    # else:
    #     keyword += '_Gaussian'
    
    if pardict['constrain'] == 'Single':
        keyword += '_single'
    
    if pardict['prior'] == 'MinF':
        keyword += '_MinF'
        MinF = True
    elif pardict['prior'] == 'MaxF':
        keyword += '_MaxF'
        MinF = False
    elif pardict['prior'] == 'BOSS_MaxF':
        keyword += '_BOSS_MaxF'
        MinF = False
    elif pardict['prior'] == 'BOSS_MinF':
        keyword += '_BOSS_MinF'
        MinF = True
    else:
        raise ValueError('Enter either "MinF", "MaxF", "BOSS_MaxF", "BOSS_MinF" to determine to prior for the marginalized parameters. ')
        
    if MinF == True or pardict['prior'] == 'BOSS_MaxF':
        pardict['vary_c4'] = 0
    
    # if int(pardict['vary_c4']) == 0 and MinF == False:
    #     keyword += '_anticorr'
    
    if FullShape == True:
        # chainfile = ['../../data/DESI_KP4_LRG_8_DE_pk_0.20hex0.20_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mean_single_BOSS_MaxF.hdf5', 
        #              '../../data/DESI_KP4_ELG_8_DE_pk_0.20hex0.20_3order_nohex_marg_kmin0p02_fewerbias_bin_1_mean_single_BOSS_MaxF.hdf5', 
        #              '../../data/DESI_KP4_QSO_8_DE_pk_0.20hex0.20_3order_nohex_marg_kmin0p02_fewerbias_bin_2_mean_single_BOSS_MaxF.hdf5']
        chainfile = ['../../data/DESI_KP4_LRG_8_DE_pk_0.20hex0.20_3order_nohex_marg_Shapefit_planck_mock_mean_single_BOSS_MaxF_Approx_EH98.hdf5', 
                     '../../data/DESI_KP4_ELG_8_DE_pk_0.20hex0.20_3order_nohex_marg_Shapefit_planck_mock_mean_single_BOSS_MaxF_Approx_EH98.hdf5', 
                     '../../data/DESI_KP4_QSO_8_DE_pk_0.20hex0.20_3order_nohex_marg_Shapefit_planck_mock_mean_single_BOSS_MaxF_Approx_EH98.hdf5']
        
    for i in range(nz):
    
        # Read in the chain and get the template parameters
        if FullShape == True:
            print("Using FullShape fit")
            c = ChainConsumer()

            burntin, bestfit, like = read_chain_backend(chainfile[i])
            
            c.add_chain(burntin[:, :5], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}}h^2$", r"$\Omega_{\mathrm{b}}h^2$", r"$w$"])

            mean = np.mean(burntin, axis=0)[:5]
            cov = c.analysis.get_covariance()[1]
            # useful = np.load("FullShape_template.npy")
            fitdata = mean
            fitcov = cov
        elif Shapefit == True:
            # fitdata, fitcov = read_chain(pardict, Shapefit, i, with_zeus=True)
            # fitdata, fitcov = read_chain(pardict, Shapefit, i, with_zeus=False)
            
            if Approx_Gaussian == True:
                if nz != 1:
                    fitdata, fitcov = read_chain(pardict, Shapefit, np.int32(pardict['red_index'])[i], with_zeus=False, index = i)
                    print(fsigma8_fid[i])
                else:
                    fitdata, fitcov = read_chain(pardict, Shapefit, int(pardict['red_index']), with_zeus=False)
                    print(fsigma8_fid)
            else:
                if nz != 1:
                    interpolator = read_chain(pardict, Shapefit, np.int32(pardict['red_index'])[i], with_zeus=False, index = i)
                    print(interpolator([1.0, 1.0, fsigma8_fid[i], 0.0]))
                else:
                    interpolator = read_chain(pardict, Shapefit, int(pardict['red_index']), with_zeus=False)
                    
                    print(interpolator([1.0, 1.0, fsigma8_fid, 0.0]))
                    
        # else:
        #     fitdata = np.array([2.95623019, 0.67753166, 0.11692257, 0.02236975])
        #     fitcov = np.eye(4)
            
        if Approx_Gaussian == True:
            fit_data_all.append(fitdata)
            fit_cov_all.append(fitcov)
        else:
            interpolator_all.append(interpolator)
    # np.savetxt('../../data/cov_mark.txt', fit_cov_all[0])
    if Approx_Gaussian == True:
        cov_all = sp.linalg.block_diag(*fit_cov_all)
        cov_inv_all = np.linalg.inv(cov_all)
        data_all = np.concatenate(fit_data_all)
        # np.save("fit_data.npy", fitdata)
        # np.save("fit_cov.npy", fitcov)
        # np.save("Pk_ratio_fid.npy", Pk_ratio_fid)
        print("The mean posterior is ", data_all)
    
    # interpolation_lnlike = read_chain(pardict, Shapefit, 0, with_zeus=False)

    #We start MCMC with the True cosmological parameters. 
    # start = np.array([birdmodel.valueref[0], birdmodel.valueref[1], birdmodel.valueref[2], birdmodel.valueref[3]])
    if with_w0 == True:
        start = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], pardict['w']]))
    elif with_w0_wa == True:
        start = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], pardict['w'], pardict['wa']]))
    elif with_omegak == True:
        start = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], 1e-4]))
    else:
        start = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b']]))
    # start = np.float64(np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b'], 2.0, 1.0, 1.0]))
    # start = np.array([birdmodel.valueref[0], birdmodel.valueref[1], (birdmodel.valueref[2] + birdmodel.valueref[3])/birdmodel.valueref[1]**2])
    
    print(start)
    
    # print(cov_all, np.shape(cov_all))
    
    # print(template.Neff())
    
    # print(interpolation_function(start))
    
    if Approx_Gaussian == True:
        keyword += '_Approx'
    else:
        keyword += '_Interpolate'

    # Does an optimization
    # result = do_optimization(lambda *args: -lnpost(*args), start)
    # print(result)

    # Does an MCMC and then post-processes to get some derived params
    do_emcee(lnpost, start)
    
    # test = np.array([2.9624, 0.6673, 0.1147, 0.022])
    
    # print(lnlike(test))
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
    
    # def find_diff(final_new, num, index):
    #     low = np.min(final_new[:, index])
    #     high = np.max(final_new[:, index])
    #     slide = np.linspace(low, high, num = num)
    #     diff = slide[1] - slide[0]
    #     return low, high, slide, diff
    
    # from numba import jit
    
    # @jit(nopython=True)
    # def find_index(num, final_new, low, high, diff):
    #     low_0, low_1, low_2, low_3 = low
    #     high_0, high_1, high_2, high_3 = high
    #     diff_0, diff_1, diff_2, diff_3 = diff
    #     # low_0, high_0, slide_0, diff_0 = find_diff(final_new, num, 0)
    #     # low_1, high_1, slide_1, diff_1 = find_diff(final_new, num, 1)
    #     # low_2, high_2, slide_2, diff_2 = find_diff(final_new, num, 2)
    #     # low_3, high_3, slide_3, diff_3 = find_diff(final_new, num, 3)
        
    #     point_0 = np.linspace(low_0+diff_0/2.0, high_0-diff_0/2.0, num-1)
    #     point_1 = np.linspace(low_1+diff_1/2.0, high_1-diff_1/2.0, num-1)
    #     point_2 = np.linspace(low_2+diff_2/2.0, high_2-diff_2/2.0, num-1)
    #     point_3 = np.linspace(low_3+diff_3/2.0, high_3-diff_3/2.0, num-1)
        
    #     length = np.float64(len(final_new))
        
    #     # all_point = np.zeros((num-1, num-1, num-1, num-1))
    #     prob_all = np.zeros((num-1, num-1, num-1, num-1))
        
    #     for a in range(num-1):
    #         index_0 = np.where(np.logical_and(final_new[:, 0] >= low_0 + a*diff_0, final_new[:, 0] < low_0 + (a+1)*diff_0))[0]
    #         for b in range(num-1):
    #             index_1 = np.where(np.logical_and(final_new[:, 1] >= low_1 + b*diff_1, final_new[:, 1] < low_1 + (b+1)*diff_1))[0]
    #             index_dash_1 = np.intersect1d(index_0, index_1)
    #             if len(index_dash_1) != 0:
    #                 for c in range(num - 1):
    #                     index_2 = np.where(np.logical_and(final_new[:, 2] >= low_2 + c*diff_2, final_new[:, 2] < low_2 + (c+1)*diff_2))[0]
    #                     index_dash_2 = np.intersect1d(index_dash_1, index_2)
    #                     if len(index_dash_2) != 0:
    #                         for d in range(num - 1):
    #                             index_3 = np.where(np.logical_and(final_new[:, 3] >= low_3 + d*diff_3, final_new[:, 3] < low_3 + (d+1)*diff_3))[0]
                                
    #                             index_dash_3 = np.intersect1d(index_dash_2, index_3)
                                
    #                             prob_all[a][b][c][d] = np.float64(index_dash_3)/length
                                
    #     return prob_all
                        
        
        # a = 0
        # b = 0
        # c = 0
        # d = 0
        # print('Start')
        # for i in range((num-1)**4):
        #     index_0 = np.where(np.logical_and(final_new[:, 0] >= low_0 + a*diff_0, final_new[:, 0] < low_0 + (a+1)*diff_0))[0]
        #     index_1 = np.where(np.logical_and(final_new[:, 1] >= low_1 + b*diff_1, final_new[:, 1] < low_1 + (b+1)*diff_1))[0]
        #     index_2 = np.where(np.logical_and(final_new[:, 2] >= low_2 + c*diff_2, final_new[:, 2] < low_2 + (c+1)*diff_2))[0]
        #     index_3 = np.where(np.logical_and(final_new[:, 3] >= low_3 + d*diff_3, final_new[:, 3] < low_3 + (d+1)*diff_3))[0]
            
        #     # index = np.intersect1d(index_0, index_1)
        #     index = np.intersect1d(np.intersect1d(np.intersect1d(index_0, index_1), index_2), index_3)
        #     prob = np.float64(len(index))/np.float64(length)
            
        #     all_point.append([point_0[a], point_1[b], point_2[c], point_3[d]])
        #     prob_all.append(prob)
            
        #     a += 1
            
        #     if a == num - 2:
        #         print(a, b, c, d, prob)
        #         a = 0
        #         b += 1
                
        #     if b == num - 2:
        #         b = 0 
        #         c += 1
                
        #     if c == num - 2:
                
        #         c = 0
        #         d += 1
                
        # return np.array(all_point), np.array(prob_all)
            
            
            
    