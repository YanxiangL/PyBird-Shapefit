# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:32:00 2023

@author: s4479813
"""

import numpy as np
import sys
from configobj import ConfigObj
from multiprocessing import Pool
from scipy.linalg import lapack, cholesky
import copy
import emcee
from chainconsumer import ChainConsumer
import scipy as sp
from scipy import interpolate

sys.path.append("../")
from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
)


def do_emcee(func, start):

    

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 8

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    chainfile = '../../data/check_prior_LCDM.hdf5'    

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)

    # with Pool(processes=ncpu) as pool:
    with Pool() as pool:

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, nparams, func, pool=pool, backend=backend, vectorize=False)

        # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
        max_iter = 70000
        index = 0
        old_tau = np.inf
        autocorr = np.empty(max_iter)
        counter = 0
        for sample in sampler.sample(begin, iterations=max_iter, progress=True):

            # Only check convergence every 100 steps
            if sampler.iteration % 1000:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            # autocorr[index] = np.max(tau)
            autocorr[index] = np.max(tau)
            counter += 1000
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print("Max Auto-Correlation time: {0:.3f}".format(autocorr[index]))

            # Check convergence
            # converged = np.all(tau * 100 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            
            # converged = np.all(tau[:cosmo_num] * 50 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            
            if converged:
                print("Reached Auto-Correlation time goal: %d > 50 x %.3f" % (counter, autocorr[index]))
                break
            old_tau = tau
            index += 1
            


def lnpost(params):

    # # # This returns the posterior distribution which is given by the log prior plus the log likelihood
    # prior = lnprior(params, birdmodel_all, Shapefit)
    # if not np.isfinite(prior):
    #     return -np.inf
    
    # # if one_nz == False:
    # #     like = lnlike(params, birdmodel_all, fittingdata, plt, Shapefit)
    # # else:
    # #     like = lnlike(params, birdmodel_all, fittingdata, plt, Shapefit)
    
    # like = lnlike(params, birdmodel_all, fittingdata, plt, Shapefit)
    
    if params.ndim == 1:
        try:
            prior = lnprior(params, Shapefit)[0]
        except:
            prior = lnprior(params, Shapefit)
        
    else:
        prior = lnprior(params, Shapefit)
        
            
    return prior


def lnprior(params, Shaptfit):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    
    
    
    if params.ndim == 1:
        params = params.reshape((-1, len(params)))
    params = params.T
    
    # print(params)
    # print(params)
    
    priors = np.zeros(np.shape(params[1]))
    
    alpha_perp, alpha_par, fsigma8, m = params
    
    
    priors += np.where(np.logical_or(alpha_perp < 0.90, alpha_perp > 1.10), -1.0e30, 0.0)
    priors += np.where(np.logical_or(alpha_par < 0.90, alpha_par > 1.10), -1.0e30, 0.0)
    priors += np.where(np.logical_or(fsigma8 < 0.0, fsigma8 > 1.0), -1.0e30, 0.0)
    if Shapefit == True:
        priors += np.where(np.logical_or(m < -0.4, m > 0.4), -1.0e30, 0.0)
        

       
    
    return priors

    def read_chain_backend(chainfile):

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
    
    
def read_chain_backend(chainfile):

    reader = emcee.backends.HDFBackend(chainfile)

    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin, flat=True)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples


if __name__ == "__main__":

    Shapefit = True
    start = np.array([1.0, 1.0, 0.450144, 1e-4])
    do_emcee(lnpost, start)
    
    # sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend('../../data/check_prior_LCDM.hdf5')
    
    # configfile = '../config/tbird_KP4_LRG.ini'
    
    # pardict = ConfigObj(configfile)

    # # Just converts strings in pardicts to numbers in int/float etc.
    # pardict = format_pardict(pardict)
    
    # nz = len(pardict['red_index'])
    # job_total_num = 81
    # #This stores the input parameters. 
    # # birdmodel = BirdModel(pardict, template=True, direct=True, fittingdata=fittingdata, window = None, Shapefit=Shapefit)
    # interpolation_functions = []
    # for j in range(nz):
    #     with_w0 = False
    #     with_w0_wa = False
    grid_all = []
    for i in range(job_total_num):
        if "w" in pardict.keys():
            filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_w0' + 'bin_' + str(pardict['red_index'][j]) + ".npy"
            with_w0 = True
            if "wa" in pardict.keys():
                filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_w0_wa' + 'bin_' + str(pardict['red_index'][j]) + ".npy"
                with_w0 = False
                with_w0_wa = True
        else:
            filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + 'bin_' + str(pardict['red_index'][j]) + ".npy"
        grid = np.load(filename)
        grid_all.append(grid)
    
    grid_all = np.vstack(grid_all)
    order = np.int32(pardict["order"])
    delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
    valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
    shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
    grid_all = grid_all.reshape((*shapecrd[1:], np.shape(grid_all)[1]))
    #     if with_w0 == True:
    #         interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
    #     elif with_w0_wa == True:
    #         interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
    #     else:
    #         interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
    #     # interpolation_function2 = sp.interpolate.RegularGridInterpolator(grid_all, truecrd)
    #     interpolation_functions.append(interpolation_function)
    
