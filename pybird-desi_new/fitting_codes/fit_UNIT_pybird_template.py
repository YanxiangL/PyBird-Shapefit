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

# from fitting_codes.find_prob_dist import find_prior

# def find_prior(configfile):
    
#     chainfile = '../../data/prior_transform.hdf5'
#     sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     pardict = ConfigObj(configfile)

#     # Just converts strings in pardicts to numbers in int/float etc.
#     pardict = format_pardict(pardict)

#     job_total_num = 81
    
#     grid_all = []
#     for i in range(job_total_num):
#         grid = np.load("Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + ".npy")
#         grid_all.append(grid)

#     grid_all = np.vstack(grid_all)
#     order = float(pardict["order"])
#     delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
#     valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
#     truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
#     shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
#     grid_all = grid_all.reshape((*shapecrd[1:], np.shape(grid_all)[1]))
#     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all[:, :, :, :, :4])
    
#     result = interpolation_function(sample_new)
    
#     fsigma8_fid = 0.450144
    
#     result[:, 2] = result[:, 2]*fsigma8_fid
    
#     print('Start creating linear interpolation.')
#     prior_new = sp.interpolate.LinearNDInterpolator(result, max_log_likelihood_new, fill_value = -1.0e30)
#     print('Finish creating linear interpolation.')
    
#     return prior_new

def do_emcee(func, start, keyword):

    

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 8

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = (
        "%s_%s_%2dhex%2d_%s_%s_%s_"+keyword+"_bin_"+str(redindex)+".hdf5"
        if pardict["do_corr"]
        else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_" + keyword + "_bin_"+str(redindex)+".hdf5"
    )
    fitlim = birdmodel_all[0].pardict["xfit_min"][0] if pardict["do_corr"] else birdmodel_all[0].pardict["xfit_max"][0]
    fitlimhex = birdmodel_all[0].pardict["xfit_min"][2] if pardict["do_corr"] else birdmodel_all[0].pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodel_all[0].pardict["fitfile"],
            dat_str,
            fitlim,
            fitlimhex,
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
    )
    print(chainfile)

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
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            # autocorr[index] = np.max(tau)
            autocorr[index] = np.max(tau[:cosmo_num])
            counter += 100
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print("Max Auto-Correlation time: {0:.3f}".format(autocorr[index]))

            # Check convergence
            # converged = np.all(tau * 100 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            
            # converged = np.all(tau * 50 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            
            converged = np.all(tau[:cosmo_num] * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            
            if converged:
                print("Reached Auto-Correlation time goal: %d > 50 x %.3f" % (counter, autocorr[index]))
                break
            old_tau = tau
            index += 1
            
def do_zeus(func, start, keyword):

    import zeus

    # # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 4

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = (
        "%s_%s_%2dhex%2d_%s_%s_%s_"+keyword+"_bin_"+str(redindex)
        if pardict["do_corr"]
        else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_" + keyword + "_bin_"+str(redindex)
    )
    fitlim = birdmodel_all[0].pardict["xfit_min"][0] if pardict["do_corr"] else birdmodel_all[0].pardict["xfit_max"][0]
    fitlimhex = birdmodel_all[0].pardict["xfit_min"][2] if pardict["do_corr"] else birdmodel_all[0].pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodel_all[0].pardict["fitfile"],
            dat_str,
            fitlim,
            fitlimhex,
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
        )
    )
    print(chainfile)

    # Set up the backend
    # with Pool(processes=ncpu) as pool:
    with Pool() as pool:

        # Initialize the sampler
        sampler = zeus.EnsembleSampler(nwalkers, nparams, func, pool=pool)

        old_tau = np.inf
        niter = 0
        converged = 0
        max_iter = 70000
        autocorr = np.empty(max_iter)
        counter = 0
        index = 0
        # sampler.run_mcmc(begin, 1000)
        for sample in sampler.sample(begin, iterations=max_iter, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = zeus.AutoCorrTime(sampler.get_chain())
            autocorr[index] = np.mean(tau)
            counter += 100
            
            # print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))

            # Check convergence
            # converged = np.all(tau * 100 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            
            if converged:
                print("Reached Auto-Correlation time goal: %d > 50 x %.3f" % (counter, autocorr[index]))
                break
            old_tau = tau
            index += 1
        # while ~converged:
        #     sampler.run_mcmc(begin, nsteps=20)
        #     tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
        #     converged = np.all(50 * tau < niter)
        #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
        #     old_tau = tau
        #     begin = None
        #     niter += 1000
        #     # print("Niterations/Max Iterations: ", niter, "/", 5000)
        #     print("Integrated ACT/Min Convergence Iterations: ", tau, "/", np.amax(50 * tau))
        #     # if niter >= 5000:
        #     #     break
        #     if converged:
        #         print("Reached Auto-Correlation time goal: %d > 50 x %.3f" % (niter, 50*tau))
        #         break

        # Remove burn-in and and save the samples
        tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True).T
        
        np.save(chainfile, samples)


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
            prior = lnprior(params, birdmodel_all, Shapefit)[0]
        except:
            prior = lnprior(params, birdmodel_all, Shapefit)
        like = 0.0
        # if ~np.isinf(prior):
        #     like = lnlike(params, birdmodel_all, fittingdata, plt, Shapefit)
        if prior > -1e-10:
            like = lnlike(params, birdmodel_all, fittingdata, plt, Shapefit)
    else:
        prior = lnprior(params, birdmodel_all, Shapefit)
        # index = np.where(~np.isinf(prior))[0]
        index = np.where(prior > -1e-10)[0]
        like = np.zeros(len(prior))
        if len(index) > 0:
            like[index] = lnlike(params[index], birdmodel_all, fittingdata, plt, Shapefit)
            
    return prior + like


def lnprior(params, birdmodel, Shaptfit):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
    
    
    
    if params.ndim == 1:
        params = params.reshape((-1, len(params)))
    params = params.T
    
    # print(params)
    # print(params)
    
    priors = np.zeros(np.shape(params[1]))
    
    # lower_bounds = birdmodel.valueref - birdmodel.pardict["template_order"] * birdmodel.delta
    # upper_bounds = birdmodel.valueref + birdmodel.pardict["template_order"] * birdmodel.delta
    
    if onebin == False:
        nz = len(pardict['z_pk'])
    else:
        nz = 1
        
    for i in range(nz):
        
        alpha_perp, alpha_par, fsigma8 = params[i*(cosmo_num + bias_num):3 + i*(cosmo_num + bias_num)]
        # alpha_perp, alpha_par, fsigma8 = 1.0, 1.0, 0.45014412
        # print(alpha_perp, alpha_par, fsigma8)
        if Shapefit == True:
            m = params[3 + i*(cosmo_num + bias_num)]
            # m = 0.0
            # print(m)
            # power = params[4 + i*(cosmo_num + bias_num)]
            # if 0.5 <= power <= 10.5:
            #     power_prior = 0.0
            # else:
            #     return -np.inf
            if vary_sigma8 == True:
                sigma8 = params[4 + i*(cosmo_num + bias_num)]
                if np.int16(pardict['vary_shapefit']) == 1:
                    a = params[5 + i*(cosmo_num + bias_num)]
                    kp = params[6 + i*(cosmo_num + bias_num)]
                    
                    # prior += np.where(np.logical_or(a < 0.1, a > 0.9), -1.0e30, 0.0)
                    # prior += np.where(np.logical_or(kp < 0.01, kp > 0.09), -1.0e30, 0.0)
                    
                    if (0.1 <= a <= 0.9):
                        a_prior = 0
                    else:
                        return -np.inf
                    
                    if (0.01 <= kp <= 0.09):
                        kp_prior = 0
                    else:
                        return -np.inf
            else:
                if np.int16(pardict['vary_shapefit']) == 1:
                    a = params[4 + i*(cosmo_num + bias_num)]
                    kp = params[5 + i*(cosmo_num + bias_num)]
                    
                    # prior += np.where(np.logical_or(a < 0.1, a > 0.9), -1.0e30, 0.0)
                    # prior += np.where(np.logical_or(kp < 0.01, kp > 0.09), -1.0e30, 0.0)
                    
                    if (0.1 <= a <= 0.9):
                        a_prior = 0
                    else:
                        return -np.inf
                    
                    if (0.01 <= kp <= 0.09):
                        kp_prior = 0
                    else:
                        return -np.inf
        else:
            if vary_sigma8 == True:
                sigma8 = params[3 + i*(cosmo_num + bias_num)]
                
        # prior += np.where(np.logical_or(alpha_perp < 0.9, alpha_perp > 1.1), -1.0e30, 0.0)
        # prior += np.where(np.logical_or(alpha_par < 0.9, alpha_par > 1.1), -1.0e30, 0.0)
        # prior += np.where(np.logical_or(fsigma8 < 0.0, fsigma8 > 1.0), -1.0e30, 0.0)
                
        # if (0.90 <= alpha_perp <= 1.10):
        #     alpha_perp_prior = 0.0
        # else:
        #     return -np.inf
        
        # if (0.90 <= alpha_par <= 1.10):
        #     alpha_par_prior = 0.0
        # else:
        #     return -np.inf
        
        # if vary_sigma8 == True:
        #     if (0.0 <= sigma8 <= 1.0):
        #         sigma8_prior = 0.0
        #         # if fsigma8/sigma8 > 0.95 or fsigma8/sigma8 < 0.6:
        #         #     return -np.inf
        #     else:
        #         return -np.inf
        #     # prior += np.where(np.logical_or(sigma8 < 0.0, sigma8 > 1.0), -1.0e30, 0.0)
        
        # if Shapefit == True:
        #     if m < -0.14 or m > 0.14:
        #         return -np.inf
        
        
            # prior += np.where(np.logical_or(m < 0.15, m > 0.15), -1.0e30, 0.0)
        
        # if fsigma8 < 0.0 or fsigma8 > 1.0:
        #     return -np.inf
        
        priors += np.where(np.logical_or(alpha_perp < 0.90, alpha_perp > 1.10), -1.0e30, 0.0)
        priors += np.where(np.logical_or(alpha_par < 0.90, alpha_par > 1.10), -1.0e30, 0.0)
        priors += np.where(np.logical_or(fsigma8 < 0.0, fsigma8 > 1.0), -1.0e30, 0.0)
        if Shapefit == True:
            priors += np.where(np.logical_or(m < -0.4, m > 0.4), -1.0e30, 0.0)
        
        # a_perp_prior = prior1(alpha_perp)
        # a_par_prior = prior2(alpha_par)
        # fsigma8_prior = prior3(fsigma8)
        # m_prior = prior4(m)
        
        # # print(np.log([a_perp_prior, a_par_prior, fsigma8_prior, m_prior]))
        
        
        # priors += np.log(a_perp_prior, out=a_perp_prior, where=a_perp_prior>0.0)
        # priors += np.log(a_par_prior, out=a_par_prior, where=a_par_prior>0.0)
        # priors += np.log(fsigma8_prior, out=fsigma8_prior, where=fsigma8_prior>0.0)
        # priors += np.log(m_prior, out=m_prior, where=m_prior>0.0)
        
        # var_inp = np.array([alpha_perp, alpha_par, fsigma8, m]).T
        # prior_new = prior_template(var_inp)
        # priors += np.log(prior_new, out=np.ones_like(alpha_perp)*1e-30, where=prior_new>0.0)
        
        # priors += prior_template(var_inp)
        
        bias = params[i*(cosmo_num + bias_num) + cosmo_num:(i+1)*(cosmo_num + bias_num)]
        # print(bias)
        if bias_num == 2:
            # if int(pardict['vary_c4']) == 1:
            #     b1, b2 = bias
            # else:
            #     b1, c2 = bias
            b1, c2 = bias
        elif bias_num == 3:
            # b1, c2, c4 = bias
            # b1, b2, b4 = bias
            b1, c2, c4 = bias
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = bias
            # b1, bp, b3, bd, cct, cr1, ce1, cemono, cequad = bias

            
        # Flat prior for b1
        # if b1 < 0.0 or b1 > 4.0:
        #     return -np.inf
        priors += np.where(np.logical_or(b1 < 0.0, b1 > 4.0), -1.0e30, 0.0)
        
        # b1_prior = np.where(np.logical_or(b1 < 0.0, b1 > 3.0), -np.inf, 0.0)

        # Flat prior for c2
        # c2_prior = -0.5 * 0.25 * c2 ** 2
        # if c2 < -10 or c2 > 10:
        #     return -np.inf
        
        # if b2 < -100.0 or b2 > 100.0:
        #     return -np.inf
        
        # if bp < 0.0 or bp > 5.0:
        #     return -np.inf
        
        # bp_prior = -0.5*(1.0/4.0)**2*bp**2
        # if int(pardict['vary_c4']) == 1:
        #     priors += np.where(np.logical_or(bp < -5.0, bp > 5.0), -1.0e30, 0.0)
        # else:
        #     priors += np.where(np.logical_or(c2 < -10.0, c2 > 10.0), -1.0e30, 0.0)
        priors += np.where(np.logical_or(c2 < -10.0, c2 > 10.0), -1.0e30, 0.0)
        
        if bias_num > 2.5:
        #     # if c4 < -10.0 or c4 > 10.0:
        #     #     return -np.inf
        #     # else:
        #     #     c4_prior = 0.0
        #     # c4_prior = -0.5 * 0.0625 * c4 ** 2
        #     # c4_prior = -0.5 * 0.25 * c4 ** 2
        #     # c4_prior = -0.5*(1.0/10.0)**2*c4**2
        #     # prior += -0.5 * c4 ** 2/2.0**2
        #     # c4_prior = 0.0
        #     # if b4 < -100.0 or b4 > 100.0:
        #     #     return -np.inf
            
        #     if bd < -20.0 or bd > 20.0:
        #         return -np.inf
        #     # bd_prior = -0.5*(1.0/2.0)**2*(bd+1.0)**2
        #     # if bp < -50.0 or bp > 50.0:
        #     #     return -np.inf
            # priors += np.where(np.logical_or(bd < -20.0, bd > 20.0), -1.0e30, 0.0)
            priors += np.where(np.logical_or(c4 < -10.0, c4 > 10.0), -1.0e30, 0.0)
   

                
        # print(priors)
        
    # if (0.75 <= alpha_perp <= 1.25):
    #     alpha_perp_prior = 0.0
    # else:
    #     return -np.inf
    
    # if (0.75 <= alpha_par <= 1.25):
    #     alpha_par_prior = 0.0
    # else:
    #     return -np.inf
    
    # priors = np.zeros(np.shape(params[1]))
    
    # lower_bounds = [0.90, 0.90, 0.0]
    # upper_bounds = [1.10, 1.10, 1.0]
    
    # if Shapefit == True:
    #     lower_bounds.append(-1.0)
    #     upper_bounds.append(1.0)
    # if vary_sigma8 == True:
    #     lower_bounds.append(0.0)
    #     upper_bounds.append(1.0)
        
    # lower_bounds = np.array(lower_bounds)
    # upper_bounds = np.array(upper_bounds)
    
    # if Shapefit == False:
    #     if vary_sigma8 == False:
    #         for i, param in enumerate([alpha_perp, alpha_par, fsigma8]):
    #             priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)
    #     else:
    #         for i, param in enumerate([alpha_perp, alpha_par, fsigma8, sigma8]):
    #             priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)
    # else:
    #     if vary_sigma8 == False:
    #         for i, param in enumerate([alpha_perp, alpha_par, fsigma8, m]):
    #             priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)
    #     else:
    #         for i, param in enumerate([alpha_perp, alpha_par, fsigma8, m, sigma8]):
    #             priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)
            
    # for i in range(nz):
    #     if birdmodel[0].pardict["do_marg"]:
    #         b1, c2 = params[-2 * (nz - i) : -2 * (nz - i - 1)] if i != nz - 1 else params[-2 * (nz - i) :]
    #         c4 = 0
    #         # b1, c2, c4 = params[-3 * (nz - i) : -3 * (nz - i - 1)] if i != nz - 1 else params[-3 * (nz - i) :]
    #     else:
    #         b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = (
    #             params[-10 * (nz - i) : -10 * (nz - i - 1)] if i != nz - 1 else params[-10 * (nz - i) :]
    #         )


    # alpha_perp, alpha_par, f = birdmodel.valueref[:3]

    # Flat priors for alpha_perp, alpha_par f and sigma8
    # if np.any(np.less([alpha_perp, alpha_par, f, birdmodel.valueref[3]], lower_bounds)) or np.any(
    #    np.greater([alpha_perp, alpha_par, f, birdmodel.valueref[3]], upper_bounds)
    # ):
    #    return -np.inf

    # # Flat prior for b1
    # if b1 < 0.0 or b1 > 4.0:
    #     return -np.inf
    
    # # b1_prior = np.where(np.logical_or(b1 < 0.0, b1 > 3.0), -np.inf, 0.0)

    # # Flat prior for c2
    # # c2_prior = -0.5 * 0.25 * c2 ** 2
    # if c2 < -1000.0 or c2 > 1000.0:
    #     return -np.inf

    # Gaussian prior for c4
    # c4_prior = -0.5 * 0.25 * c4 ** 2
    
    return priors

    # if birdmodel[0].pardict["do_marg"]:

    #     if bias_num > 2.5:
    #         # return c4_prior
    #         return 0.0
    #     else:
    #         return 0.0
    #     # # return 0.0

    # else:
    #     # # Gaussian prior for c4
    #     # c4_prior = -0.5 * 0.25 * c4 ** 2
    #     # # Gaussian prior for b3 of width 2 centred on 0
    #     # b3_prior = -0.5 * 0.25 * b3 ** 2

    #     # # Gaussian prior for cct of width 2 centred on 0
    #     # cct_prior = -0.5 * 0.25 * cct ** 2

    #     # # Gaussian prior for cr1 of width 4 centred on 0
    #     # cr1_prior = -0.5 * cr1 ** 2/10.0**2

    #     # # Gaussian prior for cr1 of width 4 centred on 0
    #     # cr2_prior = -0.5 * cr2 ** 2/10.0**2

    #     # # Gaussian prior for ce1 of width 2 centred on 0
    #     # ce1_prior = -0.5 * ce1 ** 2/10.0**2

    #     # # Gaussian prior for cemono of width 2 centred on 0
    #     # cemono_prior = -0.5 * cemono ** 2/10.0**2

    #     # # Gaussian prior for cequad of width 2 centred on 0
    #     # cequad_prior = -0.5 * cequad ** 2/10.0**2

    #     # # # Gaussian prior for bnlo of width 2 centred on 0
    #     # # bnlo_prior = -0.5 * 0.25 * bnlo ** 2

    #     return (
    #         0.0
    #         # c4_prior
    #         # + b3_prior
    #         # + cct_prior
    #         # + cr1_prior
    #         # + cr2_prior
    #         # + ce1_prior
    #         # + cemono_prior
    #         # + cequad_prior
    #     )


# def lnlike(params, birdmodel, fittingdata, plt, Shapefit):
    
#     # if np.shape(params)[0] > 1.5:
#     #     params = np.array(params).T
#     # else:
#     #     params = np.reshape(np.array(params), (1, np.shape(params)[1]))
    
#     onedflag = False
#     if params.ndim == 1:
#         onedflag = True
#         params = params.reshape((-1, len(params)))
#     params = params.T
    
#     ln10As, h, omega_cdm, omega_b = params[:4]
    
#     nz = len(pardict['z_pk'])
    
#     #The birdmodels are built with respect to the shot noise of the first
#     #data set. 
    
#     Picount = 0
#     P_models, Plins, Ploops = [], [], []
#     nmarg = len(birdmodel_all[0].eft_priors)
#     Pi_full = np.zeros((nz * len(birdmodel_all[0].eft_priors), len(fittingdata.data["fit_data"]), len(ln10As)))
#     shot_noise_fid = np.float64(fittingdata.data["shot_noise"])[0]
#     for i in range(nz):
    
#         shot_noise = np.float64(fittingdata.data["shot_noise"])[i]
#         shot_noise_ratio = shot_noise/shot_noise_fid
        
#         if birdmodel_all[0].pardict["do_marg"]:
#             b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
#             b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
#             margb = np.zeros(np.shape(params)[1])
#             bs = [params[-3], b2, margb, b4, margb, margb, margb, margb, margb, margb]
#         else:
#             b2 = (params[-10] + params[-8]) / np.sqrt(2.0)
#             b4 = (params[-10] - params[-8]) / np.sqrt(2.0)
#             bs = [
#                 params[-11],
#                 b2,
#                 params[-9],
#                 b4,
#                 params[-7],
#                 params[-6],
#                 params[-5],
#                 params[-4] * shot_noise_ratio,
#                 params[-3] * shot_noise_ratio,
#                 params[-2] * shot_noise_ratio,
#                 params[-1],
#             ]
        
#         if Shapefit == False:
#             # Get the bird model
#             Plin, Ploop = [], []
#             for i in range(np.shape(params)[1]):
#                 Plin_i, Ploop_i = birdmodel_all[i].modify_template(params[:3][:, i], fittingdata=fittingdata)
#                 Plin.append(Plin_i)
#                 Ploop.append(Ploop_i)
            
#             Plin = np.array(Plin)
#             Ploop = np.array(Ploop)
            
#             Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
#             Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
#         else:
#             # Get the bird model
#             Plin, Ploop = [], []
#             for i in range(np.shape(params)[1]):
#                 Plin_i, Ploop_i = birdmodel_all[i].modify_template(params[:3][:, i], fittingdata=fittingdata, factor_m=params[3, i])
#                 Plin.append(Plin_i)
#                 Ploop.append(Ploop_i)
            
#             Plin = np.array(Plin)
#             Ploop = np.array(Ploop)
            
#             Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
#             Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
        
#         # Plin, Ploop = birdmodel.modify_template(params[:3])
#         P_model_lin, P_model_loop, P_model_interp = birdmodel_all[i].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][0], Pstl=Pstl_all[i])
#         Pi = birdmodel_all[i].get_Pi_for_marg(Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][0], Pstl_in = Pstl_all[i])
        
#         Plins.append(Plin)
#         Ploops.append(Ploop)
#         P_models.append(P_model_interp)
#         Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + fittingdata.data["ndata"][i]] = Pi
#         Picount += fittingdata.data["ndata"][i]
    
#     P_model = np.concatenate(P_models)
#     # print(np.shape(Pi))
#     # chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
    
#     # Sort out the different flavours of marginalisation
#     if birdmodel_all[0].pardict["do_marg"] == 0:
#         chi_squared = birdmodel_all[0].compute_chi2(P_model, fittingdata.data)
#         chi_squared_print = chi_squared
#     elif birdmodel_all[0].pardict["do_marg"]:
#         P_models, P_model_lins, P_model_loops = [], [], []
#         if plt is not None or birdmodel.pardict["do_marg"] > 1:
#             bs_analytic = birdmodel_all[0].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data)
#             for i in range(1):
#                 counter = -3 * (1 - i)
#                 b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
#                 b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
#                 # counter = -2 * (nz - i)
#                 # b2 = (params[counter + 1]) / np.sqrt(2.0)
#                 # b4 = (params[counter + 1]) / np.sqrt(2.0)
#                 bs = np.array(
#                     [
#                         params[counter],
#                         b2,
#                         bs_analytic[7 * i],
#                         b4,
#                         bs_analytic[7 * i + 1],
#                         bs_analytic[7 * i + 2],
#                         bs_analytic[7 * i + 3],
#                         bs_analytic[7 * i + 4] * float(fittingdata.data["shot_noise"][i]),
#                         bs_analytic[7 * i + 5] * float(fittingdata.data["shot_noise"][i]),
#                         bs_analytic[7 * i + 6] * float(fittingdata.data["shot_noise"][i]),
#                     ]
#                 )
#                 P_model_lin, P_model_loop, P_model_interp = birdmodel.compute_model(
#                     bs, Plin, Ploop, fittingdata.data["x_data"][i]
#                 )
#                 P_models.append(P_model_interp)
#                 P_model_lins.append(P_model_lin)
#                 P_model_loops.append(P_model_loop)
#         if birdmodel.pardict["do_marg"] == 1:
#             chi_squared = birdmodel_all[0].compute_chi2_marginalised(P_model, Pi_full, fittingdata.data)
#             if plt is not None:
#                 chi_squared_print = birdmodel_all[0].compute_chi2(np.concatenate(P_models), fittingdata.data)
#         else:
#             chi_squared = birdmodel_all[0].compute_chi2(np.concatenate(P_models), fittingdata.data)
#             chi_squared_print = chi_squared

#     if plt is not None:
#         chi_squared_print = chi_squared
#         if birdmodel.pardict["do_marg"]:
#             bs_analytic = birdmodel.compute_bestfit_analytic(Pi, fittingdata.data, P_model_interp)
#             pardict["do_marg"] = 0
#             b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
#             b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
#             bs = [
#                 params[-3],
#                 b2,
#                 bs_analytic[0],
#                 b4,
#                 bs_analytic[1],
#                 bs_analytic[2],
#                 bs_analytic[3],
#                 bs_analytic[4] * fittingdata.data["shot_noise"],
#                 bs_analytic[5] * fittingdata.data["shot_noise"],
#                 bs_analytic[6] * fittingdata.data["shot_noise"],
#                 bs_analytic[7],
#             ]
#             P_model_lin, P_model_loop, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
#             chi_squared_print = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
#             pardict["do_marg"] = 1
#         update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt)
#         if np.random.rand() < 0.1:
#             print(params, chi_squared_print)

#     if onedflag:
#         return -0.5 * chi_squared[0]
#     else:
#         return -0.5 * chi_squared

# def lnlike(params, birdmodel, fittingdata, plt, Shapefit):
    
#     # if np.shape(params)[0] > 1.5:
#     #     params = np.array(params).T
#     # else:
#     #     params = np.reshape(np.array(params), (1, np.shape(params)[1]))
    
#     onedflag = False
#     if params.ndim == 1:
#         onedflag = True
#         params = params.reshape((-1, len(params)))
#     params = params.T
    
#     # print(np.shape(params))
#     # raise Exception('Stop')
    
#     #The birdmodels are built with respect to the shot noise of the first
#     #data set. 
#     if onebin == False:
#         nz = len(pardict['z_pk'])
#     else:
#         nz = 1
    
#     Picount = 0
#     P_models, Plins, Ploops = [], [], []
#     P_model_lins, P_model_loops = [], []
#     nmarg = len(birdmodel[0].eft_priors)
     
#     Pi_full = np.zeros((nz * len(birdmodel[0].eft_priors), len(fittingdata.data["fit_data"]), len(params[0])))
        
#     for i in range(nz):
        
#         if onebin == True:
#             if one_nz == False:
#                 model_index = redindex
#             else:
#                 model_index = i
#         else:
#             model_index = i
            
#         if one_nz == True:
#             shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
#         else:
#             shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[model_index]/shot_noise_fid
            
#         alpha_perp, alpha_par, fsigma8 = params[i*(cosmo_num + bias_num):3 + i*(cosmo_num + bias_num)]
#         if Shapefit == True:
#             m = params[3 + i*(cosmo_num + bias_num)]
#             # power = params[4 + i*(cosmo_num + bias_num)]
#             if vary_sigma8 == True:
#                 sigma8 = params[4 + i*(cosmo_num + bias_num)]
#                 if np.int16(pardict['vary_shapefit']) == 1:
#                     a = params[5 + i*(cosmo_num + bias_num)]
#                     kp = params[6 + i*(cosmo_num + bias_num)]
#             else:
#                 if np.int16(pardict['vary_shapefit']) == 1:
#                     a = params[4 + i*(cosmo_num + bias_num)]
#                     kp = params[5 + i*(cosmo_num + bias_num)]
            
                
#         else:
#             if vary_sigma8 == True:
#                 sigma8 = params[3 + i*(cosmo_num + bias_num)]
                
#         bias = params[i*(cosmo_num + bias_num) + cosmo_num:(i+1)*(cosmo_num + bias_num)]
#         if bias_num == 2:
#             if int(pardict['vary_c4']) == 1:
#                 b1, b2 = bias
#                 # b4 = -0.85251299*b2 + 1.76418598
#                 b4 = -b2
#             else:
#                 b1, c2 = bias
#         elif bias_num == 3:
#             # b1, bp, bd = bias
#             b1, b2, b4 = bias
#             # b1, c2, c4 = bias
#         else:
#             b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = bias
            
#         if birdmodel[0].pardict["do_marg"]:
#             # b2 = (params[-1]) / np.sqrt(2.0)
#             # b4 = (params[-1]) / np.sqrt(2.0)
#             # # b2 = (params[-2]) / np.sqrt(2.0)
#             # # b4 = (params[-2]) / np.sqrt(2.0)
#             # # b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
#             # # b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            
#             # margb = np.zeros(np.shape(params)[1])
#             # bs = [params[-2], b2, margb, b4, margb, margb, margb, margb, margb, margb]
#             # # bs = [params[-3], b2, margb, b4, margb, margb, margb, margb, margb, margb]
            
#             # counter = -2 * (nz - i)
#             # b2 = (params[counter + 1]) / np.sqrt(2.0)
#             # b4 = (params[counter + 1]) / np.sqrt(2.0)
#             if bias_num == 2 and int(pardict['vary_c4']) == 0:
#                 b2 = c2/np.sqrt(2.0)
#                 b4 = c2/np.sqrt(2.0)
                
#             elif bias_num == 3:
#                 # b4 = bp/(bd + 1.0)
#                 # b2 = bd*bp/(bd+1.0)
                
#                 # b4 = bd
#                 # b2 = bp - bd
#                 b2 = b2
#                 b4 = b4
                
#                 # b2 = (c2+c4)/np.sqrt(2.0)
#                 # b4 = (c2-c4)/np.sqrt(2.0)
#             # else:
#             #     b2 = (c2+c4)/np.sqrt(2.0)
#             #     b4 = (c2-c4)/np.sqrt(2.0)
            
#             margb = np.zeros(np.shape(params)[1])
#             bs = np.array(
#                 [
#                     b1,
#                     b2,
#                     margb,
#                     b4,
#                     margb,
#                     margb,
#                     margb,
#                     margb,
#                     margb,
#                     margb,
#                 ]
#             )
            
            
            
#         else:
#             # # b2 = (params[-8]) / np.sqrt(2.0)
#             # # b4 = (params[-8]) / np.sqrt(2.0)
#             # b2 = (params[-9] + params[-7]) / np.sqrt(2.0)
#             # b4 = (params[-9] - params[-7]) / np.sqrt(2.0)
#             # bs = [
#             #     params[-10],
#             #     b2,
#             #     params[-8],
#             #     b4,
#             #     params[-6],
#             #     params[-5],
#             #     params[-4],
#             #     # params[-3] * shot_noise_ratio,
#             #     params[-3],
#             #     params[-2] * shot_noise_ratio,
#             #     params[-1] * shot_noise_ratio,
#             # ]
            
#             # counter = -10 * (nz - i)
#             # b2 = (params[counter + 1] + params[counter + 3]) / np.sqrt(2.0)
#             # b4 = (params[counter + 1] - params[counter + 3]) / np.sqrt(2.0)
            
#             b2 = (c2+c4)/np.sqrt(2.0)
#             b4 = (c2-c4)/np.sqrt(2.0)
            
#             bs = np.array(
#                 [
#                     b1,
#                     b2,
#                     b3,
#                     b4,
#                     cct,
#                     cr1,
#                     cr2,
#                     # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
#                     # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
#                     # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
#                     ce1 * shot_noise_ratio,
#                     cemono * shot_noise_ratio,
#                     cequad * shot_noise_ratio,
#                 ]
#             )
        
#         if Shapefit == False:
#             # Get the bird model
#             Plin, Ploop = [], []
#             for j in range(np.shape(params)[1]):
#                 if vary_sigma8 == False:
#                     Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, redindex=redindex, one_nz = one_nz, resum = resum)
#                 else:
#                     Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum)
#                 Plin.append(Plin_i)
#                 Ploop.append(Ploop_i)
            
#             Plin = np.array(Plin)
#             Ploop = np.array(Ploop)
            
#             Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
#             Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
#         else:
#             # Get the bird model
#             Plin, Ploop = [], []
#             for j in range(np.shape(params)[1]):
#                 if vary_sigma8 == False:
#                     if np.int16(pardict['vary_shapefit']) == 0:
#                         Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, resum = resum)
#                     # Plin_i, Ploop_i = birdmodel.modify_template([1.0, 1.0, birdmodel.fN*birdmodel.sigma8], fittingdata=fittingdata, factor_m=0.0, redindex=redindex, one_nz = one_nz)
#                     else:
#                         Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, resum = resum, 
#                                                                                  factor_a = a, factor_kp = kp)
    
#                 else:
#                     if np.int16(pardict['vary_shapefit']) == 0:
#                         Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum)
#                     else:
#                         Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum, 
#                                                                                  factor_a = a, factor_kp = kp)
#                 Plin.append(Plin_i)
#                 Ploop.append(Ploop_i)
            
#             Plin = np.array(Plin)
#             Ploop = np.array(Ploop)
            
#             Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
#             Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
        
#         # Plin, Ploop = birdmodel.modify_template(params[:3])
#         P_model_lin, P_model_loop, P_model_interp = birdmodel[model_index].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][model_index])
#         Pi = birdmodel[model_index].get_Pi_for_marg(Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][model_index])
        
#         Plins.append(Plin)
#         Ploops.append(Ploop)
#         P_models.append(P_model_interp)
#         P_model_lins.append(P_model_lin)
#         P_model_loops.append(P_model_loop)
#         Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + fittingdata.data["ndata"][model_index]] = Pi
#         Picount += fittingdata.data["ndata"][model_index]

#     P_model = np.concatenate(P_models)
    
#     # print(np.shape(Pi))
#     # chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
    
#     # Sort out the different flavours of marginalisation
#     if birdmodel[0].pardict["do_marg"] == 0:
#         chi_squared = birdmodel[model_index].compute_chi2(P_model, fittingdata.data)
#         chi_squared_print = chi_squared
#     elif birdmodel[0].pardict["do_marg"]:
#         P_models, P_model_lins, P_model_loops = [], [], []
#         if plt is not None or birdmodel[0].pardict["do_marg"] > 1:
#             bs_analytic = birdmodel[model_index].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin)
#             for i in range(1):
#                 counter = -3 * (1 - i)
#                 b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
#                 b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
#                 # counter = -2 * (nz - i)
#                 # b2 = (params[counter + 1]) / np.sqrt(2.0)
#                 # b4 = (params[counter + 1]) / np.sqrt(2.0)
#                 bs = np.array(
#                     [
#                         params[counter],
#                         b2,
#                         bs_analytic[7 * i],
#                         b4,
#                         bs_analytic[7 * i + 1],
#                         bs_analytic[7 * i + 2],
#                         bs_analytic[7 * i + 3],
#                         bs_analytic[7 * i + 4],
#                         bs_analytic[7 * i + 5] * shot_noise_ratio,
#                         bs_analytic[7 * i + 6] * shot_noise_ratio,
#                     ]
#                 )
#                 P_model_lin, P_model_loop, P_model_interp = birdmodel.compute_model(
#                     bs, Plin, Ploop, fittingdata.data["x_data"][i]
#                 )
#                 P_models.append(P_model_interp)
#                 P_model_lins.append(P_model_lin)
#                 P_model_loops.append(P_model_loop)
#         if birdmodel[0].pardict["do_marg"] == 1:
#             # chi_squared = birdmodel[model_index].compute_chi2_marginalised(P_model, Pi_full, fittingdata.data, onebin = onebin)
#             bs_analytic = birdmodel[model_index].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin)
#             P_models, Plins, Ploops = [], [], []
#             P_model_lins, P_model_loops = [], []
#             for i in range(nz):
                
#                 if onebin == True:
#                     if one_nz == False:
#                         model_index = redindex
#                     else:
#                         model_index = i
#                 else:
#                     model_index = i
                    
#                 if one_nz == True:
#                     shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
#                 else:
#                     shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[model_index]/shot_noise_fid
                    
#                 alpha_perp, alpha_par, fsigma8 = params[i*(cosmo_num + bias_num):3 + i*(cosmo_num + bias_num)]
#                 if Shapefit == True:
#                     m = params[3 + i*(cosmo_num + bias_num)]
#                     # power = params[4 + i*(cosmo_num + bias_num)]
#                     if vary_sigma8 == True:
#                         sigma8 = params[4 + i*(cosmo_num + bias_num)]
#                         if np.int16(pardict['vary_shapefit']) == 1:
#                             a = params[5 + i*(cosmo_num + bias_num)]
#                             kp = params[6 + i*(cosmo_num + bias_num)]
#                     else:
#                         if np.int16(pardict['vary_shapefit']) == 1:
#                             a = params[4 + i*(cosmo_num + bias_num)]
#                             kp = params[5 + i*(cosmo_num + bias_num)]
                    
                        
#                 else:
#                     if vary_sigma8 == True:
#                         sigma8 = params[3 + i*(cosmo_num + bias_num)]
                        
#                 bias = params[i*(cosmo_num + bias_num) + cosmo_num:(i+1)*(cosmo_num + bias_num)]
#                 if bias_num == 2:
#                     if int(pardict['vary_c4']) == 1:
#                         b1, b2 = bias
#                         # b4 = -0.85251299*b2 + 1.76418598
#                         b4 = -b2
#                     else:
#                         b1, c2 = bias
#                 elif bias_num == 3:
#                     # b1, bp, bd = bias
#                     b1, b2, b4 = bias
#                 else:
#                     b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = bias
                    
#                 # b2 = bp - bd
#                 # b4 = bd
                
#                 if bias_num == 2 and int(pardict['vary_c4']) == 0:
#                     b2 = c2/np.sqrt(2.0)
#                     b4 = c2/np.sqrt(2.0)
                    
#                 elif bias_num == 3:
#                     # b4 = bp/(bd + 1.0)
#                     # b2 = bd*bp/(bd+1.0)
                    
#                     # b4 = bd
#                     # b2 = bp - bd
#                     b2 = b2
#                     b4 = b4
                
#                 bs = np.array(
#                     [
#                         b1,
#                         b2,
#                         bs_analytic[0],
#                         b4,
#                         bs_analytic[1],
#                         bs_analytic[2],
#                         bs_analytic[3],
#                         # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
#                         # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
#                         # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
#                         bs_analytic[4] * shot_noise_ratio,
#                         bs_analytic[5] * shot_noise_ratio,
#                         bs_analytic[6] * shot_noise_ratio,
#                     ]
#                 )
                
#                 if Shapefit == False:
#                     # Get the bird model
#                     Plin, Ploop = [], []
#                     for j in range(np.shape(params)[1]):
#                         if vary_sigma8 == False:
#                             Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, redindex=redindex, one_nz = one_nz, resum = resum)
#                         else:
#                             Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum)
#                         Plin.append(Plin_i)
#                         Ploop.append(Ploop_i)
                    
#                     Plin = np.array(Plin)
#                     Ploop = np.array(Ploop)
                    
#                     Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
#                     Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
#                 else:
#                     # Get the bird model
#                     Plin, Ploop = [], []
#                     for j in range(np.shape(params)[1]):
#                         if vary_sigma8 == False:
#                             if np.int16(pardict['vary_shapefit']) == 0:
#                                 Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, resum = resum)
#                             # Plin_i, Ploop_i = birdmodel.modify_template([1.0, 1.0, birdmodel.fN*birdmodel.sigma8], fittingdata=fittingdata, factor_m=0.0, redindex=redindex, one_nz = one_nz)
#                             else:
#                                 Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, resum = resum, 
#                                                                                           factor_a = a, factor_kp = kp)
            
#                         else:
#                             if np.int16(pardict['vary_shapefit']) == 0:
#                                 Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum)
#                             else:
#                                 Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum, 
#                                                                                           factor_a = a, factor_kp = kp)
#                         Plin.append(Plin_i)
#                         Ploop.append(Ploop_i)
                    
#                     Plin = np.array(Plin)
#                     Ploop = np.array(Ploop)
                    
#                     Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
#                     Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
                
#                 # Plin, Ploop = birdmodel.modify_template(params[:3])
#                 P_model_lin, P_model_loop, P_model_interp = birdmodel[model_index].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][model_index])
#                 Pi = birdmodel[model_index].get_Pi_for_marg(Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][model_index])
                
#                 Plins.append(Plin)
#                 Ploops.append(Ploop)
#                 P_models.append(P_model_interp)
#                 P_model_lins.append(P_model_lin)
#                 P_model_loops.append(P_model_loop)

#             P_model = np.concatenate(P_models)
            
#             chi_squared = birdmodel[model_index].compute_chi2(P_model, fittingdata.data)
                    
                
            
#             if plt is not None:
#                 chi_squared_print = birdmodel[model_index].compute_chi2(np.concatenate(P_models), fittingdata.data)
#                 # bs_analytic = birdmodel.compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin)
#                 # pardict["do_marg"] = 0
#                 # b2 = (params[-1]) / np.sqrt(2.0)
#                 # b4 = (params[-1]) / np.sqrt(2.0)
#                 # bs = [
#                 #     params[-2],
#                 #     b2,
#                 #     bs_analytic[0],
#                 #     b4,
#                 #     bs_analytic[1],
#                 #     bs_analytic[2],
#                 #     bs_analytic[3],
#                 #     bs_analytic[4],
#                 #     bs_analytic[5] * shot_noise_ratio,
#                 #     bs_analytic[6] * shot_noise_ratio,
#                 # ]
#                 # P_model_lin, P_model_loop, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][redindex])
#                 # chi_squared_print = birdmodel.compute_chi2(P_model_interp, fittingdata.data)
                
#                 # print(bs_analytic)
                
#                 # pardict["do_marg"] = 1
                
#                 # return chi_squared_print
#         else:
#             chi_squared = birdmodel[model_index].compute_chi2(P_model, fittingdata.data)
#             chi_squared_print = chi_squared

#     if plt is not None:
#         chi_squared_print = chi_squared
#         if birdmodel[0].pardict["do_marg"]:
#             bs_analytic = birdmodel[model_index].compute_bestfit_analytic(P_model, fittingdata.data, P_model_interp, onebin=onebin)
#             pardict["do_marg"] = 0
#             b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
#             b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
#             bs = [
#                 params[-3],
#                 b2,
#                 bs_analytic[0],
#                 b4,
#                 bs_analytic[1],
#                 bs_analytic[2],
#                 bs_analytic[3],
#                 bs_analytic[4] * fittingdata.data["shot_noise"],
#                 bs_analytic[5] * fittingdata.data["shot_noise"],
#                 bs_analytic[6] * fittingdata.data["shot_noise"],
#             ]
#             P_model_lin, P_model_loop, P_model_interp = birdmodel[model_index].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
#             chi_squared_print = birdmodel[model_index].compute_chi2(P_model_interp, P_model, fittingdata.data)
#             pardict["do_marg"] = 1
#         update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt)
#         if np.random.rand() < 0.1:
#             print(params, chi_squared_print)
#     else:
#         if np.random.rand() < 0.001:
#             print(params[:, 0], chi_squared[0], len(fittingdata.data["fit_data"]), shot_noise_ratio)

#     if onedflag:
#         return -0.5 * chi_squared[0]
#     else:
#         return -0.5 * chi_squared

def lnlike(params, birdmodel, fittingdata, plt, Shapefit):
    
    # if np.shape(params)[0] > 1.5:
    #     params = np.array(params).T
    # else:
    #     params = np.reshape(np.array(params), (1, np.shape(params)[1]))
    
    onedflag = False
    if params.ndim == 1:
        onedflag = True
        params = params.reshape((-1, len(params)))
    params = params.T
    
    # print(np.shape(params))
    # raise Exception('Stop')
    
    #The birdmodels are built with respect to the shot noise of the first
    #data set. 
    if onebin == False:
        nz = len(pardict['z_pk'])
    else:
        nz = 1
    
    Picount = 0
    P_models, Plins, Ploops = [], [], []
    P_model_lins, P_model_loops = [], []
    # nmarg = len(birdmodel[0].eft_priors)
    # nmarg = 7
    
    if MinF == False:
        if int(pardict['do_hex']) == 1:
            nmarg = 7
        else:
            if pardict['prior'] == 'BOSS_MaxF':
                nmarg = 6
            else:
                nmarg = 5

    else: 
        if int(pardict['do_hex']) == 1:
            nmarg = 5
        else:
            nmarg = 4
     
    Pi_full = np.zeros((nz * nmarg, len(fittingdata.data["fit_data"]), len(params[0])))
        

    i = 0
    model_index = 0
        
    if one_nz == True:
        shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
    else:
        shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[model_index]/shot_noise_fid
        
    alpha_perp, alpha_par, fsigma8 = params[i*(cosmo_num + bias_num):3 + i*(cosmo_num + bias_num)]
    # alpha_perp, alpha_par, fsigma8 = [1.0], [1.0], [0.45014412]
    if Shapefit == True:
        m = params[3 + i*(cosmo_num + bias_num)]
        # m = [0]
        # power = params[4 + i*(cosmo_num + bias_num)]
        if vary_sigma8 == True:
            sigma8 = params[4 + i*(cosmo_num + bias_num)]
            if np.int16(pardict['vary_shapefit']) == 1:
                a = params[5 + i*(cosmo_num + bias_num)]
                kp = params[6 + i*(cosmo_num + bias_num)]
        else:
            if np.int16(pardict['vary_shapefit']) == 1:
                a = params[4 + i*(cosmo_num + bias_num)]
                kp = params[5 + i*(cosmo_num + bias_num)]
        
            
    else:
        if vary_sigma8 == True:
            sigma8 = params[3 + i*(cosmo_num + bias_num)]
            
    bias = params[i*(cosmo_num + bias_num) + cosmo_num:(i+1)*(cosmo_num + bias_num)]
    # if bias_num == 2:
    #     if int(pardict['vary_c4']) == 1:
    #         b1, b2 = bias
    #         # b4 = -0.85251299*b2 + 1.76418598
    #         b4 = -b2
    #     else:
    #         b1, c2 = bias
    # elif bias_num == 3:
    #     b1, bp, bd = bias
    #     # b1, b2, b4 = bias
    #     # b1, c2, c4 = bias
    # else:
    #     # b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = bias
    #     b1, bp, b3, bd, cct, cr1, ce1, cemono, cequad = bias

        
    if birdmodel[0].pardict["do_marg"]:
        # b2 = (params[-1]) / np.sqrt(2.0)
        # b4 = (params[-1]) / np.sqrt(2.0)
        # # b2 = (params[-2]) / np.sqrt(2.0)
        # # b4 = (params[-2]) / np.sqrt(2.0)
        # # b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
        # # b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
        
        # margb = np.zeros(np.shape(params)[1])
        # bs = [params[-2], b2, margb, b4, margb, margb, margb, margb, margb, margb]
        # # bs = [params[-3], b2, margb, b4, margb, margb, margb, margb, margb, margb]
        
        # counter = -2 * (nz - i)
        # b2 = (params[counter + 1]) / np.sqrt(2.0)
        # b4 = (params[counter + 1]) / np.sqrt(2.0)
        # if bias_num == 2 and int(pardict['vary_c4']) == 0:
        #     b2 = c2/np.sqrt(2.0)
        #     b4 = c2/np.sqrt(2.0)
            
        # elif bias_num == 3:
        #     # b4 = bp/(bd + 1.0)
        #     # b2 = bd*bp/(bd+1.0)
            
        #     b4 = bd
        #     b2 = (bp - bd)/0.86
        #     # b2 = b2
        #     # b4 = b4
        #     # b2 = bd
        #     # b4 = bp - 0.86*bd
        #     # b2 = (bp+bd)/(43.0/50.0 +  50.0/43.0)
        #     # b4 = bp - 0.86*b2
            
        #     # b2 = (c2+c4)/np.sqrt(2.0)
        #     # b4 = (c2-c4)/np.sqrt(2.0)
        # # else:
        # #     b2 = (c2+c4)/np.sqrt(2.0)
        # #     b4 = (c2-c4)/np.sqrt(2.0)
        
        # margb = np.zeros(np.shape(params)[1])
        
        margb = np.zeros(np.shape(params)[1])
        
        if MinF == True:
            b1, b2_SPT = bias
            b2 = [1.0]
            b3 = b1 + 15.0*(-2.0/7.0*(b1-1.0))+6.0*23.0/42.0*(b1-1.0)
            b4 = 0.5*(b2_SPT) + b1 - 1.0
        else:
            if int(pardict['vary_c4']) == 1:
                b1, c2, c4 = bias
                b2 = (c2+c4)/np.sqrt(2.0)
                b4 = (c2-c4)/np.sqrt(2.0)
            else:
                b1, c2 = bias
                b2 = (c2)/np.sqrt(2.0)
                b4 = (c2)/np.sqrt(2.0)
            b3 = margb
            
        # print(np.shape(b1), np.shape(margb))
        
        bs = np.array(
            [
                b1,
                b2,
                b3,
                b4,
                margb,
                margb,
                margb,
                margb,
                margb,
                margb,
            ]
        )
        
        
        
    else:
        
            
        # if bias_num == 2 and int(pardict['vary_c4']) == 0:
        #     b2 = c2/np.sqrt(2.0)
        #     b4 = c2/np.sqrt(2.0)
            
        # elif bias_num == 3:
        #     # b4 = bp/(bd + 1.0)
        #     # b2 = bd*bp/(bd+1.0)
            
        # b4 = bd
        # b2 = (bp - bd)/0.86
        
        # # b2 = (c2+c4)/np.sqrt(2.0)
        # # b4 = (c2-c4)/np.sqrt(2.0)
        # cr2 = np.zeros(np.shape(params)[1])
        b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = bias
        bs = np.array(
            [
                b1,
                b2,
                b3,
                b4,
                cct,
                cr1,
                cr2,
                # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
                # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
                # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
                ce1 * shot_noise_ratio,
                cemono * shot_noise_ratio,
                cequad * shot_noise_ratio,
            ]
        )
        
        # margb = np.zeros(np.shape(params)[1])
        # bs = np.array(
        #     [
        #         margb,
        #         margb,
        #         margb,
        #         margb,
        #         margb,
        #         margb,
        #         margb,
        #         # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
        #         # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
        #         # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
        #         margb * shot_noise_ratio,
        #         margb * shot_noise_ratio,
        #         margb * shot_noise_ratio,
        #     ]
        # )
    
    if Shapefit == False:
        # Get the bird model
        Plin, Ploop = [], []
        for j in range(np.shape(params)[1]):
            if vary_sigma8 == False:
                Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, redindex=redindex, one_nz = one_nz, resum = resum)
            else:
                Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum)
            Plin.append(Plin_i)
            Ploop.append(Ploop_i)
        
        Plin = np.array(Plin)
        Ploop = np.array(Ploop)
        
        Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
        Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
    else:
        # Get the bird model
        Plin, Ploop = [], []
        for j in range(np.shape(params)[1]):
            if vary_sigma8 == False:
                if np.int16(pardict['vary_shapefit']) == 0:
                    Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, resum = resum, 
                                                                             factor_a = np.float64(pardict['factor_a']), factor_kp = np.float64(pardict['factor_kp']))
                    
                # Plin_i, Ploop_i = birdmodel.modify_template([1.0, 1.0, birdmodel.fN*birdmodel.sigma8], fittingdata=fittingdata, factor_m=0.0, redindex=redindex, one_nz = one_nz)
                else:
                    Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, resum = resum, 
                                                                              factor_a = a, factor_kp = kp)

            else:
                if np.int16(pardict['vary_shapefit']) == 0:
                    Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum)
                else:
                    Plin_i, Ploop_i = birdmodel[model_index].modify_template([alpha_perp[j], alpha_par[j], fsigma8[j]], fittingdata=fittingdata, factor_m=m[j], redindex=redindex, one_nz = one_nz, sigma8 = sigma8[j], resum = resum, 
                                                                              factor_a = a, factor_kp = kp)
            
            # print(np.shape(Plin_i), np.shape(Ploop_i))
            # Plin_i = np.einsum("abc, dc -> abd", Plin_i, birdmodel[model_index].correlator.projection.xbin_mat)
            # Ploop_i = np.einsum("abc, dc -> abd", Ploop_i, birdmodel[model_index].correlator.projection.xbin_mat)
            Plin.append(Plin_i)
            Ploop.append(Ploop_i)
        
        Plin = np.array(Plin)
        Ploop = np.array(Ploop)
        Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
        Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
    
    # Plin, Ploop = birdmodel[model_index].modify_template([alpha_perp, alpha_par, fsigma8], fittingdata=fittingdata, factor_m=m, redindex=redindex, one_nz = one_nz, resum = resum)
        
    
    # Ploop = np.transpose(Ploop, axes=[0, 2, 1, 3])
    
    # Plin, Ploop = birdmodel.modify_template(params[:3])
    P_model_lin, P_model_loop, P_model_interp = birdmodel[model_index].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][model_index])
    Pi = birdmodel[model_index].get_Pi_for_marg(Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][model_index], MinF = MinF)
    
    # print(np.shape(Pi), np.shape(Pi_full))
    
    Plins.append(Plin)
    Ploops.append(Ploop)
    P_models.append(P_model_interp)
    P_model_lins.append(P_model_lin)
    P_model_loops.append(P_model_loop)
    Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + fittingdata.data["ndata"][model_index]] = Pi
    Picount += fittingdata.data["ndata"][model_index]

    P_model = np.concatenate(P_models)
    
    # print(np.shape(Pi))
    # chi_squared = birdmodel.compute_chi2(P_model_interp, Pi, fittingdata.data)
    
    # Sort out the different flavours of marginalisation
    if birdmodel[0].pardict["do_marg"] == 0:
        chi_squared = birdmodel[model_index].compute_chi2(P_model, fittingdata.data)
        chi_squared_print = chi_squared
    elif birdmodel[0].pardict["do_marg"]:
        P_models, P_model_lins, P_model_loops = [], [], []
        if plt is not None or birdmodel[0].pardict["do_marg"] > 1:
            bs_analytic = birdmodel[model_index].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin)
            for i in range(1):
                counter = -3 * (1 - i)
                b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
                b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
                # counter = -2 * (nz - i)
                # b2 = (params[counter + 1]) / np.sqrt(2.0)
                # b4 = (params[counter + 1]) / np.sqrt(2.0)
                bs = np.array(
                    [
                        params[counter],
                        b2,
                        bs_analytic[7 * i],
                        b4,
                        bs_analytic[7 * i + 1],
                        bs_analytic[7 * i + 2],
                        bs_analytic[7 * i + 3],
                        bs_analytic[7 * i + 4],
                        bs_analytic[7 * i + 5] * shot_noise_ratio,
                        bs_analytic[7 * i + 6] * shot_noise_ratio,
                    ]
                )
                P_model_lin, P_model_loop, P_model_interp = birdmodel.compute_model(
                    bs, Plin, Ploop, fittingdata.data["x_data"][i]
                )
                P_models.append(P_model_interp)
                P_model_lins.append(P_model_lin)
                P_model_loops.append(P_model_loop)
        if birdmodel[0].pardict["do_marg"] == 1:
            chi_squared = birdmodel[model_index].compute_chi2_marginalised(P_model, Pi_full, fittingdata.data, onebin = onebin, MinF=MinF)
            # bs_analytic = birdmodel[model_index].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin).T
            # # print(bs_analytic)
            # # print(np.shape(bs_analytic))
            
            # bs = np.array(
            #     [
            #         b1,
            #         b2,
            #         bs_analytic[0],
            #         b4,
            #         bs_analytic[1],
            #         bs_analytic[2],
            #         bs_analytic[3],
            #         # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
            #         # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
            #         # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
            #         bs_analytic[4] * shot_noise_ratio,
            #         bs_analytic[5] * shot_noise_ratio,
            #         bs_analytic[6] * shot_noise_ratio,
            #     ]
            # )
            
            # P_model_lin, P_model_loop, P_model_interp = birdmodel[model_index].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][model_index])
            # P_model = np.concatenate([P_model_interp])
            # chi_squared = birdmodel[model_index].compute_chi2(P_model, fittingdata.data)
            
            if plt is not None:
                chi_squared_print = birdmodel[model_index].compute_chi2(np.concatenate(P_models), fittingdata.data)
                # bs_analytic = birdmodel.compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin)
                # pardict["do_marg"] = 0
                # b2 = (params[-1]) / np.sqrt(2.0)
                # b4 = (params[-1]) / np.sqrt(2.0)
                # bs = [
                #     params[-2],
                #     b2,
                #     bs_analytic[0],
                #     b4,
                #     bs_analytic[1],
                #     bs_analytic[2],
                #     bs_analytic[3],
                #     bs_analytic[4],
                #     bs_analytic[5] * shot_noise_ratio,
                #     bs_analytic[6] * shot_noise_ratio,
                # ]
                # P_model_lin, P_model_loop, P_model_interp = birdmodel.compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][redindex])
                # chi_squared_print = birdmodel.compute_chi2(P_model_interp, fittingdata.data)
                
                # print(bs_analytic)
                
                # pardict["do_marg"] = 1
                
                # return chi_squared_print
        else:
            chi_squared = birdmodel[model_index].compute_chi2(P_model, fittingdata.data)
            chi_squared_print = chi_squared

    if plt is not None:
        chi_squared_print = chi_squared
        if birdmodel[0].pardict["do_marg"]:
            bs_analytic = birdmodel[model_index].compute_bestfit_analytic(P_model, fittingdata.data, P_model_interp, onebin=onebin)
            pardict["do_marg"] = 0
            b2 = (params[-2] + params[-1]) / np.sqrt(2.0)
            b4 = (params[-2] - params[-1]) / np.sqrt(2.0)
            bs = [
                params[-3],
                b2,
                bs_analytic[0],
                b4,
                bs_analytic[1],
                bs_analytic[2],
                bs_analytic[3],
                bs_analytic[4] * fittingdata.data["shot_noise"],
                bs_analytic[5] * fittingdata.data["shot_noise"],
                bs_analytic[6] * fittingdata.data["shot_noise"],
            ]
            P_model_lin, P_model_loop, P_model_interp = birdmodel[model_index].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"])
            chi_squared_print = birdmodel[model_index].compute_chi2(P_model_interp, P_model, fittingdata.data)
            pardict["do_marg"] = 1
        update_plot(pardict, fittingdata.data["x_data"], P_model_interp, plt)
        if np.random.rand() < 0.1:
            print(params, chi_squared_print)
    else:
        if np.random.rand() < 0.001:
            print(params[:, 0], chi_squared[0], len(fittingdata.data["fit_data"]), shot_noise_ratio)
            # print(birdmodel[model_index].compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin))

    if onedflag:
        return -0.5 * chi_squared[0]
    else:
        return -0.5 * chi_squared
    
    # return P_model
    
def read_chain_backend(chainfile):

    reader = emcee.backends.HDFBackend(chainfile)

    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin, flat=True)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples

def best_fit_chi_squared(mock_num):
    
    best = []
    mean = []
    
    for i in range(mock_num+1):
        c = ChainConsumer()
        
        chainfile = '../../data/chainfile/DESI_KP4_LRG_pk_0.20hex0.20_3order_nohex_marg_Shapefit_mock_'+str(i)+'_bin_0.hdf5'
        
        sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
        
        c.add_chain(sample_new, parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"], name = 'Shapefit old')    
        
        data = c.analysis.get_summary(parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"])
        
        mean_data = np.array([data[r"$\alpha_{\perp}$"][1], data[r"$\alpha_{\parallel}$"][2], data["$f\sigma_8$"][1], data["$m$"][1], data["$b_1$"][1], data["$c_2$"][1]])
        
        best_data = sample_new[np.argmax(max_log_likelihood_new)]
        
        # print(mean_data)
        # print(best_data)
        
        if one_nz == False:
            mean_chisquared = lnlike(mean_data, birdmodel_all[redindex], fittingdata, plt, Shapefit)
            best_chisquared = lnlike(best_data, birdmodel_all[redindex], fittingdata, plt, Shapefit)
        else:
            mean_chisquared = lnlike(mean_data, birdmodel_all[0], fittingdata, plt, Shapefit)
            best_chisquared = lnlike(best_data, birdmodel_all[0], fittingdata, plt, Shapefit)
            
        # best.append(best_chisquared/(-0.5)/(len(fittingdata.data["fit_data"]) - len(best_data)))
        # mean.append(mean_chisquared/(-0.5)/(len(fittingdata.data["fit_data"]) - len(mean_data)))
        
        best.append(best_chisquared/(len(fittingdata.data["fit_data"]) - len(best_data)))
        mean.append(mean_chisquared/(len(fittingdata.data["fit_data"]) - len(mean_data)))
        
    np.savetxt('best_fit_LRG_chi2.txt', np.array(best))
    np.savetxt('mean_fit_LRG_chi2.txt', np.array(mean))


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    plot_flag = int(sys.argv[2])
    Shapefit = bool(int(sys.argv[3])) #Enter 0 for normal template fit and 1 for shapefit. 
    # redindex = int(sys.argv[4])
    # onebin = bool(int(sys.argv[5])) #Enter 1 if you are just using one redshift bin, 0 otherwise. 
    # one_nz = bool(int(sys.argv[6])) #Enter 1 if the input ini file only has one redshift bin, 0 otherwise. 
    onebin = True
    one_nz = True
    # vary_sigma8 = bool(int(sys.argv[4]))
    # # vary_sigma8 = False
    # fixedbias = bool(int(sys.argv[5]))
    # # resum = bool(int(sys.argv[6])) #Whether to include IR resummation.
    # resum = True
    # flatprior = bool(int(sys.argv[6]))
    # ncpu = int(sys.argv[7])
    resum = True
        
    try:
        mock_num = int(sys.argv[4])
        mean = False
    except:
        mean = True
    
    pardict = ConfigObj(configfile)
    
    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    
    vary_sigma8 = bool(int(pardict['vary_sigma8']))

    
    redindex = int(pardict['red_index'])
    
    if Shapefit == True:
        print("Using shapefit for redshift bin "+str(redindex) +"!")
        if mean == False:
            keyword = str("Shapefit_mock_") + str(mock_num)
            
            datafiles = np.loadtxt(pardict['datafile'] + '.txt', dtype=str)
            mockfile = str(datafiles) + str(mock_num) + '.dat'
            newfile = '../config/data_mock_' + str(mock_num) + '.txt'
            text_file = open(newfile, "w")
            n = text_file.write(mockfile)
            text_file.close()
            pardict['datafile'] = newfile
            pardict['covfile'] = pardict['covfile'] + '.txt'
        else:
            keyword = str("Shapefit_mock_mean")
            pardict['datafile'] = pardict['datafile'] + '_mean.txt'
            # pardict['covfile'] = pardict['covfile'] + '_mean.txt'
            # pardict['covfile'] = pardict['covfile'] + '.txt'
            # pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
            # pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
            if pardict['constrain'] == 'Single':
                pardict['covfile'] = pardict['covfile'] + '.txt'
            elif pardict['constrain'] == 'Mean':
                pardict['covfile'] = pardict['covfile'] + '_mean.txt'
            else:
                raise ValueError('Enter either "Single" or "Mean" to use the normal or reduced covariance matrix. ')
    else:
        print("Using template fit for redshift bin "+str(redindex) +"!")
        if mean == False:
            keyword = str("Templatefit_mock_") + str(mock_num)
            
            datafiles = np.loadtxt(pardict['datafile'] + '.txt', dtype=str)
            mockfile = str(datafiles) + str(mock_num) + '.dat'
            newfile = '../config/data_mock_' + str(mock_num) + '.txt'
            text_file = open(newfile, "w")
            n = text_file.write(mockfile)
            text_file.close()
            pardict['datafile'] = newfile
            pardict['covfile'] = pardict['covfile'] + '.txt'
        else:
            keyword = str("Templatefit_mock_mean")
            # pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
            # pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
            pardict['datafile'] = pardict['datafile'] + '_mean.txt'
            pardict['covfile'] = pardict['covfile'] + '_mean.txt'
    
    if resum == False:
        try: 
            keyword = keyword + '_noresum'
        except:
            keyword = 'noresum'
    
    # if Shapefit == False:
    #     keyword = keyword + '_template'
        
    print(np.loadtxt(pardict["gridname"], dtype=str))
    

    # Set up the data
    fittingdata = FittingData(pardict)
    
    # Apply the Hartlap correction to the data as if each redshift bin used independent EZmocks
    # Works because the full matrix is block diagonal
    # n_sims = [978, 1000, 1000]
    # n_sims = [1000, 1000, 1000]
    
    # if onebin == False:
    #     hartlap = [(ns - fittingdata.data["ndata"][i] - 2.0) / (ns - 1.0) for i, ns in enumerate(n_sims)]
    #     cov_inv_new = copy.copy(fittingdata.data["cov_inv"])
    #     for i, (nd, ndcum) in enumerate(
    #         zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
    #     ):
    #         cov_inv_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= hartlap[i]
    # else:
    #     length_all = []
    #     for i in range(len(pardict['z_pk'])):
    #         length = len(fittingdata.data["x_data"][i][0]) + len(fittingdata.data["x_data"][i][1])
    #         if pardict['do_hex'] == True:
    #             length += len(fittingdata.data["x_data"][i][2])
    #         length_all.append(length)
        
    #     length_start = 0
    #     length_end = 0
    #     for i in range(1):
    #         if i == 0:
    #             length_start += 0    
    #         else:
    #             length_start += length_all[i-1]
    #         length_end += length_all[i]   
            
    #     print(length_start, length_end)
        
    #     hartlap = (n_sims[redindex] - fittingdata.data["ndata"][0] - 2.0) / (n_sims[redindex] - 1.0)
    #     print(hartlap)
        
    #     nparams = 7.0
    #     percival_A = 2.0/((n_sims[redindex] - fittingdata.data["ndata"][0]-1.0)*(n_sims[redindex] - fittingdata.data['ndata'][0]-4.0))
    #     percival_B = percival_A/2.0*(n_sims[redindex] - fittingdata.data['ndata'][0]-2.0)
    #     percival_m = (1.0+percival_B*(fittingdata.data['ndata'][0] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
    #     print(percival_m)
        
    #     cov_part = fittingdata.data['cov'][length_start:length_end, length_start:length_end]*percival_m
    #     fitdata_part = fittingdata.data['fit_data'][length_start:length_end]
        
    #     cov_lu, pivots, cov_part_inv, info = lapack.dgesv(cov_part, np.eye(len(cov_part)))
        
    #     cov_part_inv = cov_part_inv*hartlap
        
    #     chi2data_part = np.dot(fitdata_part, np.dot(cov_part_inv, fitdata_part))
    #     invcovdata_part = np.dot(fitdata_part, cov_part_inv)
        
    #     fittingdata.data['cov'] = cov_part
    #     fittingdata.data['cov_inv'] = cov_part_inv
    #     fittingdata.data['chi2data'] = chi2data_part
    #     fittingdata.data['invcovdata'] = invcovdata_part
    #     fittingdata.data['fit_data'] = fitdata_part
    
    # Set up the BirdModel
    # birdmodel = BirdModel(pardict, template=True, direct=True, fittingdata=fittingdata)
    
    # print('Start finding prior')
    # bins, freq = np.load('prior_grid_large_random.npy', allow_pickle = True)
    # prior_template = sp.interpolate.RegularGridInterpolator(bins, freq, bounds_error = False, fill_value = -1.0e30, method = 'linear')
    # # prior1, prior2, prior3, prior4 = find_prior(configfile)
    # # prior_template = find_prior(configfile)
    # # print(prior_template([1.0, 1.0, 0.45014, 0.0]))
    # print('Finish finding prior')
    # if fixedbias == True:
    #     keyword = keyword + '_fixedbias'
    #     print('We are fixing b3 and ce1.')
        
    # keyword = keyword + '_best'
    
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
    
    birdmodel_all = []
    for i in range(len(pardict["z_pk"])): 
        # if onebin == True:
        #     i = redindex
        if fittingdata.data["windows"] == None:
            print("No window function!")
            birdmodel_i = BirdModel(pardict, redindex=i, template=True, direct=True, fittingdata=fittingdata, window = None, Shapefit=Shapefit)
        else:
            birdmodel_i = BirdModel(pardict, redindex=i, template=True, direct=True, window = str(fittingdata.data["windows"]), fittingdata=fittingdata, Shapefit=Shapefit)
        
        # if one_nz == False:
        #     shot_noise_fid = (1.0/birdmodel_i.correlator.birds[i].co.nd)
        #     shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
        # else:
        #     shot_noise_fid = (1.0/birdmodel_i.correlator.bird.co.nd)
        #     shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
        
        if one_nz == False:
            shot_noise_fid = (1.0/birdmodel_i.correlator.birds[redindex].co.nd)
            # print(np.float64(fittingdata.data["shot_noise"]))
            # shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[redindex]/shot_noise_fid
           
            shot_noise_ratio_prior = shot_noise_fid/np.float64(fittingdata.data["shot_noise"])[i]
        else:
            shot_noise_fid = (1.0/birdmodel_i.correlator.bird.co.nd)
            # shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
            
            shot_noise_ratio_prior = shot_noise_fid/np.float64(fittingdata.data["shot_noise"])
        
        # birdmodel_i.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio])
        # birdmodel_i.eft_priors = np.array([2.0, 2.0, 10.0, 10.0, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio, 10.0*shot_noise_ratio])
        # birdmodel_i.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24*shot_noise_ratio_prior, 4.0, 20.0])
        # birdmodel_i.eft_priors = np.array([2.0, 20.0, 400.0, 400.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 10.0*shot_noise_ratio_prior])
        # if flatprior == False:
        #     keyword += '_Gaussian'
        #     if fixedbias == False:
        #         # birdmodel_i.eft_priors = np.array([2.0, 20.0, 100.0, 100.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 4.0*shot_noise_ratio_prior])
        #         # birdmodel_i.eft_priors = np.array([2.0, 2.0, 4.0, 1e-10, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])
        #         birdmodel_i.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])
        #         # birdmodel_i.eft_priors = np.array([1e-10, 2.0, 4.0, 4.0, 1e-10*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])




        #     else:
        #         birdmodel_i.eft_priors = np.array([1e-10, 2.0, 4.0, 1e-10, 0.24*shot_noise_ratio_prior, 1e-10*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])
        # else:
        #     keyword += '_flat'
        #     print('Flat prior')
        
        if pardict['prior'] == 'BOSS_MaxF':
            if int(pardict['do_hex']) == 1:
                birdmodel_i.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])
            else:
                birdmodel_i.eft_priors = np.array([2.0, 2.0, 8.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])

            
        elif pardict['prior'] == 'BOSS_MinF':
            if int(pardict['do_hex']) == 1:
                birdmodel_i.eft_priors = np.array([2.0, 4.0, 4.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])
            else:
                birdmodel_i.eft_priors = np.array([2.0, 8.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior])
        else:
            print('Flat prior')

        print(birdmodel_i.eft_priors)
        birdmodel_all.append(birdmodel_i)
        print(i)
        
    # if Shapefit == True:
    #     birdmodel.correlator.projection = Projection(
    #         birdmodel.correlator.config["xdata"],
    #         DA_AP=birdmodel.correlator.config["DA_AP"],
    #         H_AP=birdmodel.correlator.config["H_AP"],
    #         window_fourier_name=birdmodel.correlator.config["windowPk"],
    #         path_to_window="",
    #         window_configspace_file=birdmodel.correlator.config["windowCf"],
    #         binning=birdmodel.correlator.config["with_binning"],
    #         fibcol=birdmodel.correlator.config["with_fibercol"],
    #         Nwedges=birdmodel.correlator.config["wedge"],
    #         wedges_bounds=birdmodel.correlator.config["wedges_bounds"],
    #         zz=birdmodel.correlator.config["zz"],
    #         nz=birdmodel.correlator.config["nz"],
    #         co=birdmodel.correlator.co,
    #     )
    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata)
        
    # if int(pardict['do_hex']) == 1:
    #     pardict['vary_c4'] = 1
    
    if MinF == True or pardict['prior'] == 'BOSS_MaxF':
        pardict['vary_c4'] = 0
        
    # if int(pardict['vary_c4']) == 0 and MinF == False:
    #     keyword += '_anticorr'
    
    start = []
    cosmo_num = 0
    bias_num = 0
    if onebin == True:
        nz = 1
        
        if one_nz == False:
            start.extend([1.0, 1.0, birdmodel_all[0].fN[redindex]*birdmodel_all[0].sigma8[redindex]])
        else:
            start.extend([1.0, 1.0, birdmodel_all[0].fN*birdmodel_all[0].sigma8])
            print(birdmodel_all[0].fN, birdmodel_all[0].sigma8)
            
        cosmo_num += 3
        if Shapefit == True:
            start.append(0.01)
            cosmo_num += 1
            # start.append(1.0)
            # cosmo_num += 1
        if vary_sigma8 == True:
            if one_nz == True:
                start.append(birdmodel_all[0].sigma8)
            else:
                start.append(birdmodel_all[0].sigma8[redindex])
            cosmo_num += 1
        
        if np.int16(pardict['vary_shapefit']) == 1:
            start.append(0.6)
            start.append(0.03)
            cosmo_num += 2
            
        if np.int16(pardict['vary_c4']) == 0:
            start.extend([1.8, 0.5])
            bias_num += 2
            if pardict["do_marg"] == False:
                start.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                bias_num += 8
        else:
            start.extend([1.8, 0.1, 0.1])
            bias_num += 3
            if pardict["do_marg"] == False:
                # start.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                # bias_num += 7
                start.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                bias_num += 6
        
        
    else:
        nz = len(pardict["z_pk"])
        
        for i in range(nz):
            start.extend([1.0, 1.0, birdmodel_all[0].fN[i]*birdmodel_all[0].sigma8[i]])
            if i == 0:
                cosmo_num += 3
            if Shapefit == True:
                start.append(0.01)
                if i == 0:
                    cosmo_num += 1
            if vary_sigma8 == True:
                start.append(birdmodel_all[0].sigma8[i])
                if i == 0:
                    cosmo_num += 1
                    
            if np.int16(pardict['vary_shapefit']) == 1:
                start.append(0.6)
                start.append(0.03)
                if i == 0:
                    cosmo_num += 2
    
            # start.extend([1.8, 0.5])
            # if i == 0:
            #     bias_num += 2
            # if pardict["do_marg"] == False:
            #     start.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            #     if i == 0:
            #         bias_num += 8
            if np.int16(pardict['vary_c4']) == 0:
                start.extend([1.8, 0.5])
                bias_num += 2
                if pardict["do_marg"] == False:
                    start.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                    bias_num += 8
            else:
                start.extend([1.8, 0.5, -0.9])
                bias_num += 3
                if pardict["do_marg"] == False:
                    start.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                    bias_num += 7
    
    # start = [1.8, 0.1, 0.1]
    
    # start = [1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # bias_num = len(start)
    # cosmo_num = 0
    start = np.array(start)
    
    # start = np.array([ 1.0, 1.0, 0.45014412,  0.0,
    #     1.98851054,   0.94830624,   3.16696953,  -5.38132616,
    #      2.11210608,  -0.23025114,   0.6747541 ,  -7.3439635 ,
    #    -10.68175077])
    print(start)
    
    # print(birdmodel_all[0].sigma8)
    
    # params = start 
    
    # for i in range(nz):
        
    #     alpha_perp, alpha_par, fsigma8 = params[i*(cosmo_num + bias_num):3 + i*(cosmo_num + bias_num)]
    #     print(alpha_perp, alpha_par, fsigma8)
    #     if Shapefit == True:
    #         m = params[3 + i*(cosmo_num + bias_num)]
    #         print(m)
    #         if vary_sigma8 == True:
    #             sigma8 = params[4 + i*(cosmo_num + bias_num)]
    #     else:
    #         if vary_sigma8 == True:
    #             sigma8 = params[3 + i*(cosmo_num + bias_num)]
    
    #     bias = params[i*(cosmo_num + bias_num) + cosmo_num :(i+1)*(cosmo_num + bias_num)]
    #     print(bias)
    #     if bias_num == 2:
    #         b1, c2 = bias
    #     else:
    #         b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = bias
            
    #     if (0.90 <= alpha_perp <= 1.10):
    #         alpha_perp_prior = 0.0
    #     else:
    #         print("alpha_perp")
        
    #     if (0.90 <= alpha_par <= 1.10):
    #         alpha_par_prior = 0.0
    #     else:
    #         print("alpha_par")
        
    #     if vary_sigma8 == True:
    #         if (0.0 <= sigma8 <= 1.0):
    #             sigma8_prior = 0.0
    #         else:
    #             print("sigma8")
            
    #     if m < -1.0 or m > 1.0:
    #         print("m")
        
    #     if fsigma8 < 0.0 or fsigma8 > 1.0:
    #         print("fsigma8")
        
    #     bias = params[i*(cosmo_num + bias_num) + cosmo_num:(i+1)*(cosmo_num + bias_num)]
    #     # print(bias)
    #     if bias_num == 2:
    #         b1, c2 = bias
    #     else:
    #         b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = bias
            
    #     # Flat prior for b1
    #     if b1 < 0.0 or b1 > 4.0:
    #         print("b1")
    #     if c2 < -10.0 or c2 > 10.0:
    #         print("c2")
    
    # Does an optimization
    # result = do_optimization(lambda *args: -lnpost(*args), start)
    # print(result["x"])
    
    # params = []
    # models = []
    # ratio = np.linspace(0.9, 1.1, 11)
    # for i in range(4):
    #     for j in range(len(ratio)):
    #         fiducial = np.array([ 1.0, 1.0, 0.45014412,  1.0,
    #             1.98851054,   0.94830624,   3.16696953,  -5.38132616,
    #               2.11210608,  -0.23025114,   0.6747541 ,  -7.3439635 ,
    #             -10.68175077])
    #         fiducial_new = fiducial 
    #         fiducial_new[i] *= ratio[j]
    #         fiducial_new[3] = fiducial_new[3] - 1.0
    #         print(fiducial_new)
    #         model = lnlike(fiducial_new, birdmodel_all, fittingdata, plt, Shapefit)
    #         models.append(model)
    # #         params.append(fiducial_new)
    
    # np.save('SF_model_new.npy', models)
    # model = lnlike(np.array([ 0.9, 1.0, 0.45014412,  1.0,
    #             1.98851054,   0.94830624,   3.16696953,  -5.38132616,
    #               2.11210608,  -0.23025114,   0.6747541 ,  -7.3439635 ,
    #             -10.68175077]), birdmodel_all, fittingdata, plt, Shapefit)
    # birdmodel_all[0].correlator.resum.makeQ(birdmodel_all[0].fN)

    # Does an MCMC
    # output = lnpost(start)
    do_emcee(lnpost, start, keyword)
    # do_zeus(lnpost, start, keyword)
    # print(np.shape(birdmodel_all[0].correlator.bird.P11l), np.shape(birdmodel_all[0].correlator.projection.xbin_mat))
    # np.save('xbinmat.npy', birdmodel_all[0].correlator.projection.xbin_mat)
    
    # c = ChainConsumer()
    
    # sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
    
    # c.add_chain(sample_new, parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"], name = 'Shapefit')
    
    # data = c.analysis.get_summary(parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"])
    
    # mean_data = np.array([data[r"$\alpha_{\perp}$"][1], data[r"$\alpha_{\parallel}$"][2], data["$f\sigma_8$"][1], data["$m$"][1], data["$b_1$"][1], data["$c_2$"][1]])
    
    # best_data = sample_new[np.argmax(max_log_likelihood_new)]
    
    # if one_nz == False:
    #     mean_chisquared = lnlike(mean_data, birdmodel_all[redindex], fittingdata, plt, Shapefit)
    #     best_chisquared = lnlike(best_data, birdmodel_all[redindex], fittingdata, plt, Shapefit)
    # else:
    #     mean_chisquared = lnlike(mean_data, birdmodel_all[0], fittingdata, plt, Shapefit)
    #     best_chisquared = lnlike(best_data, birdmodel_all[0], fittingdata, plt, Shapefit)
        
    
    
    #Find best fit 
    # best_fit_chi_squared(mock_num)
    
    
