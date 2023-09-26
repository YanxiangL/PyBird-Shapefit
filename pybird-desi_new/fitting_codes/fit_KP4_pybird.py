import numpy as np
import sys
import copy
from scipy.stats import norm
from configobj import ConfigObj
from multiprocessing import Pool
from chainconsumer import ChainConsumer
from scipy.linalg import lapack, cholesky, block_diag

sys.path.append("../")
from fitting_codes.fitting_utils import (
    read_chain_backend,
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    update_plot_lin_loop,
    format_pardict,
    do_optimization,
    get_Planck,
)


def do_emcee(func, start):

    import emcee

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 4

    begin = [[(0.1 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if (pardict["do_corr"] or pardict['corr_convert']) else "pk"
    fmt_str = (
        # "%s_%s_%2dhex%2d_%s_%s_%s"+keyword+".hdf5"
        # if (pardict["do_corr"] or pardict['corr_convert'])
        # else 
        "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_kmin0p02_fewerbias"+keyword+".hdf5"
    )
    fitlim = float(birdmodels[0].pardict["sfit_min"][0]) if (pardict["do_corr"] or pardict['corr_convert']) else birdmodels[0].pardict["xfit_max"][0]
    fitlimhex = float(birdmodels[0].pardict["sfit_min"][2]) if (pardict["do_corr"] or pardict['corr_convert']) else birdmodels[0].pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodels[0].pardict["fitfile"],
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

    with Pool() as pool:

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, nparams, func, backend=backend, vectorize=True)
        # sampler = emcee.EnsembleSampler(nwalkers, nparams, func, backend=backend, vectorize=False)


        # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
        if int(pardict['do_marg'] == 1):
            max_iter = 140000
        else:
            max_iter = 2000000
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
            # autocorr[index] = np.max(tau[:cosmo_num])
            autocorr[index] = np.max(tau)
            counter += 1000
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print("Max Auto-Correlation time: {0:.3f}".format(autocorr[index]))

            # Check convergence
            # converged = np.all(tau[:cosmo_num] * 50 < sampler.iteration)
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                print("Reached Auto-Correlation time goal: %d > 50 x %.3f" % (counter, autocorr[index]))
                break
            old_tau = tau
            index += 1


def do_zeus(func, start):

    import zeus

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 4

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s.npy"
    fitlim = birdmodels[0].pardict["xfit_min"][0] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][0]
    fitlimhex = birdmodels[0].pardict["xfit_min"][2] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodels[0].pardict["fitfile"],
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
    with Pool() as pool:

        # Initialize the sampler
        sampler = zeus.EnsembleSampler(nwalkers, nparams, func, pool=pool, vectorize=True)

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
            print("Maximum Auto-Correlation time: {0:.3f}".format(np.max(tau)))

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


def do_dynesty(func, prior_transform, start, jobid):

    from dynesty import NestedSampler

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = (
        "dynesty_%s_%s_%2dhex%2d_%s_%s_%s_%d.hdf5"
        if pardict["do_corr"]
        else "dynesty_%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_%d.hdf5"
    )
    fitlim = birdmodels[0].pardict["xfit_min"][0] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][0]
    fitlimhex = birdmodels[0].pardict["xfit_min"][2] if pardict["do_corr"] else birdmodels[0].pardict["xfit_max"][2]

    taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    chainfile = str(
        fmt_str
        % (
            birdmodels[0].pardict["fitfile"],
            dat_str,
            fitlim,
            fitlimhex,
            taylor_strs[pardict["taylor_order"]],
            hex_str,
            marg_str,
            jobid,
        )
    )
    print(chainfile)

    dsampler = NestedSampler(
        func,
        prior_transform,
        nparams,
        logl_args=[birdmodels, fittingdata, plt],
        ptform_args=[birdmodels],
        nlive=50,
        bound="multi",
        sample="unif",
    )
    dsampler.run_nested()
    dres = dsampler.results
    np.save(chainfile, dres)


def lnpost(params):

    # This returns the posterior distribution which is given by the log prior plus the log likelihood
    if params.ndim == 1:
        prior = lnprior(params, birdmodels)[0]
        like = 0.0
        if prior > -9.9e29:
            like = lnlike(params, birdmodels, fittingdata, plt)
    else:
        prior = lnprior(params, birdmodels)
        index = np.where(prior > -9.9e29)[0]
        like = np.zeros(len(prior))
        if len(index) > 0:
            like[index] = lnlike(params[index], birdmodels, fittingdata, plt)

    return prior + like


def lnprior(params, birdmodels):

    if params.ndim == 1:
        params = params.reshape((-1, len(params)))
    params = params.T

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors

    
    if with_w0 == True:
        ln10As, h, omega_cdm, omega_b, w = params[:5]
    elif with_w0_wa == True:
        ln10As, h, omega_cdm, omega_b, w, wa = params[:6]
    elif with_omegak == True:
        ln10As, h, omega_cdm, omega_b, omega_k = params[:5]
    else:
        ln10As, h, omega_cdm, omega_b = params[:4]
    
    # ln10As, h, omega_cdm, omega_b = 3.0364, 0.6736, 0.1200, 0.02237
    # ln10As, h, omega_cdm, omega_b, omega_k = birdmodels[0].valueref[:, None]
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    lower_bounds = birdmodels[0].valueref - birdmodels[0].pardict["order"] * birdmodels[0].delta
    upper_bounds = birdmodels[0].valueref + birdmodels[0].pardict["order"] * birdmodels[0].delta
    
    # lower_bounds = [2.5, 0.64, 0.10, 0.020]
    # upper_bounds = [3.5, 0.70, 0.13, 0.024]

    priors = np.zeros(np.shape(params[1]))

    # Flat priors for cosmological parameters
    if with_w0 == True:
        for i, param in enumerate([ln10As, h, omega_cdm, omega_b, w]):
            priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)
    elif with_w0_wa == True:
        for i, param in enumerate([ln10As, h, omega_cdm, omega_b, w, wa]):
            priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)
    elif with_omegak == True:
        for i, param in enumerate([ln10As, h, omega_cdm, omega_b, omega_k]):
            priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)
    else:
        for i, param in enumerate([ln10As, h, omega_cdm, omega_b]):
            priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)

    # BBN (D/H) inspired prior on omega_b
    omega_b_prior = -0.5 * (omega_b - birdmodels[0].valueref[3]) ** 2 / 0.00037 ** 2
    priors += omega_b_prior

    # Planck prior
    # diff = params[:4] - birdmodel.valueref
    # Planck_prior = -0.5 * diff @ planck_icov @ diff

    if onebin == True:
        nz = 1
    else:
        nz = len(pardict['z_pk'])
    for i in range(nz):
        if birdmodels[0].pardict["do_marg"]:
            if int(pardict['vary_c4']) == 0:
                b1, c2 = params[-2 * (nz - i) : -2 * (nz - i - 1)] if i != nz - 1 else params[-2 * (nz - i) :]
                c4 = 0
            else:
                b1, c2, c4 = params[-3 * (nz - i) : -3 * (nz - i - 1)] if i != nz - 1 else params[-3 * (nz - i) :]
                # b1, b2, b4 = params[-3 * (nz - i) : -3 * (nz - i - 1)] if i != nz - 1 else params[-3 * (nz - i) :]
                # b1, bp, bd = params[-3 * (nz - i) : -3 * (nz - i - 1)] if i != nz - 1 else params[-3 * (nz - i) :]
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = (
                params[-10 * (nz - i) : -10 * (nz - i - 1)] if i != nz - 1 else params[-10 * (nz - i) :]
            )
            # b1, bp, b3, bd, cct, cr1, cr2, ce1, cemono, cequad = (
            #     params[-10 * (nz - i) : -10 * (nz - i - 1)] if i != nz - 1 else params[-10 * (nz - i) :]
            # )
            # b1, bp, b3, bd, cct, cr1, cr2, ce1, cemono = (
            #     params[-9 * (nz - i) : -9 * (nz - i - 1)] if i != nz - 1 else params[-9 * (nz - i) :]
            # )
            # b1, bp, b3, bd, cct, cr1, ce1, cemono, cequad = (
            #     params[-9 * (nz - i) : -9 * (nz - i - 1)] if i != nz - 1 else params[-9 * (nz - i) :]
            # )

        # Flat prior for b1
        priors += np.where(np.logical_or(b1 < 0.0, b1 > 4.0), -1.0e30, 0.0)

        # Flat prior for c2
        # priors += -0.5*(1.0/4.0)**2*c2**2
        # priors += np.where(np.logical_or(c2 < -100.0, c2 > 100.0), -1.0e30, 0.0)
        # priors += np.where(np.logical_or(c2 < -4.0, c2 > 4.0), -1.0e30, 0.0)
        # priors += np.where(np.logical_or(b2 < -100.0, b2 > 100.0), -1.0e30, 0.0)
        # if int(pardict['vary_c4']) == 1:
        #     priors += np.where(np.logical_or(bp < -5.0, bp > 5.0), -1.0e30, 0.0)
        # else:
        priors += np.where(np.logical_or(c2 < -10.0, c2 > 10.0), -1.0e30, 0.0)
        # priors += np.where(np.logical_or(c2 < -4.0, c2 > 4.0), -1.0e30, 0.0)




        # Gaussian prior for c4
        # priors += -0.5 * c4 ** 2/2.0**2
        if int(pardict['vary_c4']) == 1:
            priors += np.where(np.logical_or(c4 < -10.0, c4 > 10.0), -1.0e30, 0.0)
            # priors += -0.5 * c4 ** 2/2.0**2
            # priors += 0.0
            # priors += -0.5*(1.0/10.0)**2*c4**2
            # priors += np.where(np.logical_or(b4 < -100.0, b4 > 100.0), -1.0e30, 0.0)
            # priors += np.where(np.logical_or(bd < -20.0, bd > 20.0), -1.0e30, 0.0)


        # if not birdmodels[0].pardict["do_marg"]:
        if int(pardict['do_marg']) == 0:
            
            priors += 0
            
            # Gaussian prior for b3 of width 2 centred on 0
            # priors += -0.5 * b3 ** 2/birdmodels[0].eft_priors[0]**2
    
            # # Gaussian prior for cct of width 2 centred on 0
            # priors += -0.5 * cct ** 2/birdmodels[0].eft_priors[1]**2
    
            # # Gaussian prior for cr1 of width 4 centred on 0
            # priors += -0.5 * cr1 ** 2 / birdmodels[0].eft_priors[2]**2
    
            # # Gaussian prior for cr1 of width 4 centred on 0
            # priors += -0.5 * cr2 ** 2 / birdmodels[0].eft_priors[3]**2
    
            # # Gaussian prior for ce1 of width 2 centred on 0
            # priors += -0.5 * ce1 ** 2 / birdmodels[0].eft_priors[4]**2
    
            # # Gaussian prior for cemono of width 2 centred on 0
            # priors += -0.5 * cemono ** 2 / birdmodels[0].eft_priors[5]**2
    
            # # Gaussian prior for cequad of width 2 centred on 0
            # priors += -0.5 * cequad ** 2/birdmodels[0].eft_priors[6]**2
            
            # priors += -0.5*cr2**2/10e-10
            # priors += -0.5*cequad**2/10e-10

    return priors


def lnprior_transform(u, birdmodels):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors

    params = np.zeros(len(u))

    ncosmo = len(birdmodels[0].valueref)
    lower_bounds = birdmodels[0].valueref - birdmodels[0].pardict["order"] * birdmodels[0].delta
    upper_bounds = birdmodels[0].valueref + birdmodels[0].pardict["order"] * birdmodels[0].delta

    # Flat priors for cosmological parameters except omega_b
    params[:3] = u[:3] * (upper_bounds[:3] - lower_bounds[:3]) + lower_bounds[:3]
    params[4] = u[4] * (upper_bounds[4] - lower_bounds[4]) + lower_bounds[4]

    # BBN (D/H) inspired prior on omega_b
    params[3] = norm.ppf(u[3], loc=birdmodels[0].valueref[3], scale=0.00037)

    nz = len(birdmodels[0].pardict["z_pk"])
    for i in range(nz):
        if birdmodels[0].pardict["do_marg"]:
            params[3 * i + ncosmo] = u[3 * i + ncosmo] * (3.5 - 0.5) + 0.5  # b1
            params[3 * i + ncosmo + 1] = norm.ppf(u[3 * i + ncosmo + 1], loc=0.0, scale=2.0)  # c2
            params[3 * i + ncosmo + 2] = norm.ppf(u[3 * i + ncosmo + 2], loc=0.0, scale=2.0)  # c4
        else:
            params[10 * i + ncosmo] = u[10 * i + ncosmo] * (3.5 - 0.5) + 0.5  # b1
            params[10 * i + ncosmo + 1] = norm.ppf(u[10 * i + ncosmo + 1], loc=0.0, scale=2.0)  # c2
            params[10 * i + ncosmo + 2] = norm.ppf(u[10 * i + ncosmo + 2], loc=0.0, scale=2.0)  # b3
            params[10 * i + ncosmo + 3] = norm.ppf(u[10 * i + ncosmo + 3], loc=0.0, scale=2.0)  # c4
            params[10 * i + ncosmo + 4] = norm.ppf(u[10 * i + ncosmo + 4], loc=0.0, scale=2.0)  # cct
            params[10 * i + ncosmo + 5] = norm.ppf(u[10 * i + ncosmo + 5], loc=0.0, scale=2.0)  # cr1
            params[10 * i + ncosmo + 6] = norm.ppf(u[10 * i + ncosmo + 6], loc=0.0, scale=2.0)  # cr2
            params[10 * i + ncosmo + 7] = norm.ppf(u[10 * i + ncosmo + 7], loc=0.0, scale=0.2)  # ce1
            params[10 * i + ncosmo + 8] = norm.ppf(u[10 * i + ncosmo + 8], loc=0.0, scale=1.0)  # cemono
            params[10 * i + ncosmo + 9] = norm.ppf(u[10 * i + ncosmo + 9], loc=0.0, scale=1.0)  # cequad

    return params


# def lnlike(params, birdmodels, fittingdata, plt):

#     onedflag = False
#     if params.ndim == 1:
#         onedflag = True
#         params = params.reshape((-1, len(params)))
#     params = params.T
#     # print(np.shape(params))

#     # Get the bird model
#     ln10As, h, omega_cdm, omega_b = params[:4]
#     # ln10As, h, omega_cdm, omega_b = np.repeat(np.array([3.0364, 0.6736, 0.1200, 0.02237]), np.shape(params)[1]).reshape(4, np.shape(params)[1])

#     Picount = 0
#     P_models, Plins, Ploops = [], [], []
#     P_model_lins, P_model_loops = [], []
#     nmarg = len(birdmodels[0].eft_priors)
#     if onebin == True:
#         nz = 1
#     else:
#         nz = len(pardict['z_pk'])
        
#     Pi_full = np.zeros((nz * len(birdmodels[0].eft_priors), len(fittingdata.data["fit_data"]), len(ln10As)))
#     for i in range(nz):
#         # shot_noise_fid = (1.0/birdmodels[i].correlator.birds[i].co.nd)
        
#         if onebin == True:
#             i = redindex
#             model_index = 0
#         else:
#             model_index = i
            
#         shot_noise_fid = (1.0/3e-4)
#         # shot_noise_fid = (1.0)
#         if onebin == False:
#             shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
#         else:
#             shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
            
#         if birdmodels[0].pardict["do_marg"]:
#             if int(pardict['vary_c4']) == 1:
#                 counter = -3 * (nz - i)
#                 # b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
#                 # b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
                
#                 # b2 = params[counter + 1]
#                 # b4 = params[counter + 2]
                
#                 b2 = params[counter + 2]
#                 b4 = params[counter + 1] - 0.86*params[counter+2]
            
#             else:
#                 counter = -2 * (nz - i)
#                 b2 = (params[counter + 1]) / np.sqrt(2.0)
#                 b4 = (params[counter + 1]) / np.sqrt(2.0)
            
#             margb = np.zeros(np.shape(params)[1])
#             bs = np.array(
#                 [
#                     params[counter],
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
#             counter = -10 * (nz - i)
#             b2 = (params[counter + 1] + params[counter + 3]) / np.sqrt(2.0)
#             b4 = (params[counter + 1] - params[counter + 3]) / np.sqrt(2.0)
#             bs = np.array(
#                 [
#                     params[counter],
#                     b2,
#                     params[counter + 2],
#                     b4,
#                     params[counter + 4],
#                     params[counter + 5],
#                     params[counter + 6],
#                     # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
#                     # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
#                     # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
#                     params[counter + 7] * shot_noise_ratio,
#                     params[counter + 8] * shot_noise_ratio,
#                     params[counter + 9] * shot_noise_ratio,
#                 ]
#             )

#         Plin, Ploop = birdmodels[model_index].compute_pk(np.array([ln10As, h, omega_cdm, omega_b]))
#         P_model_lin, P_model_loop, P_model_interp = birdmodels[model_index].compute_model(
#             bs, Plin, Ploop, fittingdata.data["x_data"][i]
#         )
#         # Pi = birdmodels[i].get_Pi_for_marg(
#         #     Ploop, bs[0], float(fittingdata.data["shot_noise"][i]), fittingdata.data["x_data"][i]
#         # )
#         Pi = birdmodels[model_index].get_Pi_for_marg(
#             Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][i]
#         )

#         Plins.append(Plin)
#         Ploops.append(Ploop)
#         P_models.append(P_model_interp)
#         P_model_lins.append(P_model_lin)
#         P_model_loops.append(P_model_loop)
#         Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + fittingdata.data["ndata"][i]] = Pi
#         Picount += fittingdata.data["ndata"][i]

#     P_model = np.concatenate(P_models)

#     # Sort out the different flavours of marginalisation
#     if birdmodels[0].pardict["do_marg"] == 0:
#         chi_squared = birdmodels[0].compute_chi2(P_model, fittingdata.data)
#         chi_squared_print = chi_squared
#     elif birdmodels[0].pardict["do_marg"]:
#         P_models, P_model_lins, P_model_loops = [], [], []
#         if plt is not None or birdmodels[0].pardict["do_marg"] > 1:
#             if onebin == True: 
#                 bs_analytic = birdmodels[0].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin, eft_priors= eft_priors_all)
#             for i in range(nz):
#                 # counter = -3 * (nz - i)
#                 # b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
#                 # b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
#                 counter = -2 * (nz - i)
#                 b2 = (params[counter + 1]) / np.sqrt(2.0)
#                 b4 = (params[counter + 1]) / np.sqrt(2.0)
#                 bs = np.array(
#                     [
#                         params[counter],
#                         b2,
#                         bs_analytic[7 * i],
#                         b4,
#                         bs_analytic[7 * i + 1],
#                         bs_analytic[7 * i + 2],
#                         bs_analytic[7 * i + 3],
#                         # bs_analytic[7 * i + 4] * float(fittingdata.data["shot_noise"][i]),
#                         # bs_analytic[7 * i + 5] * float(fittingdata.data["shot_noise"][i]),
#                         # bs_analytic[7 * i + 6] * float(fittingdata.data["shot_noise"][i]),
#                         bs_analytic[7 * i + 4] * shot_noise_ratio,
#                         bs_analytic[7 * i + 5] * shot_noise_ratio,
#                         bs_analytic[7 * i + 6] * shot_noise_ratio,
#                     ]
#                 )
#                 P_model_lin, P_model_loop, P_model_interp = birdmodels[model_index].compute_model(
#                     bs, Plins[i], Ploops[i], fittingdata.data["x_data"][i]
#                 )
#                 P_models.append(P_model_interp)
#                 P_model_lins.append(P_model_lin)
#                 P_model_loops.append(P_model_loop)
#         if birdmodels[0].pardict["do_marg"] == 1:
#             chi_squared = birdmodels[0].compute_chi2_marginalised(P_model, Pi_full, fittingdata.data, onebin=onebin, eft_priors=eft_priors_all)
#             if plt is not None:
#                 chi_squared_print = birdmodels[0].compute_chi2(np.concatenate(P_models), fittingdata.data)
#         else:
#             chi_squared = birdmodels[0].compute_chi2(np.concatenate(P_models), fittingdata.data)
#             chi_squared_print = chi_squared

#     if plt is not None:
#         if np.random.rand() < 0.1:
#             update_plot_lin_loop(
#                 pardict,
#                 fittingdata.data["x_data"][plot_flag - 1],
#                 P_models[plot_flag - 1][:, 0],
#                 P_model_lins[plot_flag - 1][:, 0],
#                 P_model_loops[plot_flag - 1][:, 0],
#                 plt,
#             )
#             print(params[:, 0], chi_squared[0], chi_squared_print[0], len(fittingdata.data["fit_data"]))
#     else:
#         if np.random.rand() < 0.001:
#             print(params[:, 0], chi_squared[0], len(fittingdata.data["fit_data"]), shot_noise_ratio)

#     if onedflag:
#         return -0.5 * chi_squared[0]
#     else:
#         return -0.5 * chi_squared

def lnlike(params, birdmodels, fittingdata, plt):

    onedflag = False
    if params.ndim == 1:
        onedflag = True
        params = params.reshape((-1, len(params)))
    params = params.T
    # print(np.shape(params))

    # Get the bird model
    if with_w0 == True:
        ln10As, h, omega_cdm, omega_b, w = params[:5]
    elif with_w0_wa == True:
        ln10As, h, omega_cdm, omega_b, w, wa = params[:6]
    elif with_omegak == True:
        ln10As, h, omega_cdm, omega_b, omega_k = params[:5]
    else:
        ln10As, h, omega_cdm, omega_b = params[:4]
    # ln10As, h, omega_cdm, omega_b = np.repeat(np.array([3.0364, 0.6736, 0.1200, 0.02237]), np.shape(params)[1]).reshape(4, np.shape(params)[1])

    Picount = 0
    P_models, Plins, Ploops = [], [], []
    P_model_lins, P_model_loops = [], []
    # nmarg = len(birdmodels[0].eft_priors)
    # nmarg = 7
    
    if MinF == False:
        if int(pardict['do_hex']) == 1:
            nmarg = 7
        else:
            # # nmarg = 6
            # nmarg = 5
            if pardict['prior'] == 'BOSS_MaxF':
                nmarg = 6
            else:
                nmarg = 5
    else: 
        if int(pardict['do_hex']) == 1:
            nmarg = 5
        else:
            nmarg = 4
    
    
    if onebin == True:
        nz = 1
    else:
        nz = len(pardict['z_pk'])
        
    Pi_full = np.zeros((nz * nmarg, len(fittingdata.data["fit_data"]), len(ln10As)))
    
    # Om_0 = (omega_cdm + omega_b)/h**2
    # redshift = np.float64(pardict['z_pk'])
    # Om = Om_0*(1+redshift)**3/(Om_0*(1+redshift)**3 + (1-Om_0))
    # growth = Om**(6.0/11.0)
    
    # i = 0
    
    # if onebin == True:
    #     i = 0
    #     model_index = 0
    # else:
    #     model_index = i

            
    # shot_noise_fid = (1.0/3e-4)
    # # shot_noise_fid = (1.0)
    # if onebin == False:
    #     shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
    # else:
    #     shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
        
    # if birdmodels[0].pardict["do_marg"]:
    #     if int(pardict['vary_c4']) == 1:
    #         counter = -3 * (nz - i)
    #         # b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
    #         # b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
            
    #         # b2 = params[counter + 1]
    #         # b4 = params[counter + 2]
            
    #         b4 = params[counter + 2]
    #         b2 = (params[counter + 1] - params[counter+2])/0.86
        
    #     else:
    #         counter = -2 * (nz - i)
    #         b2 = (params[counter + 1]) / np.sqrt(2.0)
    #         b4 = (params[counter + 1]) / np.sqrt(2.0)
        
    #     margb = np.zeros(np.shape(params)[1])
    #     bs = np.array(
    #         [
    #             params[counter],
    #             b2,
    #             margb,
    #             b4,
    #             margb,
    #             margb,
    #             margb,
    #             margb,
    #             margb,
    #             margb,
    #         ]
    #     )
    # else:
    #     counter = -10 * (nz - i)
    #     # b2 = (params[counter + 1] + params[counter + 3]) / np.sqrt(2.0)
    #     # b4 = (params[counter + 1] - params[counter + 3]) / np.sqrt(2.0)
    #     b4 = params[counter + 3]
    #     b2 = (params[counter + 1] - params[counter+3])/0.86
    #     bs = np.array(
    #         [
    #             params[counter],
    #             b2,
    #             params[counter + 2],
    #             b4,
    #             params[counter + 4],
    #             params[counter + 5],
    #             params[counter + 6],
    #             # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
    #             # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
    #             # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
    #             params[counter + 7] * shot_noise_ratio,
    #             params[counter + 8] * shot_noise_ratio,
    #             params[counter + 9] * shot_noise_ratio,
    #         ]
    #     )

    # if with_w0 == True:
    #     Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b, w]))
    # elif with_w0_wa == True:
    #     Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b, w, wa]))
    # else:
    #     Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b]))
    # # print(np.shape(Plin), np.shape(Ploop))
    # # Plin, Ploop = birdmodels[model_index].compute_model_direct(np.array([ln10As, h, omega_cdm, omega_b]), fittingdata=fittingdata, redindex=model_index)
    # # print(np.shape(Plin), np.shape(Ploop))
    
    # P_model_lin, P_model_loop, P_model_interp = birdmodels[i].compute_model(
    #     bs, Plin, Ploop, fittingdata.data["x_data"][i]
    # )
    # # Pi = birdmodels[i].get_Pi_for_marg(
    # #     Ploop, bs[0], float(fittingdata.data["shot_noise"][i]), fittingdata.data["x_data"][i]
    # # )
    # Pi = birdmodels[i].get_Pi_for_marg(
    #     Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][i]
    # )

    # Plins.append(Plin)
    # Ploops.append(Ploop)
    # P_models.append(P_model_interp)
    # P_model_lins.append(P_model_lin)
    # P_model_loops.append(P_model_loop)
    # Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + fittingdata.data["ndata"][i]] = Pi
    # Picount += fittingdata.data["ndata"][i]
    
    for i in range(nz):
        
        shot_noise_fid = (1.0/3e-4)
        # shot_noise_fid = (1.0)
        if onebin == False:
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
        else:
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
            
        if birdmodels[0].pardict["do_marg"]:
            # if int(pardict['vary_c4']) == 1:
            #     counter = -3 * (nz - i)
            #     # b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
            #     # b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
                
            #     # b2 = params[counter + 1]
            #     # b4 = params[counter + 2]
                
            #     b4 = params[counter + 2]
            #     b2 = (params[counter + 1] - params[counter+2])/0.86
            
            # else:
            #     counter = -2 * (nz - i)
            #     b2 = (params[counter + 1]) / np.sqrt(2.0)
            #     b4 = (params[counter + 1]) / np.sqrt(2.0)
            
            margb = np.zeros(np.shape(params)[1])
            
            if MinF == True:
                counter = -2*(nz-i)
                b2 = np.ones(np.shape(params)[1])
                # b2 = 2.0 - params[counter+1]
                b3 = params[counter] + 15.0*(-2.0/7.0*(params[counter]-1.0))+6.0*23.0/42.0*(params[counter]-1.0)
                b4 = 0.5*(params[counter+1]) + params[counter] - 1.0
                # b2 = params[counter+1]/np.sqrt(2.0)
                # b4 = params[counter+1]/np.sqrt(2.0)
                # b3 = np.zeros(np.shape(params)[1])
            else:
                if int(pardict['vary_c4']) == 1:
                    counter = -3 * (nz - i)
                    b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
                    b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
                else:
                    counter = -2 * (nz - i)
                    b2 = (params[counter + 1]) / np.sqrt(2.0)
                    b4 = (params[counter + 1]) / np.sqrt(2.0)
                b3 = margb
                
            bs = np.array(
                [
                    params[counter],
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
            # counter = -10 * (nz - i)
            counter = -9 * (nz - i)
            b2 = (params[counter + 1] + params[counter + 3]) / np.sqrt(2.0)
            b4 = (params[counter + 1] - params[counter + 3]) / np.sqrt(2.0)
            # b4 = params[counter + 3]
            # b2 = (params[counter + 1] - params[counter+3])/0.86
            
            
            bs = np.array(
                [
                    params[counter],
                    b2,
                    params[counter + 2],
                    b4,
                    params[counter + 4],
                    params[counter + 5],
                    params[counter + 6],
                    # np.zeros(np.shape(params)[1]),
                    # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
                    # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
                    # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
                    # params[counter + 7] * shot_noise_ratio,
                    # params[counter + 8] * shot_noise_ratio,
                    # params[counter + 9] * shot_noise_ratio,
                    params[counter + 6] * shot_noise_ratio,
                    params[counter + 7] * shot_noise_ratio,
                    params[counter + 8] * shot_noise_ratio,
                    # (2.0/5.0)*params[counter + 8] * shot_noise_ratio
                ]
            )
    
        if with_w0 == True:
            Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b, w]))
        elif with_w0_wa == True:
            Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b, w, wa]))
        elif with_omegak == True:
            Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b, omega_k]))
        else:
            Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b]))
            
        # print(np.array([ln10As, h, omega_cdm, omega_b]))
        # print(np.shape(Plin), np.shape(Ploop))
        # Plin, Ploop = birdmodels[model_index].compute_model_direct(np.array([ln10As, h, omega_cdm, omega_b]), fittingdata=fittingdata, redindex=model_index)
        # print(np.shape(Plin), np.shape(Ploop))
        
        P_model_lin, P_model_loop, P_model_interp = birdmodels[i].compute_model(
            bs, Plin, Ploop, fittingdata.data["x_data"][i]
        )
        # Pi = birdmodels[i].get_Pi_for_marg(
        #     Ploop, bs[0], float(fittingdata.data["shot_noise"][i]), fittingdata.data["x_data"][i]
        # )
        
        Pi = birdmodels[i].get_Pi_for_marg(
            Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][i], MinF = MinF
        )
        
        # Pi = birdmodels[i].get_Pi_for_marg(
        #     Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][i], growth
        # )
    
        Plins.append(Plin)
        Ploops.append(Ploop)
        P_models.append(P_model_interp)
        P_model_lins.append(P_model_lin)
        P_model_loops.append(P_model_loop)
        Pi_full[i * nmarg : (i + 1) * nmarg, Picount : Picount + fittingdata.data["ndata"][i]] = Pi
        Picount += fittingdata.data["ndata"][i]

    P_model = np.concatenate(P_models)

    # Sort out the different flavours of marginalisation
    if birdmodels[0].pardict["do_marg"] == 0:
        chi_squared = birdmodels[0].compute_chi2(P_model, fittingdata.data)
        chi_squared_print = chi_squared
    elif birdmodels[0].pardict["do_marg"]:
        P_models, P_model_lins, P_model_loops = [], [], []
        if plt is not None or birdmodels[0].pardict["do_marg"] > 1:
            if onebin == True: 
                bs_analytic = birdmodels[0].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin, eft_priors= eft_priors_all).T
                # print(np.shape(bs_analytic))
                # print(bs_analytic)
            for i in range(nz):
                # counter = -3 * (nz - i)
                # b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
                # b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
                counter = -2 * (nz - i)
                b2 = (params[counter + 1]) / np.sqrt(2.0)
                b4 = (params[counter + 1]) / np.sqrt(2.0)
                bs = np.array(
                    [
                        params[counter],
                        b2,
                        bs_analytic[7 * i],
                        b4,
                        bs_analytic[7 * i + 1],
                        bs_analytic[7 * i + 2],
                        bs_analytic[7 * i + 3],
                        # bs_analytic[7 * i + 4] * float(fittingdata.data["shot_noise"][i]),
                        # bs_analytic[7 * i + 5] * float(fittingdata.data["shot_noise"][i]),
                        # bs_analytic[7 * i + 6] * float(fittingdata.data["shot_noise"][i]),
                        bs_analytic[7 * i + 4] * shot_noise_ratio,
                        bs_analytic[7 * i + 5] * shot_noise_ratio,
                        bs_analytic[7 * i + 6] * shot_noise_ratio,
                    ]
                )
                P_model_lin, P_model_loop, P_model_interp = birdmodels[i].compute_model(
                    bs, Plins[i], Ploops[i], fittingdata.data["x_data"][i]
                )
                P_models.append(P_model_interp)
                P_model_lins.append(P_model_lin)
                P_model_loops.append(P_model_loop)
        if birdmodels[0].pardict["do_marg"] == 1:
            chi_squared = birdmodels[0].compute_chi2_marginalised(P_model, Pi_full, fittingdata.data, onebin=onebin, eft_priors=eft_priors_all, MinF = MinF)
            # bs_analytic = birdmodels[0].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin, eft_priors= eft_priors_all).T
            
            # bs = np.array(
            #     [
            #         params[counter],
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
            
            # P_model_lin, P_model_loop, P_model_interp = birdmodels[0].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][model_index])
            # P_model = np.concatenate([P_model_interp])
            # chi_squared = birdmodels[0].compute_chi2(P_model, fittingdata.data)
            
            # prior_G = np.sum(bs_analytic**2/np.tile(birdmodels[0].eft_priors**2, np.shape(bs_analytic)[1]).reshape((np.shape(bs_analytic)[1], 7)).T, axis = 0)
            # # print(np.shape(bs_analytic), np.shape(birdmodels[0].eft_priors))
            
            if plt is not None:
                chi_squared_print = birdmodels[0].compute_chi2(np.concatenate(P_models), fittingdata.data)
        else:
            np.save('Bestfit_model.npy', P_models)
            chi_squared = birdmodels[0].compute_chi2(np.concatenate(P_models), fittingdata.data)
            chi_squared_print = chi_squared

    if plt is not None:
        if np.random.rand() < 0.1:
            update_plot_lin_loop(
                pardict,
                fittingdata.data["x_data"][plot_flag - 1],
                P_models[plot_flag - 1][:, 0],
                P_model_lins[plot_flag - 1][:, 0],
                P_model_loops[plot_flag - 1][:, 0],
                plt,
            )
            print(params[:, 0], chi_squared[0], chi_squared_print[0], len(fittingdata.data["fit_data"]))
    else:
        if np.random.rand() < 0.001:
            print(params[:, 0], chi_squared[0], len(fittingdata.data["fit_data"]), shot_noise_ratio)

    # if onedflag:
    #     if birdmodels[0].pardict["do_marg"] == 0:
    #         return -0.5 * chi_squared[0]
    #     else:
    #         return -0.5 * (chi_squared[0] + prior_G[0])
    # else:
    #     if birdmodels[0].pardict["do_marg"] == 0:
    #         return -0.5 * chi_squared
    #     else:
    #         return -0.5 * (chi_squared + prior_G)
    
    if onedflag:
        return -0.5 * chi_squared[0]
    else:
        return -0.5 * chi_squared
    
    # return P_model

def percival_factor(redindex):
    if int(pardict['do_marg']) == 1:
        nparams = len(pardict['freepar']) + 3 
    else:
        nparams = len(pardict['freepar']) + 10
        
    # print(nparams)
        
    if onebin == False:
        percival_A = 2.0/((n_sims[redindex] - fittingdata.data["ndata"][0]-1.0)*(n_sims[redindex] - fittingdata.data['ndata'][0]-4.0))
        percival_B = percival_A/2.0*(n_sims[redindex] - fittingdata.data['ndata'][0]-2.0)
        percival_m = (1.0+percival_B*(fittingdata.data['ndata'][0] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
    else:
        percival_A = 2.0/((n_sims - fittingdata.data["ndata"][0]-1.0)*(n_sims - fittingdata.data['ndata'][0]-4.0))
        percival_B = percival_A/2.0*(n_sims - fittingdata.data['ndata'][0]-2.0)
        percival_m = (1.0+percival_B*(fittingdata.data['ndata'][0] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
    return percival_m

if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    plot_flag = int(sys.argv[2])
    # jobid = int(sys.argv[3])
    # fixedbias = bool(int(sys.argv[3]))
    # try:
    #     redindex = int(sys.argv[4])
    #     print('Using redshift bin '+ str(redindex))
    #     onebin = True
    # except:
    #     print('Using all redshift bins')
    #     onebin = False
    
    # flatprior = bool(int(sys.argv[4]))
    
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    
    # try:
    #     mock_num = int(sys.argv[5])
    #     print("Using one of the mocks.")
    #     datafiles = np.loadtxt(pardict['datafile'], dtype=str)
    #     mockfile = str(datafiles) + str(mock_num) + '.dat'
    #     newfile = '../config/data_mock_' + str(mock_num) + '.txt'
    #     text_file = open(newfile, "w")
    #     n = text_file.write(mockfile)
    #     text_file.close()
    #     pardict['datafile'] = newfile
    #     single_mock = True
    # except:
    #     print("Using the mean of mocks of the same redshift")
    #     single_mock = False
    #     pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
    #     pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
    
    nz = len(pardict['z_pk'])
    if nz > 1.5:
        onebin = False
        print('Using all redshift bins')
    else:
        onebin = True
        redindex = np.int32(pardict['red_index'])
        print('Using redshift bin '+ str(redindex))
    try:
        mock_num = int(sys.argv[3])
        if nz == 1:
            datafiles = np.loadtxt(pardict['datafile'] + '.txt', dtype=str)
            mockfile = str(datafiles) + str(mock_num) + '.dat'
            newfile = '../config/data_mock_' + str(mock_num) + '.txt'
            text_file = open(newfile, "w")
            n = text_file.write(mockfile)
            text_file.close()
            pardict['datafile'] = newfile
            pardict['covfile'] = pardict['covfile'] + '.txt'
            single_mock = True
        else: 
            raise ValueError('Cannot fit multiple redshift for one single mock at the moment.')
    except:
        if nz == 1:
            pardict['datafile'] = pardict['datafile'] + '_mean.txt'
            # pardict['covfile'] = pardict['covfile'] + '_mean.txt'
            if pardict['constrain'] == 'Single':
                pardict['covfile'] = pardict['covfile'] + '.txt'
            elif pardict['constrain'] == 'Mean':
                pardict['covfile'] = pardict['covfile'] + '_mean.txt'
            else:
                raise ValueError('Enter either "Single" or "Mean" to use the normal or reduced covariance matrix. ')
            # pardict['covfile'] = pardict['covfile'] + '_100_mean.txt'
            print(pardict['covfile'])


        else:
            cov_all = []
            string = ''
            for i in range(nz):
                if pardict['constrain'] == 'Single':
                    pardict['covfile'][i] = pardict['covfile'][i] + '.txt'
                elif pardict['constrain'] == 'Mean':
                    pardict['covfile'][i] = pardict['covfile'][i] + '_mean.txt'
                else:
                    raise ValueError('Enter either "Single" or "Mean" to use the normal or reduced covariance matrix. ')
                
                cov = np.loadtxt(pardict['covfile'][i])
                # cov = np.loadtxt(pardict['covfile'][i] + '_mean.txt')
                cov_all.append(cov)
                print(pardict['covfile'][i])
                string += '_' + str(pardict['z_pk'][i])
            cov_new = block_diag(*cov_all)
            newfile = '../../data/cov' + string + '_mean.txt'
            np.savetxt(newfile, cov_new)
            pardict['covfile'] = newfile
            pardict['datafile'] = pardict['datafile'] + '_mean.txt'
        single_mock = False

    # Set up the data
    fittingdata = FittingData(pardict)
    
    # n_sims = [978, 1000, 1000]
    # n_sims = [1000, 1000, 1000]

    # if onebin == False:
        
    #     # Apply the Hartlap correction to the data as if each redshift bin used independent EZmocks
    #     # Works because the full matrix is block diagonal
        
    #     hartlap = [(ns - fittingdata.data["ndata"][i] - 2.0) / (ns - 1.0) for i, ns in enumerate(n_sims)]
    #     cov_inv_new = copy.copy(fittingdata.data["cov_inv"])
    #     for i, (nd, ndcum) in enumerate(
    #         zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
    #     ):
    #         cov_inv_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= hartlap[i]
            
    #     # Overwrite with the corrected ones
    #     fittingdata.data["cov_inv"] = cov_inv_new
    #     fittingdata.data["chi2data"] = np.dot(
    #         fittingdata.data["fit_data"], np.dot(fittingdata.data["cov_inv"], fittingdata.data["fit_data"])
    #     )
    #     fittingdata.data["invcovdata"] = np.dot(fittingdata.data["fit_data"], fittingdata.data["cov_inv"])
        
    #     nz = len(pardict['z_pk'])
    #     keyword = '_all_old'
    # else:
    #     length_all = []
    #     for i in range(len(pardict['z_pk'])):
    #         length = len(fittingdata.data["x_data"][i][0]) + len(fittingdata.data["x_data"][i][1])
    #         if pardict['do_hex'] == True:
    #             length += len(fittingdata.data["x_data"][i][2])
    #         length_all.append(length)
        
    #     length_start = 0
    #     length_end = 0
    #     for i in range(np.int32(redindex+1)):
    #         if i == 0:
    #             length_start += 0    
    #         else:
    #             length_start += length_all[i-1]
    #         length_end += length_all[i]   
            
    #     print(length_start, length_end)
        
    #     cov_part = fittingdata.data['cov'][length_start:length_end, length_start:length_end]
        
    #     hartlap = (n_sims[redindex] - fittingdata.data["ndata"][redindex] - 2.0) / (n_sims[redindex] - 1.0)
        
    #     cov_part = cov_part*hartlap
        
    #     fitdata_part = fittingdata.data['fit_data'][length_start:length_end]
        
    #     cov_lu, pivots, cov_part_inv, info = lapack.dgesv(cov_part, np.eye(len(cov_part)))
        
    #     chi2data_part = np.dot(fitdata_part, np.dot(cov_part_inv, fitdata_part))
    #     invcovdata_part = np.dot(fitdata_part, cov_part_inv)
        
    #     fittingdata.data['cov'] = cov_part
    #     fittingdata.data['cov_inv'] = cov_part_inv
    #     fittingdata.data['chi2data'] = chi2data_part
    #     fittingdata.data['invcovdata'] = invcovdata_part
    #     fittingdata.data['fit_data'] = fitdata_part
        
    #     nz = 1 
        
    #     if single_mock == False:
    #         keyword = '_bin_'+str(redindex) + '_mean'
    #     else:
    #         keyword = '_bin_'+str(redindex) + '_mock_' + str(mock_num)
    
    # Apply the Hartlap correction to the data as if each redshift bin used independent EZmocks
    # Works because the full matrix is block diagonal
    # n_sims = [978, 1000, 1000]
    n_sims = np.float64(pardict['n_sims'])
    
    if onebin == False:
        # hartlap = [(ns - fittingdata.data["ndata"][i] - 2.0) / (ns - 1.0) for i, ns in enumerate(n_sims)]
        # percival = [percival_factor(i) for i in range(nz)]
        
        # length_all = []
        # for i in range(len(pardict['z_pk'])):
        #     length = len(fittingdata.data["x_data"][i][0]) + len(fittingdata.data["x_data"][i][1])
        #     if pardict['do_hex'] == True:
        #         length += len(fittingdata.data["x_data"][i][2])
        #     length_all.append(length)
            
        # length_start = 0
        # length_end = 0
        # cov_full = []
        # for i in range(nz):
        #     if i == 0:
        #         length_start += 0    
        #     else:
        #         length_start += length_all[i-1]
        #     length_end += length_all[i]
            
        #     hartlap = (n_sims[i] - fittingdata.data["ndata"][0] - 2.0) / (n_sims[i] - 1.0)
            
        #     percival_m = percival_factor(i)
            
        #     cov_part = fittingdata.data['cov'][length_start:length_end, length_start:length_end]*percival_m/hartlap
            
        #     cov_full.append(cov_part)
            
        # cov_new = block_diag(*cov_full)
        
        # cov_lu, pivots, cov_new_inv, info = lapack.dgesv(cov_new, np.eye(len(cov_new)))
        
        # fitdata = fittingdata.data['fit_data']
        
        # chi2data = np.dot(fitdata, np.dot(cov_new_inv, fitdata))
        
        # invcovdata = np.dot(fitdata, cov_new_inv)
        
        # fittingdata.data['cov'] = cov_new
        # fittingdata.data['cov_inv'] = cov_new_inv
        # fittingdata.data['chi2data'] = chi2data
        # fittingdata.data['invcovdata'] = invcovdata
        
        # # cov_new = copy.copy(fittingdata.data['cov'])
        # # for i, (nd, ndcum) in enumerate(
        # #     zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
        # # ):
        # #     cov_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= percival[i]
        
        # # cov_inv_new = copy.copy(fittingdata.data["cov_inv"])
        # # for i, (nd, ndcum) in enumerate(
        # #     zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
        # # ):
        # #     cov_inv_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= hartlap[i]
        
        # # nz = len(pardict['z_pk'])
        keyword = '_all_mean'
    else:
        # length_all = []
        # for i in range(len(pardict['z_pk'])):
        #     length = len(fittingdata.data["x_data"][i][0]) + len(fittingdata.data["x_data"][i][1])
        #     if pardict['do_hex'] == True:
        #         length += len(fittingdata.data["x_data"][i][2])
        #     length_all.append(length)
        
        
        # length_start = 0
        # length_end = 0
        # for i in range(1):
        #     if i == 0:
        #         length_start += 0    
        #     else:
        #         length_start += length_all[i-1]
        #     length_end += length_all[i]   
            
        # print(length_start, length_end)
        
        # hartlap = (n_sims - fittingdata.data["ndata"][0] - 2.0) / (n_sims - 1.0)
        # print(hartlap)
        
        # percival_m = percival_factor(redindex)
        # print(percival_m)
        
        # cov_part = fittingdata.data['cov'][length_start:length_end, length_start:length_end]*percival_m
        # fitdata_part = fittingdata.data['fit_data'][length_start:length_end]
        
        # cov_lu, pivots, cov_part_inv, info = lapack.dgesv(cov_part, np.eye(len(cov_part)))
        
        # cov_part_inv = cov_part_inv*hartlap
        
        # chi2data_part = np.dot(fitdata_part, np.dot(cov_part_inv, fitdata_part))
        # invcovdata_part = np.dot(fitdata_part, cov_part_inv)
        
        # fittingdata.data['cov'] = cov_part
        # fittingdata.data['cov_inv'] = cov_part_inv
        # fittingdata.data['chi2data'] = chi2data_part
        # fittingdata.data['invcovdata'] = invcovdata_part
        # fittingdata.data['fit_data'] = fitdata_part
        
        # nz = 1 
        
        if single_mock == False:
            keyword = '_bin_'+str(redindex) + '_mean'
        else:
            keyword = '_bin_'+str(redindex) + '_mock_' + str(mock_num)
    
    # keyword = keyword + '_noresum_'
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
        
        
        
    # keyword = keyword + '_old'
        
    # Set up the BirdModels
    birdmodels = []
    for i in range(nz):
        #This is the default shot-noise in the pybird.py 
        shot_noise_fid = (1.0/3e-4)
        if onebin ==True:
            i = redindex
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
        else:
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
        
        print('redshift bin ' + str(i))
        
        model = BirdModel(pardict, redindex=i)
        # model = BirdModel(pardict, redindex=i, direct=True, fittingdata=fittingdata)
        
        # model.eft_priors = np.array([0.002, 0.002, 0.002, 2.0, 0.2, 0.0002, 0.0002])
        # model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio])
        # model.eft_priors = np.array([2.0, 2.0, 10.0, 10.0, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio, 10.0/shot_noise_ratio])
        # model.eft_priors = np.array([2.0, 8.0, 10.0, 10.0, 0.012/shot_noise_ratio, 10.0, 10.0])
        # model.eft_priors = np.array([10.0, 500.0, 5000.0, 5000.0, 40.0/shot_noise_ratio, 40.0, 100.0])
        # model.eft_priors = np.array([4.0, 40.0, 400.0, 400.0, 0.24/shot_noise_ratio, 4.0, 20.0])
        # model.eft_priors = np.array([4.0, 10.0, 10.0, 10.0, 0.24/shot_noise_ratio, 4.0, 10.0])
        # model.eft_priors = np.array([10.0, 200.0, 2000.0, 2000.0, 2.4/shot_noise_ratio, 50.0, 20.0])
        
        # if flatprior == False:
        #     if fixedbias == False:
        #         # model.eft_priors = np.array([1e-10, 100.0, 100.0, 1e-10, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
        #         # model.eft_priors = np.array([5.0, 30.0, 100.0, 1e-10, 2.0/shot_noise_ratio, 50.0/shot_noise_ratio, 50.0/shot_noise_ratio])
        #         # model.eft_priors = np.array([2.0, 2.0, 5.0, 1e-10, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])
        #         model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])
        #         # model.eft_priors = np.array([1e-10, 2.0, 4.0, 4.0, 1e-10/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])
        #         # model.eft_priors = np.array([1e-10, 2.0, 4.0, 1e-10, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
        #         # model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 1e-10, 1e-10, 1e-10])


        #     else:
        #         model.eft_priors = np.array([1e-10, 2.0, 4.0, 1e-10, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
        #     print(model.eft_priors)
        # else:
        #     print('Flat prior')
        
        if pardict['prior'] == 'BOSS_MaxF':
            if int(pardict['do_hex']) == 1:
                model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])
            else:
                model.eft_priors = np.array([2.0, 2.0, 8.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])

            # model.eft_priors = np.array([1e-10, 2.0, 4.0, 1e-10, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
            
        elif pardict['prior'] == 'BOSS_MinF':
            if int(pardict['do_hex']) == 1:
                # model.eft_priors = np.array([1e-10, 2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
                model.eft_priors = np.array([2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio])
            else:
            # model.eft_priors = np.array([2.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio])
                # model.eft_priors = np.array([1e-10, 2.0, 4.0, 1e-10, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
                model.eft_priors = np.array([2.0, 8.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio])
        else:
            print('Flat prior')
            
        birdmodels.append(model)
    
    # if flatprior == False:
    #     keyword += '_Gaussian'
    #     if onebin == False:
    #         eft_priors_all = np.hstack([birdmodels[i].eft_priors for i in range(nz)])
    #         # print(eft_priors_all)
    #     else:
    #         eft_priors_all = birdmodels[0].eft_priors
    # else:
    #     eft_priors_all = None
    #     keyword += '_flat'
    
    if pardict['prior'] == 'BOSS_MaxF' or pardict['prior'] == 'BOSS_MinF':
        if onebin == False:
            eft_priors_all = np.hstack([birdmodels[i].eft_priors for i in range(nz)])
            # print(eft_priors_all)
        else:
            eft_priors_all = birdmodels[0].eft_priors
    else:
        eft_priors_all = None
        
    if MinF == True or pardict['prior'] == 'BOSS_MaxF':
        pardict['vary_c4'] = 0
    
    # if int(pardict['vary_c4']) == 0 and MinF == False:
    #     keyword += '_anticorr'

    # # Read in and create a Planck prior covariance matrix
    # Planck_file = "/Volumes/Work/UQ/CAMB/COM_CosmoParams_fullGrid_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
    # planck_mean, planck_cov, planck_icov = get_Planck(Planck_file, 4)

    # Plotting (for checking/debugging, should turn off for production runs)
    
    # if int(pardict['do_hex']) == 1:
    #     pardict['vary_c4'] = 1
    
    with_w0 = False
    with_w0_wa = False
    with_omegak = False
    plt = None
    cosmo_num = 4
    if plot_flag:
        plt = create_plot(pardict, fittingdata, plotindex=plot_flag - 1)

    if "w" in pardict.keys():
        start = [
            [birdmodels[0].valueref[0], birdmodels[0].valueref[1], birdmodels[0].valueref[2], birdmodels[0].valueref[3], birdmodels[0].valueref[4]],
        ]
        with_w0 = True
        cosmo_num = 5
        if "wa" in pardict.keys():
            start = [
                [birdmodels[0].valueref[0], birdmodels[0].valueref[1], birdmodels[0].valueref[2], birdmodels[0].valueref[3], birdmodels[0].valueref[4], 
                 birdmodels[0].valuered[5]],
            ]
            with_w0 = False
            with_w0_wa = True
            cosmo_num = 6
    elif pardict['freepar'][-1] == 'Omega_k':
        start = [
            [birdmodels[0].valueref[0], birdmodels[0].valueref[1], birdmodels[0].valueref[2], birdmodels[0].valueref[3], -0.01],
        ]
        with_omegak = True
        cosmo_num = 5
        
    else:
        start = [
            [birdmodels[0].valueref[0], birdmodels[0].valueref[1], birdmodels[0].valueref[2], birdmodels[0].valueref[3]],
        ]
        
    if birdmodels[0].pardict["do_marg"]:
        for i in range(nz):
            if int(pardict['vary_c4']) == 0:
                start.append([1.8, 0.5])
            else:
                start.append([1.8, 0.1, 0.1])
    else:
        for i in range(nz):
            start.append([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            # start = ([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            # start.append([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            
    # start = [1.8, 0.1, 0.1]
    # start = [2.04975774,   2.25471899,  -3.14055891,   6.40960192,
    #     -1.34327667, -13.12456156,  -1.89855842,   4.05446468,
    #     -1.63358167]
    start = np.concatenate(start)
    
    # start = np.array(np.array([3.00898406, 0.67587391, 0.12082079, 0.02228337, 2.03935988,
    #         1.79887585, 7.76322793]))
    # start = ([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    print(start)
    
    # start = np.array([ 3.01245146e+00,  6.77394731e-01,  1.19686894e-01,  2.21357835e-02,
    #         2.09043032e+00,  2.52056425e+00, -8.30423069e+00,  1.34075740e+01,
    #         -6.01932976e+00, -1.68115562e+01,  2.35823442e-06, -2.39499744e+00,
    #         1.61882552e+01])
    
    # start = np.array([3.05027989, 0.67762633, 0.1214984 , 0.02235069, 1.98879658,
    #     1.72133191, 8.72687848])

    # Does an optimization
    # print(start)
    # result = do_optimization(lambda *args: -lnpost(*args), start)
    # print(result["x"])
    
    # birdmodels[0].pardict["do_marg"] = 2
    # print(-2.0*lnlike(result["x"], birdmodels, fittingdata, plt))

    # Does an MCMC
    do_emcee(lnpost, start)
    # do_zeus(lnpost, start)
    # do_dynesty(lnlike, lnprior_transform, start, jobid)
    # print(-lnpost(np.array([3.00898406, 0.67587391, 0.12082079, 0.02228337, 2.03935988,
    #         1.79887585, 7.76322793])))
    # params = []
    # ratio = np.linspace(0.9, 1.1, 11)
    # for i in range(4):
    #     for j in range(len(ratio)):
    #         fiducial = np.array([3.0364, 0.6736, 0.1200, 0.02237, 1.98866159,   0.94786764,   3.16135093,  -5.38572144,
    #                  2.11842905,  -0.23023721, 0.0,   0.67113864,  -7.33575874,
    #                -10.6802009])
    #         fiducial_new = fiducial 
    #         fiducial_new[i] *= ratio[j]
    #         # print(fiducial_new)
    #         params.append(fiducial_new)
            
    # params = np.array(params)
    # # print(np.shape(params))
    # models = lnlike(params, birdmodels, fittingdata, plt)
    # np.save('FS_model.npy', models)
    
    # fiducial = np.array([3.0364, 0.6736, 0.12, 0.02237, 2.04973726,   2.25430457,  -3.13967641,   6.41162702,
    #     -1.34190849, -13.12405747,  -1.89736831,   4.05387642,
    #     -1.63381558])
    
    # model = lnlike(fiducial, birdmodels, fittingdata, plt)
    # np.save('model_pybird.npy', model)

    # # Does some plotting of a previously run chain
    # c = ChainConsumer()

    # # chainfiles = [
    # #     "/Volumes/Work/UQ/DESI/KP4/fits/DESI_KP4_LRG_ELG_QSO_pk_0.15hex0.15_3order_hex_marg_kmin0p02_fewerbias.hdf5",
    # # ]
    
    # chainfiles = [
    #     "../../data/DESI_KP4_LRG_ELG_QSO_pk_0.15hex0.15_3order_hex_marg_kmin0p02_fewerbias"+keyword+".hdf5",
    # ]

    # bestfits = []
    # for chaini, chainfile in enumerate(chainfiles):
    #     burntin, bestfit, like = read_chain_backend(chainfile)
    #     paramnames = [r"$\mathrm{ln}(10^{10}A_{s})$", r"$h$", r"$\omega_{cdm}$", r"$\omega_{b}$"]
    #     c.add_chain(burntin[:, :4], parameters=paramnames, posterior=like)
    #     bestfits.append(bestfit)

    # print(bestfits)
    # c.add_marker(bestfits[0][:4], parameters=paramnames, marker_size=100, marker_style="*", color="r")
    # c.configure(bar_shade=True)
    # c.plotter.plot(truth=birdmodels[0].valueref, display=True)
    # print(c.analysis.get_summary())

    # # Reformat and output the chain as ASCII
    # bestfits = []
    # for chaini, chainfile in enumerate(chainfiles):
    #     burntin, bestfit, like = read_chain_backend(chainfile)
    #     As = np.exp(burntin[:, 0]) / 1.0e10
    #     Omega_m = (burntin[:, 2] + burntin[:, 3] + float(pardict["m_ncdm"]) / 93.14) / burntin[:, 1] ** 2
    #     # b2_LRG = (burntin[:, 5] + burntin[:, 6]) / np.sqrt(2.0)
    #     # b2_ELG = (burntin[:, 8] + burntin[:, 9]) / np.sqrt(2.0)
    #     # b2_QSO = (burntin[:, 11] + burntin[:, 12]) / np.sqrt(2.0)
    #     # new_chain = np.hstack(
    #     #    [
    #     #        burntin[:, :4],
    #     #        As[:, None],
    #     #        Omega_m[:, None],
    #     #        burntin[:, 4, None],
    #     #        b2_LRG[:, None],
    #     #        burntin[:, 7, None],
    #     #        b2_ELG[:, None],
    #     #        burntin[:, 10, None],
    #     #        b2_QSO[:, None],
    #     #        like[:, None],
    #     #    ]
    #     # )
    #     b2_LRG = (burntin[:, 5]) / np.sqrt(2.0)
    #     b2_ELG = (burntin[:, 7]) / np.sqrt(2.0)
    #     b2_QSO = (burntin[:, 9]) / np.sqrt(2.0)
    #     new_chain = np.hstack(
    #         [
    #             burntin[:, :4],
    #             As[:, None],
    #             Omega_m[:, None],
    #             burntin[:, 4, None],
    #             b2_LRG[:, None],
    #             burntin[:, 6, None],
    #             b2_ELG[:, None],
    #             burntin[:, 8, None],
    #             b2_QSO[:, None],
    #             like[:, None],
    #         ]
    #     )
        
    #     # np.savetxt(
    #     #     "/Volumes/Work/UQ/DESI/KP4/fits/DESI_KP4_LRG_ELG_QSO_pk_0.15hex0.15_3order_hex_marg_kmin0p02_fewerbias.txt",
    #     #     new_chain,
    #     #     header="1n10^10As      h       omega_cdm        omega_b       As      Omega_m      b1_LRG       b2_LRG      b1_ELG      b2_ELG      b1_QSO      b2_QSO      posterior",
    #     # )
        
    #     np.savetxt(
    #         "../../data/DESI_KP4_LRG_ELG_QSO_pk_0.15hex0.15_3order_hex_marg_kmin0p02_fewerbias"+keyword+".txt",
    #         new_chain,
    #         header="1n10^10As      h       omega_cdm        omega_b       As      Omega_m      b1_LRG       b2_LRG      b1_ELG      b2_ELG      b1_QSO      b2_QSO      posterior",
    #     )
    
# kmax = 0.08 
# c = ChainConsumer()
# for i in range(9):
#     kmax_new = round(kmax + 0.02*i, 2)
#     try:
#         chainfile = '../../data/DESI_KP4_LRG_fix_single_Flops_pk_' + str(kmax_new) +'hex'+str(kmax_new) + '_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mean_flat.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     except:
#         chainfile = '../../data/DESI_KP4_LRG_fix_single_Flops_pk_' + str(kmax_new) + '0' +'hex'+str(kmax_new) + '0' + '_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mean_flat.hdf5'
#         sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     print(chainfile)
#     sample_new[:, 2] = (sample_new[:, 2] + sample_new[:, 3])/sample_new[:, 1]**2
#     c.add_chain(sample_new[:, :3], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{m}$"], name = str(kmax_new))
# data_fix = c.analysis.get_summary()

# parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{m}$"]

# fig = plt.figure()
# param = parameters[0]
# plt.subplot(3, 1, 1)
# kmax = []
# central = []
# low = []
# high = []
# for j in range(4, 9):
#     kmax.append(0.08 + j*0.02)
#     central.append(data_fix[j][param][1])
#     low.append(data_fix[j][param][1] - data_fix[j][param][0])
#     high.append(data_fix[j][param][2] - data_fix[j][param][1])


# plt.errorbar(kmax, central, yerr = np.array([low, high]), fmt='.', label = 'Fix')
# plt.hlines(3.0364, xmin = 0.15, xmax = 0.25, linestyle = 'dashed')
# kmax = []
# central = []
# low = []
# high = []
# for j in range(4, 9):
#     kmax.append(0.08 + j*0.02)
#     central.append(data_free[j][param][1])
#     low.append(data_free[j][param][1] - data_free[j][param][0])
#     high.append(data_free[j][param][2] - data_free[j][param][1])
# plt.errorbar(np.array(kmax)+0.005, central, yerr = np.array([low, high]), fmt='.', label = 'Free')
# plt.xticks(fontsize=7)
# plt.ylabel(param)
# plt.legend(fontsize = 7)

# param = parameters[1]
# plt.subplot(3, 1, 2)
# kmax = []
# central = []
# low = []
# high = []
# for j in range(4, 9):
#     kmax.append(0.08 + j*0.02)
#     central.append(data_fix[j][param][1])
#     low.append(data_fix[j][param][1] - data_fix[j][param][0])
#     high.append(data_fix[j][param][2] - data_fix[j][param][1])

# plt.errorbar(kmax, central, yerr = np.array([low, high]), fmt='.', label = 'Fix')
# plt.hlines(0.6736, xmin = 0.15, xmax = 0.25, linestyle = 'dashed')
# kmax = []
# central = []
# low = []
# high = []
# for j in range(4, 9):
#     kmax.append(0.08 + j*0.02)
#     central.append(data_free[j][param][1])
#     low.append(data_free[j][param][1] - data_free[j][param][0])
#     high.append(data_free[j][param][2] - data_free[j][param][1])
# plt.errorbar(np.array(kmax)+0.005, central, yerr = np.array([low, high]), fmt='.', label = 'Free')
# plt.xticks(fontsize=7)
# plt.ylabel(param)
# plt.legend(fontsize = 7)

# param = parameters[2]
# plt.subplot(3, 1, 3)
# kmax = []
# central = []
# low = []
# high = []
# for j in range(4, 9):
#     kmax.append(0.08 + j*0.02)
#     central.append(data_fix[j][param][1])
#     low.append(data_fix[j][param][1] - data_fix[j][param][0])
#     high.append(data_fix[j][param][2] - data_fix[j][param][1])

# plt.errorbar(kmax, central, yerr = np.array([low, high]), fmt='.', label = 'Fix')
# plt.hlines(0.31377, xmin = 0.15, xmax = 0.25, linestyle = 'dashed')
# kmax = []
# central = []
# low = []
# high = []
# for j in range(4, 9):
#     kmax.append(0.08 + j*0.02)
#     central.append(data_free[j][param][1])
#     low.append(data_free[j][param][1] - data_free[j][param][0])
#     high.append(data_free[j][param][2] - data_free[j][param][1])
# plt.errorbar(np.array(kmax)+0.005, central, yerr = np.array([low, high]), fmt='.', label = 'Free')
# plt.xticks(fontsize=7)
# plt.xlabel('k (h/Mpc)')
# plt.ylabel(param)
# plt.legend(fontsize = 7)
# plt.savefig('LRG_kmax_free_vs_fix_high.png', dpi = 300)
