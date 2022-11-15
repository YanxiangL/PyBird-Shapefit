import numpy as np
import sys
import copy
from scipy.stats import norm
from configobj import ConfigObj
from multiprocessing import Pool
from chainconsumer import ChainConsumer
from scipy.linalg import lapack, cholesky

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
    dat_str = "xi" if pardict["do_corr"] else "pk"
    fmt_str = (
        "%s_%s_%2dhex%2d_%s_%s_%s"+keyword+".hdf5"
        if pardict["do_corr"]
        else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_kmin0p02_fewerbias"+keyword+".hdf5"
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
        )
    )
    print(chainfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, nparams)

    with Pool() as pool:

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, nparams, func, backend=backend, vectorize=True)

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
            autocorr[index] = np.mean(tau)
            counter += 100
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))

            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
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
    fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s.hdf5"
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
        while ~converged:
            sampler.run_mcmc(begin, nsteps=20)
            tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
            converged = np.all(10 * tau < niter)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            old_tau = tau
            begin = None
            niter += 1000
            print("Niterations/Max Iterations: ", niter, "/", 5000)
            print("Integrated ACT/Min Convergence Iterations: ", tau, "/", np.amax(10 * tau))
            if niter >= 5000:
                break

        # Remove burn-in and and save the samples
        tau = zeus.AutoCorrTime(sampler.get_chain(discard=0.5))
        burnin = int(2 * np.max(tau))
        samples = sampler.get_chain(discard=burnin, flat=True).T


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

    ln10As, h, omega_cdm, omega_b = params[:4]
    # ln10As, h, omega_cdm, omega_b = 3.0364, 0.6736, 0.1200, 0.02237
    # ln10As, h, omega_cdm, omega_b, omega_k = birdmodels[0].valueref[:, None]
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    lower_bounds = birdmodels[0].valueref - birdmodels[0].pardict["order"] * birdmodels[0].delta
    upper_bounds = birdmodels[0].valueref + birdmodels[0].pardict["order"] * birdmodels[0].delta
    
    # lower_bounds = [2.7, 0.50, 0.07, 0.01]
    # upper_bounds = [3.4, 0.80, 0.16, 0.04]

    priors = np.zeros(np.shape(params[1]))

    # Flat priors for cosmological parameters
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
            b1, c2 = params[-2 * (nz - i) : -2 * (nz - i - 1)] if i != nz - 1 else params[-2 * (nz - i) :]
            c4 = 0
            # b1, c2, c4 = params[-3 * (nz - i) : -3 * (nz - i - 1)] if i != nz - 1 else params[-3 * (nz - i) :]
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = (
                params[-10 * (nz - i) : -10 * (nz - i - 1)] if i != nz - 1 else params[-10 * (nz - i) :]
            )

        # Flat prior for b1
        priors += np.where(np.logical_or(b1 < 0.0, b1 > 4.0), -1.0e30, 0.0)

        # Flat prior for c2
        # priors += np.where(np.logical_or(c2 < -4.0, c2 > 4.0), -1.0e30, 0.0)
        priors += np.where(np.logical_or(c2 < -1000.0, c2 > 1000.0), -1.0e30, 0.0)


        # # Gaussian prior for c4
        # priors += -0.5 * c4 ** 2/2.0**2

        # if not birdmodels[0].pardict["do_marg"]:

        #     # Gaussian prior for b3 of width 2 centred on 0
        #     priors += -0.5 * 0.25 * b3 ** 2

        #     # Gaussian prior for cct of width 2 centred on 0
        #     priors += -0.5 * 0.25 * cct ** 2

        #     # Gaussian prior for cr1 of width 4 centred on 0
        #     priors += -0.5 * cr1 ** 2 / 10**2

        #     # Gaussian prior for cr1 of width 4 centred on 0
        #     priors += -0.5 * cr2 ** 2 / 10**2

        #     # Gaussian prior for ce1 of width 2 centred on 0
        #     priors += -0.5 * 0.25 * ce1 ** 2

        #     # Gaussian prior for cemono of width 2 centred on 0
        #     priors += -0.5 * 0.25 * cemono ** 2

        #     # Gaussian prior for cequad of width 2 centred on 0
        #     priors += -0.5 * cequad ** 2/10**2

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


def lnlike(params, birdmodels, fittingdata, plt):

    onedflag = False
    if params.ndim == 1:
        onedflag = True
        params = params.reshape((-1, len(params)))
    params = params.T
    # print(np.shape(params))

    # Get the bird model
    ln10As, h, omega_cdm, omega_b = params[:4]
    # ln10As, h, omega_cdm, omega_b = np.repeat(np.array([3.0364, 0.6736, 0.1200, 0.02237]), np.shape(params)[1]).reshape(4, np.shape(params)[1])

    Picount = 0
    P_models, Plins, Ploops = [], [], []
    P_model_lins, P_model_loops = [], []
    nmarg = len(birdmodels[0].eft_priors)
    if onebin == True:
        nz = 1
    else:
        nz = len(pardict['z_pk'])
        
    Pi_full = np.zeros((nz * len(birdmodels[0].eft_priors), len(fittingdata.data["fit_data"]), len(ln10As)))
    for i in range(nz):
        # shot_noise_fid = (1.0/birdmodels[i].correlator.birds[i].co.nd)
        
        if onebin == True:
            i = redindex
            model_index = 0
        else:
            model_index = i
            
        shot_noise_fid = (1.0/3e-4)
        # shot_noise_fid = (1.0)
        if onebin == False:
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
        else:
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
            
        if birdmodels[0].pardict["do_marg"]:
            # counter = -3 * (nz - i)
            # b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
            # b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
            
            counter = -2 * (nz - i)
            b2 = (params[counter + 1]) / np.sqrt(2.0)
            b4 = (params[counter + 1]) / np.sqrt(2.0)
            margb = np.zeros(np.shape(params)[1])
            bs = np.array(
                [
                    params[counter],
                    b2,
                    margb,
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
            counter = -10 * (nz - i)
            b2 = (params[counter + 1] + params[counter + 3]) / np.sqrt(2.0)
            b4 = (params[counter + 1] - params[counter + 3]) / np.sqrt(2.0)
            bs = np.array(
                [
                    params[counter],
                    b2,
                    params[counter + 2],
                    b4,
                    params[counter + 4],
                    params[counter + 5],
                    params[counter + 6],
                    # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
                    # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
                    # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
                    params[counter + 7] * shot_noise_ratio,
                    params[counter + 8] * shot_noise_ratio,
                    params[counter + 9] * shot_noise_ratio,
                ]
            )

        Plin, Ploop = birdmodels[model_index].compute_pk(np.array([ln10As, h, omega_cdm, omega_b]))
        P_model_lin, P_model_loop, P_model_interp = birdmodels[model_index].compute_model(
            bs, Plin, Ploop, fittingdata.data["x_data"][i]
        )
        # Pi = birdmodels[i].get_Pi_for_marg(
        #     Ploop, bs[0], float(fittingdata.data["shot_noise"][i]), fittingdata.data["x_data"][i]
        # )
        Pi = birdmodels[model_index].get_Pi_for_marg(
            Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][i]
        )

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
                bs_analytic = birdmodels[0].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data, onebin=onebin, eft_priors= eft_priors_all)
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
                P_model_lin, P_model_loop, P_model_interp = birdmodels[model_index].compute_model(
                    bs, Plins[i], Ploops[i], fittingdata.data["x_data"][i]
                )
                P_models.append(P_model_interp)
                P_model_lins.append(P_model_lin)
                P_model_loops.append(P_model_loop)
        if birdmodels[0].pardict["do_marg"] == 1:
            chi_squared = birdmodels[0].compute_chi2_marginalised(P_model, Pi_full, fittingdata.data, onebin=onebin, eft_priors=eft_priors_all)
            if plt is not None:
                chi_squared_print = birdmodels[0].compute_chi2(np.concatenate(P_models), fittingdata.data)
        else:
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

    if onedflag:
        return -0.5 * chi_squared[0]
    else:
        return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    plot_flag = int(sys.argv[2])
    jobid = int(sys.argv[3])
    try:
        redindex = int(sys.argv[4])
        print('Using redshift bin '+ str(redindex))
        onebin = True
    except:
        print('Using all redshift bins')
        onebin = False
    
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

    # Set up the data
    fittingdata = FittingData(pardict)
    
    n_sims = [978, 1000, 1000]
    # n_sims = [1000, 1000, 1000]

    
    if onebin == False:
        
        # Apply the Hartlap correction to the data as if each redshift bin used independent EZmocks
        # Works because the full matrix is block diagonal
        
        hartlap = [(ns - fittingdata.data["ndata"][i] - 2.0) / (ns - 1.0) for i, ns in enumerate(n_sims)]
        cov_inv_new = copy.copy(fittingdata.data["cov_inv"])
        for i, (nd, ndcum) in enumerate(
            zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
        ):
            cov_inv_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= hartlap[i]
            
        # Overwrite with the corrected ones
        fittingdata.data["cov_inv"] = cov_inv_new
        fittingdata.data["chi2data"] = np.dot(
            fittingdata.data["fit_data"], np.dot(fittingdata.data["cov_inv"], fittingdata.data["fit_data"])
        )
        fittingdata.data["invcovdata"] = np.dot(fittingdata.data["fit_data"], fittingdata.data["cov_inv"])
        
        nz = len(pardict['z_pk'])
        keyword = '_all_old'
    else:
        length_all = []
        for i in range(len(pardict['z_pk'])):
            length = len(fittingdata.data["x_data"][i][0]) + len(fittingdata.data["x_data"][i][1])
            if pardict['do_hex'] == True:
                length += len(fittingdata.data["x_data"][i][2])
            length_all.append(length)
        
        length_start = 0
        length_end = 0
        for i in range(np.int32(redindex+1)):
            if i == 0:
                length_start += 0    
            else:
                length_start += length_all[i-1]
            length_end += length_all[i]   
            
        print(length_start, length_end)
        
        cov_part = fittingdata.data['cov'][length_start:length_end, length_start:length_end]
        
        hartlap = (n_sims[redindex] - fittingdata.data["ndata"][redindex] - 2.0) / (n_sims[redindex] - 1.0)
        
        cov_part = cov_part*hartlap
        
        fitdata_part = fittingdata.data['fit_data'][length_start:length_end]
        
        cov_lu, pivots, cov_part_inv, info = lapack.dgesv(cov_part, np.eye(len(cov_part)))
        
        chi2data_part = np.dot(fitdata_part, np.dot(cov_part_inv, fitdata_part))
        invcovdata_part = np.dot(fitdata_part, cov_part_inv)
        
        fittingdata.data['cov'] = cov_part
        fittingdata.data['cov_inv'] = cov_part_inv
        fittingdata.data['chi2data'] = chi2data_part
        fittingdata.data['invcovdata'] = invcovdata_part
        fittingdata.data['fit_data'] = fitdata_part
        
        nz = 1 
        
        if single_mock == False:
            keyword = '_bin_'+str(redindex) + '_mean'
        else:
            keyword = '_bin_'+str(redindex) + '_mock_' + str(mock_num)
            
    keyword = keyword + '_noresum_'

    # Set up the BirdModels
    birdmodels = []
    for i in range(nz):
        #This is the default shot-noise in the pybird.py 
        shot_noise_fid = (1.0/3e-4)
        if onebin ==True:
            i = redindex
            print('redshift bin ' + str(i))
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
        else:
            shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
            
        model = BirdModel(pardict, redindex=i)
        # model.eft_priors = np.array([0.002, 0.002, 0.002, 2.0, 0.2, 0.0002, 0.0002])
        # model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio])
        # model.eft_priors = np.array([2.0, 2.0, 10.0, 10.0, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio, 10.0/shot_noise_ratio])
        # model.eft_priors = np.array([2.0, 8.0, 10.0, 10.0, 0.012/shot_noise_ratio, 10.0, 10.0])
        # model.eft_priors = np.array([10.0, 500.0, 5000.0, 5000.0, 40.0/shot_noise_ratio, 40.0, 100.0])
        model.eft_priors = np.array([4.0, 40.0, 400.0, 400.0, 0.24/shot_noise_ratio, 4.0, 20.0])
        print(model.eft_priors)
        birdmodels.append(model)
        
    if onebin == False:
        eft_priors_all = np.hstack([birdmodels[i].eft_priors for i in range(nz)])
        # print(eft_priors_all)
    else:
        eft_priors_all = birdmodels[0].eft_priors

    # # Read in and create a Planck prior covariance matrix
    # Planck_file = "/Volumes/Work/UQ/CAMB/COM_CosmoParams_fullGrid_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
    # planck_mean, planck_cov, planck_icov = get_Planck(Planck_file, 4)

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata, plotindex=plot_flag - 1)

    start = [
        [birdmodels[0].valueref[0], birdmodels[0].valueref[1], birdmodels[0].valueref[2], birdmodels[0].valueref[3]],
    ]
    if birdmodels[0].pardict["do_marg"]:
        for i in range(nz):
            start.append([1.8, 0.5])
            # start.append([1.8, 0.5, 0.5])
    else:
        for i in range(nz):
            start.append([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            # start = ([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    start = np.concatenate(start)

    # Does an optimization
    print(start)
    # result = do_optimization(lambda *args: -lnpost(*args), start)

    # Does an MCMC
    do_emcee(lnpost, start)
    # do_dynesty(lnlike, lnprior_transform, start, jobid)

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
