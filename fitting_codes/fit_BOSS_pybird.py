import numpy as np
import sys
from scipy.stats import norm
from configobj import ConfigObj
from multiprocessing import Pool
from chainconsumer import ChainConsumer

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
    fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_planck.hdf5"
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
        max_iter = 20000
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
    fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s.hdf5" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_planck.hdf5"
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
    # ln10As, h, omega_cdm, omega_b, omega_k = birdmodels[0].valueref[:, None]
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    lower_bounds = birdmodels[0].valueref - birdmodels[0].pardict["order"] * birdmodels[0].delta
    upper_bounds = birdmodels[0].valueref + birdmodels[0].pardict["order"] * birdmodels[0].delta

    priors = np.zeros(np.shape(params[1]))

    # Flat priors for cosmological parameters
    for i, param in enumerate([ln10As, h, omega_cdm, omega_b]):
        priors += np.where(np.logical_or(param < lower_bounds[i], param > upper_bounds[i]), -1.0e30, 0.0)

    # BBN (D/H) inspired prior on omega_b
    omega_b_prior = -0.5 * (omega_b - 0.02230) ** 2 / 0.00028 ** 2
    # omega_b_prior = 0.0
    priors += omega_b_prior

    # Planck prior
    # diff = params[:4] - birdmodel.valueref
    # Planck_prior = -0.5 * diff @ planck_icov @ diff
    priors += np.zeros(np.shape(params[1]))

    nz = len(birdmodels[0].pardict["z_pk"])
    for i in range(nz):
        if birdmodels[0].pardict["do_marg"]:
            b1, c2, c4 = params[-3 * (nz - i) : -3 * (nz - i - 1)] if i != nz - 1 else params[-3 * (nz - i) :]
            # b1, c2 = params[-2 * (nz - i) : -2 * (nz - i - 1)] if i != nz - 1 else params[-2 * (nz - i) :]
        else:
            b1, c2, b3, c4, cct, cr1, cr2, ce1, cemono, cequad = (
                params[-10 * (nz - i) : -10 * (nz - i - 1)] if i != nz - 1 else params[-10 * (nz - i) :]
            )

        # Flat prior for b1
        priors += np.where(np.logical_or(b1 < 0.0, b1 > 3.0), -1.0e30, 0.0)

        # Flat prior for c2
        priors += np.where(np.logical_or(c2 < -4.0, c2 > 4.0), -1.0e30, 0.0)

        # Gaussian prior for c4
        priors += -0.5 * 0.25 * c4 ** 2

        if not birdmodels[0].pardict["do_marg"]:

            # Gaussian prior for b3 of width 2 centred on 0
            priors += -0.5 * 0.25 * b3 ** 2

            # Gaussian prior for cct of width 2 centred on 0
            priors += -0.5 * 0.25 * cct ** 2

            # Gaussian prior for cr1 of width 4 centred on 0
            priors += -0.5 * cr1 ** 2 / 16.0

            # Gaussian prior for cr1 of width 4 centred on 0
            priors += -0.5 * cr2 ** 2 / 16.0

            # Gaussian prior for ce1 of width 2 centred on 0
            priors += -0.5 * 0.25 * ce1 ** 2

            # Gaussian prior for cemono of width 2 centred on 0
            priors += -0.5 * 0.25 * cemono ** 2

            # Gaussian prior for cequad of width 2 centred on 0
            priors += -0.5 * 0.25 * cequad ** 2

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
    params[3] = norm.ppf(u[3], loc=0.02235, scale=0.00028)

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

    # Get the bird model
    ln10As, h, omega_cdm, omega_b = params[:4]
    # ln10As, h, omega_cdm, omega_b, omega_k = birdmodels[0].valueref[:, None]
    # omega_k = [0.0]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    Picount = 0
    P_models, Plins, Ploops = [], [], []
    P_model_lins, P_model_loops = [], []
    nmarg = len(birdmodels[0].eft_priors)
    nz = len(birdmodels[0].pardict["z_pk"])
    Pi_full = np.zeros((nz * len(birdmodels[0].eft_priors), len(fittingdata.data["fit_data"]), len(ln10As)))
    for i in range(nz):
        if birdmodels[0].pardict["do_marg"]:
            counter = -3 * (nz - i)
            b2 = (params[counter + 1] + params[counter + 2]) / np.sqrt(2.0)
            b4 = (params[counter + 1] - params[counter + 2]) / np.sqrt(2.0)
            # counter = -2 * (nz - i)
            # b2 = (params[counter + 1]) / np.sqrt(2.0)
            # b4 = (params[counter + 1]) / np.sqrt(2.0)
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
                    params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
                    params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
                    params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
                ]
            )

        Plin, Ploop = birdmodels[i].compute_pk(np.array([ln10As, h, omega_cdm, omega_b]))
        P_model_lin, P_model_loop, P_model_interp = birdmodels[i].compute_model(
            bs, Plin, Ploop, fittingdata.data["x_data"][i]
        )
        Pi = birdmodels[i].get_Pi_for_marg(
            Ploop, bs[0], float(fittingdata.data["shot_noise"][i]), fittingdata.data["x_data"][i]
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
            bs_analytic = birdmodels[0].compute_bestfit_analytic(P_model, Pi_full, fittingdata.data)
            for i in range(nz):
                counter = -3 * (nz - i)
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
                        bs_analytic[7 * i + 4] * float(fittingdata.data["shot_noise"][i]),
                        bs_analytic[7 * i + 5] * float(fittingdata.data["shot_noise"][i]),
                        bs_analytic[7 * i + 6] * float(fittingdata.data["shot_noise"][i]),
                    ]
                )
                P_model_lin, P_model_loop, P_model_interp = birdmodels[i].compute_model(
                    bs, Plins[i], Ploops[i], fittingdata.data["x_data"][i]
                )
                P_models.append(P_model_interp)
                P_model_lins.append(P_model_lin)
                P_model_loops.append(P_model_loop)
        if birdmodels[0].pardict["do_marg"] == 1:
            chi_squared = birdmodels[0].compute_chi2_marginalised(P_model, Pi_full, fittingdata.data)
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
    # else:
    #    if np.random.rand() < 0.01:
    #        print(params[:, 0], chi_squared[0], len(fittingdata.data["fit_data"]))

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
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the data
    fittingdata = FittingData(pardict)

    # Set up the BirdModels
    birdmodels = []
    for i in range(len(pardict["z_pk"])):
        # birdmodels.append(BirdModel(pardict, direct=True, redindex=i, window=fittingdata.data["windows"][i]))
        birdmodels.append(BirdModel(pardict, redindex=i))

    # Read in and create a Planck prior covariance matrix
    Planck_file = "/Volumes/Work/UQ/CAMB/COM_CosmoParams_fullGrid_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
    planck_mean, planck_cov, planck_icov = get_Planck(Planck_file, 4)

    # Plotting (for checking/debugging, should turn off for production runs)
    plt = None
    if plot_flag:
        plt = create_plot(pardict, fittingdata, plotindex=plot_flag - 1)

    start = [
        [birdmodels[0].valueref[0], birdmodels[0].valueref[1], birdmodels[0].valueref[2], birdmodels[0].valueref[3]],
    ]
    if birdmodels[0].pardict["do_marg"]:
        for i in range(len(pardict["z_pk"])):
            start.append([1.8, 0.2, 0.2])
    else:
        for i in range(len(pardict["z_pk"])):
            start.append([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    start = np.concatenate(start)

    # Does an optimization
    """lower_bounds = birdmodels[0].valueref - birdmodels[0].pardict["order"] * birdmodels[0].delta
    upper_bounds = birdmodels[0].valueref + birdmodels[0].pardict["order"] * birdmodels[0].delta
    start = (
        (lower_bounds[0], upper_bounds[0]),
        (lower_bounds[1], upper_bounds[1]),
        (lower_bounds[2], upper_bounds[2]),
        (lower_bounds[3], upper_bounds[3]),
        (lower_bounds[4], upper_bounds[4]),
        (0.0, 3.0),
        (-4.0, 4.0),
        (0.0, 3.0),
        (-4.0, 4.0),
        (0.0, 3.0),
        (-4.0, 4.0),
        (0.0, 3.0),
        (-4.0, 4.0),
    )"""
    # result = do_optimization(lambda *args: -lnpost(*args), start)

    # Does an MCMC
    do_emcee(lnpost, start)
    # do_dynesty(lnlike, lnprior_transform, start, jobid)

    # Does some plotting of a previously run chain
    c = ChainConsumer()

    chainfiles = [
        "/Volumes/Work/UQ/DESI/cBIRD/BOSSdata/fits/BOSS_z1z2z3_NGCSGC_pk_0.20hex0.10_3order_hex_marg_planck.hdf5"
    ]

    bestfits = []
    for chaini, chainfile in enumerate(chainfiles):
        burntin, bestfit, like = read_chain_backend(chainfile)
        paramnames = [r"$\mathrm{ln}(10^{10}A_{s})$", r"$h$", r"$\omega_{cdm}$", r"$\omega_{b}$"]
        c.add_chain(burntin[:, :4], parameters=paramnames, posterior=like)
        bestfits.append(bestfit)

    print(bestfits)
    c.configure(bar_shade=True)
    c.add_marker(bestfits[0][:4], parameters=paramnames, marker_size=100, marker_style="*", color="r")
    c.plotter.plot(truth=birdmodels[0].valueref, display=True)
    print(c.analysis.get_summary())
