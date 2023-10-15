import numpy as np
import sys
from configobj import ConfigObj
from multiprocessing import Pool

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import format_pardict, read_chain_backend, BirdModel, do_optimization, get_Planck


def read_chain(pardict, hybrid):

    # Reads in a chain containing template or hybrid fit parameters, and returns the mean data vector and
    # inverse covariance matrix for fitting with cosmological parameters. Assumes the data
    # are suitably Gaussian.

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    if hybrid:
        fmt_str = (
            "%s_%s_%2dhex%2d_%s_%s_%s_hybrid.hdf5"
            if pardict["do_corr"]
            else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_hybrid.hdf5"
        )
    else:
        fmt_str = (
            "%s_%s_%2dhex%2d_%s_%s_%s_template.hdf5"
            if pardict["do_corr"]
            else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_template.hdf5"
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
    print(chainfile)
    burntin, bestfit, like = read_chain_backend(chainfile)
    burntin = burntin[:, :4] if hybrid else burntin[:, :3]

    return np.mean(burntin, axis=0), np.linalg.inv(np.cov(burntin, rowvar=False))


def do_emcee(func, start):

    import emcee

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    nparams = len(start)
    nwalkers = nparams * 8

    begin = [[(0.01 * (np.random.rand() - 0.5) + 1.0) * start[j] for j in range(len(start))] for i in range(nwalkers)]

    marg_str = "marg" if pardict["do_marg"] else "all"
    hex_str = "hex" if pardict["do_hex"] else "nohex"
    dat_str = "xi" if pardict["do_corr"] else "pk"
    if hybrid:
        fmt_str = (
            "%s_%s_%2dhex%2d_%s_%s_%s_hybrid_planck"
            if pardict["do_corr"]
            else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_hybrid_planck"
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
    oldfile = chainfile + ".hdf5"
    newfile = chainfile + ".dat"
    print(oldfile)

    # Set up the backend
    backend = emcee.backends.HDFBackend(oldfile)
    backend.reset(nwalkers, nparams)

    with Pool() as pool:

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, nparams, func, pool=pool, backend=backend)

        # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
        max_iter = 30000
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
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
                break
            old_tau = tau
            index += 1

    burntin, bestfit, like = read_chain_backend(oldfile)

    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    # Loop over the parameters in the chain and use the grids to compute the derived parameters
    chainvals = []
    for i, (vals, loglike) in enumerate(zip(burntin, like)):
        if i % 1000 == 0:
            print(i)
        ln10As, h, omega_cdm, omega_b = vals[:4]
        # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm
        if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
            np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
        ):
            continue
        Om, Da, Hz, f, sigma8, sigma8_0, sigma12, r_d = birdmodel.compute_params([ln10As, h, omega_cdm, omega_b])
        alpha_perp = (Da / h) * (float(pardict["h"]) / Da_fid) * (r_d_fid / (r_d))
        alpha_par = (float(pardict["h"]) * Hz_fid) / (h * Hz) * (r_d_fid / (r_d))
        chainvals.append(
            (
                ln10As,
                100.0 * h,
                omega_cdm,
                omega_b,
                alpha_perp,
                alpha_par,
                Om,
                2997.92458 * Da / h,
                100.0 * h * Hz,
                f,
                sigma8,
                sigma8_0,
                sigma12,
                loglike,
            )
        )

    np.savetxt(newfile, np.array(chainvals))


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
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta

    # Flat priors for cosmological parameters
    if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
        np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
    ):
        return -np.inf

    # BBN (D/H) inspired prior on omega_b
    # omega_b_prior = -0.5 * (omega_b - birdmodel.valueref[3]) ** 2 / 0.00037 ** 2
    omega_b_prior = 0.0

    # Planck prior
    diff = params[:4] - birdmodel.valueref
    Planck_prior = -0.5 * diff @ planck_icov @ diff
    # Planck_prior = 0.0

    return omega_b_prior + Planck_prior


def lnlike(params):

    ln10As, h, omega_cdm, omega_b = params[:4]
    # ln10As, h, omega_cdm = params[:3]
    # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm

    Om, Da, Hz, f, sigma8, sigma8_0, sigma12, r_d = birdmodel.compute_params([ln10As, h, omega_cdm, omega_b])

    # Factors of little h required as Da and Hz in Grid.py have little h scaled out.
    alpha_perp = (Da / h) * (float(pardict["h"]) / Da_fid) * (r_d_fid / (r_d))
    alpha_par = (float(pardict["h"]) * Hz_fid) / (h * Hz) * (r_d_fid / (r_d))

    model = np.array([alpha_perp, alpha_par, f * sigma8])
    if hybrid:
        model = np.concatenate([model, [omega_cdm + omega_b + birdmodel.omega_nu]])

    chi_squared = (fitdata - model) @ fitcov @ (fitdata - model)

    return -0.5 * chi_squared


if __name__ == "__main__":

    # Code to generate a power spectrum template at fixed cosmology using pybird, then fit the AP parameters and fsigma8
    # First read in the config file
    configfile = sys.argv[1]
    hybrid = int(sys.argv[2])
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the BirdModel
    birdmodel = BirdModel(pardict)

    # Read in and create a Planck prior covariance matrix
    Planck_file = "/Volumes/Work/UQ/CAMB/COM_CosmoParams_fullGrid_R3.01/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing"
    planck_mean, planck_cov, planck_icov = get_Planck(Planck_file, 4)

    # Compute the values at the central point
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_class(pardict)

    # Read in the chain and get the template parameters
    fitdata, fitcov = read_chain(pardict, hybrid)

    start = np.array([birdmodel.valueref[0], birdmodel.valueref[1], birdmodel.valueref[2], birdmodel.valueref[3]])

    # Does an optimization
    # result = do_optimization(lambda *args: -lnpost(*args), start)

    # Does an MCMC and then post-processes to get some derived params
    do_emcee(lnpost, start)
