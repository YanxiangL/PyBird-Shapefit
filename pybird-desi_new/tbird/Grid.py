import numpy as np
import copy
import camb
from classy import Class
from scipy.special import hyp2f1


def smooth_hinton2017(ks, pk, degree=13, sigma=1, weight=0.5):
    """ Smooth power spectrum based on Hinton 2017 polynomial method """
    log_ks = np.log(ks)
    log_pk = np.log(pk)
    index = np.argmax(pk)
    maxk2 = log_ks[index]
    gauss = np.exp(-0.5 * np.power(((log_ks - maxk2) / sigma), 2))
    w = np.ones(pk.size) - weight * gauss
    z = np.polyfit(log_ks, log_pk, degree, w=w)
    p = np.poly1d(z)
    polyval = p(log_ks)
    pk_smoothed = np.exp(polyval)
    return pk_smoothed


def grid_properties(pardict):
    """Computes some useful properties of the grid given the parameters read from the input file

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    valueref: np.array
        An array of the central values of the grid
    delta: np.array
        An array containing the grid cell widths
    flattenedgrid: np.array
        The number of grid cells from the center for each coordinate, flattened
    truecrd: list of np.array
        A list containing 1D numpy arrays for the values of the cosmological parameters along each grid axis
    """

    order = float(pardict["order"])
    valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    delta = np.fabs(np.array(pardict["dx"], dtype=np.float))
    squarecrd = [np.arange(-order, order + 1) for l in pardict["freepar"]]
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
    squaregrid = np.array(np.meshgrid(*squarecrd, indexing="ij"))
    flattenedgrid = squaregrid.reshape([len(pardict["freepar"]), -1]).T

    return valueref, delta, flattenedgrid, truecrd


def grid_properties_template_hybrid(pardict, fsigma8, omegamh2):
    """Computes some useful properties of the grid given the parameters read from the input file

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    valueref: np.array
        An array of the central values of the grid
    delta: np.array
        An array containing the grid cell widths
    flattenedgrid: np.array
        The number of grid cells from the center for each coordinate, flattened
    truecrd: list of np.array
        A list containing 1D numpy arrays for the values of the cosmological parameters along each grid axis
    """

    order = float(pardict["template_order"])
    valueref = np.array([1.0, 1.0, fsigma8, omegamh2])
    delta = np.array(pardict["template_dx"], dtype=np.float) * valueref
    squarecrd = [np.arange(-order, order + 1) for l in range(4)]
    truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(4)]
    squaregrid = np.array(np.meshgrid(*squarecrd, indexing="ij"))
    flattenedgrid = squaregrid.reshape([4, -1]).T

    return valueref, delta, flattenedgrid, truecrd


def run_camb(pardict, redindex=0):
    """Runs an instance of CAMB given the cosmological parameters in pardict

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    kin: np.array
        the k-values of the CAMB linear power spectrum
    Plin: np.array
        The linear power spectrum
    Da: float
        The angular diameter distance to the value of z_pk in the config file, without the factor c/H_0
    H: float
        The Hubble parameter at z_pk, without the factor H_0
    fN: float
        The scale-independent growth rate at z_pk
    """

    parlinear = copy.deepcopy(pardict)

    uniquez, uniquez_ind = np.unique(parlinear["z_pk"], return_inverse=True)

    # Set the CAMB parameters
    pars = camb.CAMBparams()
    if "A_s" not in parlinear.keys():
        if "ln10^{10}A_s" in parlinear.keys():
            parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
        else:
            print("Error: Neither ln10^{10}A_s nor A_s given in config file")
            exit()
    if "H0" not in parlinear.keys():
        if "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
        else:
            print("Error: Neither H0 nor h given in config file")
            exit()
    if "w0_fld" in parlinear.keys():
        pars.set_dark_energy(w=float(parlinear["w0_fld"]), dark_energy_model="fluid")
    pars.InitPower.set_params(As=float(parlinear["A_s"]), ns=float(parlinear["n_s"]))
    pars.set_matter_power(
        redshifts=np.concatenate([[0.0], uniquez]),
        kmax=float(parlinear["P_k_max_h/Mpc"]),
        nonlinear=False,
    )
    pars.set_cosmology(
        H0=float(parlinear["H0"]),
        omch2=float(parlinear["omega_cdm"]),
        ombh2=float(parlinear["omega_b"]),
        omk=float(parlinear["Omega_k"]),
        tau=float(parlinear["tau_reio"]),
        mnu=float(parlinear["Sum_mnu"]),
        neutrino_hierarchy=parlinear["nu_hierarchy"],
    )
    pars.NonLinear = camb.model.NonLinear_none

    # Run CAMB
    results = camb.get_results(pars)

    # Get the power spectrum, duplicated where input redshifts are duplicated
    kin, zin, Plin = results.get_matter_power_spectrum(
        minkh=9.9e-5, maxkh=float(parlinear["P_k_max_h/Mpc"]), npoints=200, var1="delta_nonu", var2="delta_nonu"
    )
    Plin = Plin[1:][uniquez_ind]  # Removes z=0.0 addition, and duplicates
    print(zin)

    # Get some derived quantities
    Omega_m = results.get_Omega("cdm") + results.get_Omega("baryon") + results.get_Omega("nu")
    Da = results.angular_diameter_distance(parlinear["z_pk"]) * float(parlinear["H0"]) / 299792.458
    H = results.hubble_parameter(parlinear["z_pk"]) / float(parlinear["H0"])
    # This weird indexing flips the order back so z=0.0 is first, removes z=0.0, and duplicates for duplicated redshifts.
    fsigma8 = results.get_fsigma8()[::-1][1:][uniquez_ind]
    # sigma8 = results.get_sigmaR(8.0, var1="delta_nonu", var2="delta_nonu")[::-1]
    # sigma12 = results.get_sigmaR(12.0, var1="delta_nonu", var2="delta_nonu", hubble_units=False)[::-1][1:][uniquez_ind]
    sigma8 = results.get_sigmaR(8.0)[::-1]
    sigma12 = results.get_sigmaR(12.0, hubble_units=False)[::-1][1:][uniquez_ind]
    r_d = results.get_derived_params()["rdrag"]

    # Get growth factor from sigma8 ratios
    D = sigma8[1:][uniquez_ind] / sigma8[0]
    f = fsigma8 / sigma8[1:][uniquez_ind]

    return (
        kin,
        Plin,
        Omega_m,
        Da,
        H,
        D,
        f,
        sigma8[1:][uniquez_ind],
        sigma8[0],
        sigma12,
        r_d,
    )


def run_class(pardict, redindex=0):
    """Runs an instance of CAMB given the cosmological parameters in pardict

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file

    Returns
    -------
    kin: np.array
        the k-values of the CAMB linear power spectrum
    Plin: np.array
        The linear power spectrum
    Da: float
        The angular diameter distance to the value of z_pk in the config file, without the factor c/H_0
    H: float
        The Hubble parameter at z_pk, without the factor H_0
    fN: float
        The scale-independent growth rate at z_pk
    """

    parlinear = copy.deepcopy(pardict)
    if int(parlinear["N_ncdm"]) == 2:
        parlinear["m_ncdm"] = parlinear["m_ncdm"][0] + "," + parlinear["m_ncdm"][1]
    if int(parlinear["N_ncdm"]) == 3:
        parlinear["m_ncdm"] = parlinear["m_ncdm"][0] + "," + parlinear["m_ncdm"][1] + "," + parlinear["m_ncdm"][2]

    # Set the CLASS parameters
    M = Class()
    if "A_s" not in parlinear.keys():
        if "ln10^{10}A_s" in parlinear.keys():
            parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
        else:
            print("Error: Neither ln10^{10}A_s nor A_s given in config file")
            exit()
    if "H0" not in parlinear.keys():
        if "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
        else:
            print("Error: Neither H0 nor h given in config file")
            exit()
    M.set(
        {
            "A_s": float(parlinear["A_s"]),
            "n_s": float(parlinear["n_s"]),
            "H0": float(parlinear["H0"]),
            "omega_b": float(parlinear["omega_b"]),
            "omega_cdm": float(parlinear["omega_cdm"]),
            "N_ur": float(parlinear["N_ur"]),
            "N_ncdm": int(parlinear["N_ncdm"]),
            "Omega_k": float(parlinear["Omega_k"]),
        }
    )
    if int(parlinear["N_ncdm"]) > 0:
        M.set({"m_ncdm": parlinear["m_ncdm"]})
    M.set(
        {
            "output": "mPk",
            "P_k_max_h/Mpc": float(parlinear["P_k_max_h/Mpc"]),
            "z_max_pk": float(parlinear["z_pk"][redindex]),
        }
    )
    M.compute()

    kin = np.logspace(np.log10(9.9e-5), np.log10(float(parlinear["P_k_max_h/Mpc"])), 200)

    # Don't use pk_cb_array - gives weird discontinuties for k < 1.0e-3 and non-zero curvature.
    if int(parlinear["N_ncdm"]) > 0:
        Plin = np.array([[M.pk_cb_lin(ki * M.h(), zi) * M.h() ** 3 for ki in kin] for zi in parlinear["z_pk"]])
    else:
        Plin = np.array([[M.pk_lin(ki * M.h(), zi) * M.h() ** 3 for ki in kin] for zi in parlinear["z_pk"]])

    # Get some derived quantities
    Omega_m = M.Om_m(0.0)
    Da = np.array([M.angular_distance(z) * M.Hubble(0.0) for z in parlinear["z_pk"]])
    H = np.array([M.Hubble(z) / M.Hubble(0.0) for z in parlinear["z_pk"]])
    D = np.array([M.scale_independent_growth_factor(z) for z in parlinear["z_pk"]])
    f = np.array([M.scale_independent_growth_factor_f(z) for z in parlinear["z_pk"]])
    sigma8 = np.array([M.sigma(8.0 / M.h(), z) for z in parlinear["z_pk"]])
    sigma8_0 = M.sigma(8.0 / M.h(), 0.0)
    sigma12 = np.array([M.sigma(12.0, z) for z in parlinear["z_pk"]])
    r_d = M.rs_drag()

    return kin, Plin, Omega_m, Da, H, D, f, sigma8, sigma8_0, sigma12, r_d


if __name__ == "__main__":

    import sys

    sys.path.append("../")
    import matplotlib.pyplot as plt
    from pybird_dev import pybird
    from configobj import ConfigObj
    from fitting_codes.fitting_utils import format_pardict, FittingData

    # Read in the config file, job number and total number of jobs
    configfile = sys.argv[1]
    pardict = format_pardict(ConfigObj(configfile))

    # Compute some stuff for the grid based on the config file
    valueref, delta, flattenedgrid, _ = grid_properties(pardict)

    # Get some cosmological values on the grid
    fig = plt.figure(0)
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])

    for i in range(-4, 5):
        theta = [0, 0, 0, 0, i]
        parameters = copy.deepcopy(pardict)
        truetheta = valueref + theta * delta
        for k, var in enumerate(pardict["freepar"]):
            parameters[var] = truetheta[k]

        for k, var in enumerate(pardict["freepar"]):
            parameters[var] = truetheta[k]
        (
            kin_camb,
            Pin_camb,
            Om_camb,
            Da_camb,
            Hz_camb,
            DN_camb,
            fN_camb,
            sigma8_camb,
            sigma8_0_camb,
            sigma12_camb,
            r_d_camb,
        ) = run_camb(parameters)
        (
            kin_class,
            Pin_class,
            Om_class,
            Da_class,
            Hz_class,
            DN_class,
            fN_class,
            sigma8_class,
            sigma8_0_class,
            sigma12_class,
            r_d_class,
        ) = run_class(parameters)

        ax.plot(kin_camb, Pin_class[0])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(9.0e-5, 1.1 * float(pardict["P_k_max_h/Mpc"]))
    ax.set_ylim(1.0e-4, 1.0e4)
    plt.show()
