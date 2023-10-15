import numpy as np
import copy
import camb
from classy import Class
from scipy.special import hyp2f1
import scipy.constants as conts



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
    else:
        parlinear["h"] = parlinear["H0"]/100.0
            
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
            'omega_ncdm': float(parlinear["omega_ncdm"]),
            'alpha_s': float(parlinear['alpha_s'])
        }
    )
    
    # M.set(
    #     {
    #         "A_s": float(parlinear["A_s"]),
    #         "H0": float(parlinear["H0"]),
    #         "omega_b": float(parlinear["omega_b"]),
    #         "omega_cdm": float(parlinear["omega_cdm"]),
    #     }
    # )
    
    if "w" in parlinear.keys():
        # parlinear['omega_fld'] = 1.0 - (float(parlinear["omega_b"]) + float(parlinear["omega_cdm"]))/(float(parlinear["h"])) - float(parlinear['Omega_k'])
        M.set(
            {"Omega_Lambda": 0.0,
             "w0_fld": float(parlinear["w"])
                })
        if "wa" in parlinear.keys():
            M.set(
                {"wa_fld": float(parlinear["wa"])
                })
            
    # print(float(parlinear["A_s"]), float(parlinear["H0"]), float(parlinear["omega_b"]), float(parlinear["omega_cdm"]), float(parlinear["w"]))
    
    # if int(parlinear["N_ncdm"]) > 0:
    #     M.set({"m_ncdm": parlinear["m_ncdm"]})
    M.set(
        {
            "output": "mPk",
            "P_k_max_h/Mpc": float(parlinear["P_k_max_h/Mpc"]),
            "z_max_pk": float(parlinear["z_pk"][redindex]),
        }
    )
    M.compute()
    
    # print(parlinear["z_pk"][redindex])

    kin = np.logspace(np.log10(9.9e-5), np.log10(float(parlinear["P_k_max_h/Mpc"])), 2000)
    
    # print(M.Omega_m(), M.Omega_Lambda(), M.Omega0_k(), M.get_current_derived_parameters(['Omega0_fld', 'm_ncdm_tot']), float(parlinear["w"]))

    # Don't use pk_cb_array - gives weird discontinuties for k < 1.0e-3 and non-zero curvature.
    if int(parlinear["N_ncdm"]) > 0:
        Plin = np.array([[M.pk_cb_lin(ki * M.h(), zi) * M.h() ** 3 for ki in kin] for zi in parlinear["z_pk"]])
    else:
        Plin = np.array([[M.pk_lin(ki * M.h(), zi) * M.h() ** 3 for ki in kin] for zi in parlinear["z_pk"]])
        
    # np.save('Plin_converge.npy', [kin, Plin])
    
    # Plin = np.array([[M.pk_lin(ki * M.h(), zi) * M.h() ** 3 for ki in kin] for zi in parlinear["z_pk"]])
    
    # Plin = EH98_old(M, kin, float(parlinear["z_pk"][redindex]), 1.0)*M.h()**3
    # Plin = Plin.reshape((1, 2000))
    
    # np.save('Class_Plin.npy', [kin, Plin])
    
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

def EH98_old(cosmo, kvector, redshift, scaling_factor):
    cdict = cosmo.get_current_derived_parameters(['z_d'])
    h = cosmo.h()
    H_at_z = cosmo.Hubble(redshift) * conts.c /1000. /(100.*h)
    Omm = cosmo.Omega_m()
    Omb = cosmo.Omega_b()
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
    alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu)*Nnu**0.2 + 0.169*fnu) \
                *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
    #eff_shape = (alpha_gamma + (1.-alpha_gamma)/(1+(0.43*kvector*rs)**4.))
    eff_shape = (np.sqrt(alpha_nu) + (1.-np.sqrt(alpha_nu))/(1+(0.43*kvector*rs)**4.))
    q0 = kvector/(keq/7.46e-2)/eff_shape
    betac = (1.-0.949*fnub)**(-1.)
    L0 = np.log(np.exp(1.)+1.84*np.sqrt(alpha_nu)*betac*q0)
    C0 = 14.4 + 325./(1+60.5*q0**1.08)
    T0 = L0/(L0+C0*q0**2.)
    if (fnu==0):
        yfs=0.
        qnu=3.92*q0
    else:
        yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6.))*(Nnu*q0/fnu)**2.
        qnu = 3.92*q0*np.sqrt(Nnu/fnu)
    D1 = (1.+zeq)/(1.+redshift)*Omm_at_z/2./(Omm_at_z**(4./7.) - OmLambda_at_z + (1.+Omm_at_z/2.)*(1.+OmLambda_at_z/70.))
    Dcbnu = (fcb**(0.7/pcb)+(D1/(1.+yfs))**0.7)**(pcb/0.7) * D1**(1.-pcb)
    Bk = 1. + 1.24*fnu**(0.64)*Nnu**(0.3+0.6*fnu)/(qnu**(-1.6)+qnu**0.8)
    Tcbnu = T0*Dcbnu/D1*Bk
    deltah = 1.94e-5 * Omm**(-0.785-0.05*np.log(Omm))*np.exp(-0.95*(ns-1)-0.169*(ns-1)**2.)
    Pk = 2*np.pi**2. * deltah**2. * (kvector)**(ns) * Tcbnu**2. * growth**2. /cosmo.Hubble(0)**(3.+ns)
    return Pk


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
