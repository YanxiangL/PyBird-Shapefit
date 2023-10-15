import os
import sys
import numpy as np
from scipy import interpolate
from configobj import ConfigObj
from chainconsumer import ChainConsumer
import scipy.constants as conts
from classy import Class

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import format_pardict, BirdModel, read_chain_backend
from tbird.computederivs import get_grids, get_ParamsTaylor

if __name__ == "__main__":

    # Reads in a chain containing cosmological parameters output from a fitting routine, and
    # converts the points to alpha_perp, alpha_par, f(z)*sigma8(z) and f(z)*sigma12(z)

    # First, read in the config file used for the fit
    configfile = sys.argv[1]
    try:
        redbin = int(sys.argv[2])
        keyword = '_bin_'+str(redbin)
    except:
        keyword = '_all'
    
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)

    # Set up the BirdModel
    birdmodel = BirdModel(pardict, redindex=redbin)
    birdmodel.eft_priors = np.array([0.002, 0.002, 0.002, 2.0, 2.0, 0.002, 0.002])
    
    z_data = pardict['z_pk'][redbin]

    # # Compute the values at the central point
    # if pardict["code"] == "CAMB":
    #     _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    # else:
    #     _, _, Om_fid, Da_fid, Hz_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_class(pardict)
    
    if pardict["code"] == "CAMB":
        _, _, Om_fid, Da_fid, Hz_fid, DN_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_camb(pardict)
    else:
        _, _, Om_fid, Da_fid, Hz_fid, DN_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12_fid, r_d_fid = run_class(pardict)
        
    h_convert = (birdmodel.valueref[1]*100*1000/conts.c)
        
    Om_fid = float(Om_fid)
    Da_fid = float(Da_fid[redbin])/h_convert*(1.0 + z_data)
    Hz_fid = float(Hz_fid[redbin])*h_convert*conts.c/1000.0
    DN_fid = float(DN_fid[redbin])
    sigma8_fid = float(sigma8_fid[redbin])
    sigma8_0_fid = float(sigma8_0_fid)
    sigma12_fid = float(sigma12_fid[redbin])
    r_d_fid = float(r_d_fid)
    
    print(Da_fid, Hz_fid, fN_fid[redbin]*sigma8_fid)

    # marg_str = "marg" if pardict["do_marg"] else "all"
    # hex_str = "hex" if pardict["do_hex"] else "nohex"
    # dat_str = "xi" if pardict["do_corr"] else "pk"
    # fmt_str = "%s_%s_%2dhex%2d_%s_%s_%s" if pardict["do_corr"] else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_planck"
    # fitlim = birdmodel.pardict["xfit_min"][0] if pardict["do_corr"] else birdmodel.pardict["xfit_max"][0]
    # fitlimhex = birdmodel.pardict["xfit_min"][2] if pardict["do_corr"] else birdmodel.pardict["xfit_max"][2]

    # taylor_strs = ["grid", "1order", "2order", "3order", "4order"]
    # chainfile = str(
    #     fmt_str
    #     % (
    #         birdmodel.pardict["fitfile"],
    #         dat_str,
    #         fitlim,
    #         fitlimhex,
    #         taylor_strs[pardict["taylor_order"]],
    #         hex_str,
    #         marg_str,
    #     )
    # )
    chainfile = "../../data/DESI_KP4_LRG_ELG_QSO_pk_0.15hex0.15_3order_hex_marg_kmin0p02_fewerbias" + keyword
    
    print(chainfile)
    oldfile = chainfile + ".hdf5"
    newfile = chainfile + "_converted_bin_"+str(redbin)+".dat"
    
    burntin, bestfit, like = read_chain_backend(oldfile)
    
    lower_bounds = birdmodel.valueref - birdmodel.pardict["order"] * birdmodel.delta
    upper_bounds = birdmodel.valueref + birdmodel.pardict["order"] * birdmodel.delta
    
    gridnames = np.loadtxt(pardict["gridname"], dtype=str)
    outgrids = np.loadtxt(pardict["outgrid"], dtype=str)
    name = pardict["code"].lower() + "-" + gridnames[redbin]
    print(name)
    
    if pardict['do_hex']:
        nmult = 3
    else:
        nmult = 2
        
    # paramsgrid, plingrid, ploopgrid = get_grids(pardict, outgrids[redbin], name, nmult=nmult, nout = nmult)
    paramsmod = np.load(
        os.path.join(outgrids[redbin], "DerParams_%s.npy" % name),
        allow_pickle=True,
    )

    # Loop over the parameters in the chain and use the grids to compute the derived parameters
    chainvals = []
    for i, (vals, loglike) in enumerate(zip(burntin, like)):
        ln10As, h, omega_cdm, omega_b = vals[:4]
        # ln10As, h, omega_cdm = vals[:3]
        # omega_b = birdmodel.valueref[3] / birdmodel.valueref[2] * omega_cdm
        if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
            np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
        ):
            continue
        # Om, Da, Hz, f, sigma8, sigma8_0, sigma12, r_d = birdmodel.compute_params([ln10As, h, omega_cdm, omega_b])
        # alpha_perp = (Da / h) * (float(pardict["h"]) / Da_fid) * (r_d_fid / (r_d))
        # alpha_par = (float(pardict["h"]) * Hz_fid) / (h * Hz) * (r_d_fid / (r_d))
        
        # Om, Da, Hz, DN, f, sigma8, sigma8_0, sigma12, r_d = birdmodel.compute_params([ln10As, h, omega_cdm, omega_b])
        
        # Da_new = Da/h_convert*(1.0 + z_data)
        # Hz_new = Hz*h_convert*conts.c/1000.0
        # alpha_perp = (Da_new) * (1.0 / Da_fid) * (r_d_fid / (r_d))
        # alpha_par = (Hz_fid) / (Hz_new) * (r_d_fid / (r_d))
        
        Om, Da, Hz, DN, f, sigma8, sigma8_0, sigma12, r_d = get_ParamsTaylor(np.array([ln10As, h, omega_cdm, omega_b])-birdmodel.valueref, paramsmod, 
                                                                             int(pardict['taylor_order']))
        
        Da_new = Da/h_convert*(1.0 + z_data)
        Hz_new = Hz*h_convert*conts.c/1000.0
        alpha_perp = (Da_new) * (1.0 / Da_fid) * (r_d_fid / (r_d))
        alpha_par = (Hz_fid) / (Hz_new) * (r_d_fid / (r_d))
        
        if i % 1000 == 0:
            print(i, alpha_par, alpha_par, f*sigma8, burntin[i, 4], burntin[i, 5])
        
        # chainvals.append(
        #     (
        #         ln10As,
        #         100.0 * h,
        #         omega_cdm,
        #         omega_b,
        #         alpha_perp,
        #         alpha_par,
        #         Om,
        #         2997.92458 * Da / h,
        #         100.0 * h * Hz,
        #         f,
        #         sigma8,
        #         sigma8_0,
        #         sigma12,
        #         loglike,
        #     )
        # )
        
        chainvals.append(
            (
                alpha_perp,
                alpha_par,
                f*sigma8,
                burntin[i, 4],
                burntin[i, 5],
            )
        )

    np.savetxt(newfile, np.array(chainvals))
