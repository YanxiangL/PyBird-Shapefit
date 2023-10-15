# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:33:27 2023

@author: s4479813
"""

import os
import sys
import numpy as np
from scipy import interpolate
from configobj import ConfigObj
from chainconsumer import ChainConsumer
import scipy.constants as conts
from classy import Class
import copy
from scipy.linalg import lapack, cholesky, block_diag
from nbodykit import cosmology
import scipy as sp

sys.path.append("../")
from tbird.Grid import run_camb, run_class
from fitting_codes.fitting_utils import format_pardict, BirdModel, FittingData
from tbird.computederivs import get_grids, get_ParamsTaylor

def read_chain_backend(chainfile):
    import copy
    import emcee
    
    #Read the MCMC chain
    reader = emcee.backends.HDFBackend(chainfile)
    
    #Find the autocorrelation time. 
    tau = reader.get_autocorr_time(tol = 0)
    #Using the autocorrelation time to figure out the burn-in. 
    burnin = int(2 * np.max(tau))
    #Retriving the chain and discard the burn-in, 
    samples = reader.get_chain(discard=burnin, flat=True)
    #The the log-posterior of the chain. 
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    #Find the best-fit parameters. 
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples

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

def slope_at_x(xvector,yvector):
    #find the slope
    diff = np.diff(yvector)/np.diff(xvector)
    diff = np.append(diff,diff[-1])
    return diff

if __name__ == "__main__":

    # Reads in a chain containing cosmological parameters output from a fitting routine, and
    # converts the points to alpha_perp, alpha_par, f(z)*sigma8(z) and f(z)*sigma12(z)

    # First, read in the config file used for the fit
    configfile = sys.argv[1]
    job_num = int(sys.argv[2])
    job_total = int(sys.argv[3])
    # fixedbias = bool(int(sys.argv[4]))
    Shapefit = bool(int(sys.argv[4]))
    # flatprior = bool(int(sys.argv[6]))
    
    if Shapefit == True:
        method = int(sys.argv[5])
        Approx_Gaussian = bool(int(sys.argv[6]))
    # try:
    #     redindex = int(sys.argv[7])
    #     print('Using redshift bin '+ str(redindex))
    #     onebin = True
    # except:
    #     print('Using all redshift bins')
    #     onebin = False
    
    
    
    pardict = ConfigObj(configfile)

    # Just converts strings in pardicts to numbers in int/float etc.
    pardict = format_pardict(pardict)
    
    
    nz = len(pardict['z_pk'])
    if nz > 1.5:
        onebin = False
        print('Using all redshift bins')
    else:
        onebin = True
        redindex = np.int32(pardict['red_index'])
        print('Using redshift bin '+ str(redindex))
    try:
        mock_num = int(sys.argv[7])
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
                cov = np.loadtxt(pardict['covfile'][i] + '_mean.txt')
                cov_all.append(cov)
                string += '_' + str(pardict['z_pk'][i])
            cov_new = block_diag(*cov_all)
            newfile = '../../data/cov' + string + '_mean.txt'
            np.savetxt(newfile, cov_new)
            pardict['covfile'] = newfile
            pardict['datafile'] = pardict['datafile'] + '_mean.txt'
        single_mock = False
        
    if single_mock == False:
        if Shapefit == False:
            keyword = '_bin_'+str(redindex) + '_mean'
        else: 
            keyword = '_mean'
    else:
        keyword = '_bin_'+str(redindex) + '_mock_' + str(mock_num)
         
    # keyword = keyword + '_noresum_'
    # if fixedbias == True:
    #     keyword = keyword + '_fixedbias'
    #     print('We are fixing b3 and ce1.')
        
    # if flatprior == False:
    #     keyword += '_Gaussian'
    # else:
    #     keyword += '_flat'
    
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
    
    if MinF == True:
        pardict['vary_c4'] = 0
    
    if int(pardict['vary_c4']) == 0 and MinF == False:
        keyword += '_anticorr'
        
    if Shapefit == True:
        if Approx_Gaussian == True:
            keyword += '_Approx'
        else:
            keyword += '_Interpolate'
        
    
    z_data = np.float64(pardict["z_pk"])[0]
    
    job_total_num = 81
    # grid_all = []
    # for i in range(job_total_num):
    #     grid = np.load("Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + ".npy")
    #     grid_all.append(grid)
    
    # grid_all = np.vstack(grid_all)
    # order = float(pardict["order"])
    # delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
    # valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    # truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
    # shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
    # grid_all = grid_all.reshape((*shapecrd[1:], np.shape(grid_all)[1]))
    # interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
    
    with_w0 = False
    with_w0_wa = False
    with_omegak = False
    grid_all = []
    # for i in range(job_total_num):
    #     if "w" in pardict.keys():
    #         filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_w0' + ".npy"
    #         with_w0 = True
    #         if "wa" in pardict.keys():
    #             filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_w0_wa' + ".npy"
    #             with_w0 = False
    #             with_w0_wa = True
    #     elif pardict['freepar'][-1] == 'Omega_k':
    #         filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + '_omegak' + ".npy"
    #         with_omegak = True
    #     else:
    #         filename = "Shapefit_Grid_" + str(i) + "_" + str(job_total_num) + ".npy"
    #     grid = np.load(filename)
    #     grid_all.append(grid)
    
    # grid_all = np.vstack(grid_all)
    # order = np.int32(pardict["order"])
    # delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
    # valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    # truecrd = [valueref[l] + delta[l] * np.arange(-order, order + 1) for l in range(len(pardict["freepar"]))]
    # shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
    # grid_all = grid_all.reshape((*shapecrd[1:], np.shape(grid_all)[1]))
    # if with_w0 == True or with_omegak == True:
    #     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all[:, :, :, :, :, :4])
    # elif with_w0_wa == True:
    #     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all[:, :, :, :, :, :, :4])
    # else:
    #     interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all[:, :, :, :, :4])
    
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
        if with_w0 == True:
            interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        elif with_w0_wa == True:
            interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        elif with_omegak == True:
            interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        else:
            interpolation_function = sp.interpolate.RegularGridInterpolator(truecrd, grid_all)
        # interpolation_function2 = sp.interpolate.RegularGridInterpolator(grid_all, truecrd)
        interpolation_functions.append(interpolation_function)
        
    
    if Shapefit == False:
        marg_str = "marg" if pardict["do_marg"] else "all"
        hex_str = "hex" if pardict["do_hex"] else "nohex"
        dat_str = "xi" if pardict["do_corr"] else "pk"
        fmt_str = (
            "%s_%s_%2dhex%2d_%s_%s_%s"+keyword
            if pardict["do_corr"]
            else "%s_%s_%3.2lfhex%3.2lf_%s_%s_%s_kmin0p02_fewerbias"+keyword
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
        oldfile = chainfile + ".hdf5"
        
    else:
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
        chainfile = chainfile + "_mock" + keyword
            
        if method == 1:
            NWM = '_EH98'
        elif method == 2:
            NWM = '_Hinton2017'
        elif method == 3:
            NWM = '_Wallisch2018'
        else:
            raise ValueError('Incorrect method for de-wiggle power spectrum. Enter 1 for EH98, 2 for Hinton2017 and 3 for Wallisch2018.')
        
        oldfile = chainfile + NWM + ".hdf5"
        print(oldfile)
        
    burntin, bestfit, like = read_chain_backend(oldfile)
    
    total = len(burntin)
    print(total)
    job_length = np.int32(total/job_total)
    start = job_num*job_length
    end = np.int32(job_num+1)*job_length
    print(start, end, job_num)
    newfile = chainfile + "_converted_bin_"+str(redindex) + '_' + str(job_num) +".dat"
    print(newfile)
    
    valueref = np.array([float(pardict[k]) for k in pardict["freepar"]])
    delta = np.fabs(np.array(pardict["dx"], dtype=np.float64))
    
    lower_bounds = valueref - pardict["order"] * delta
    upper_bounds = valueref + pardict["order"] * delta
    
    
    # #Compute the compressed parameters based on the template (true) cosmology. 
    # template = Class()
    
    # template.set({
    # "A_s": np.exp(float(pardict["ln10^{10}A_s"]))/1e10,
    # "n_s": float(pardict["n_s"]),
    # "H0": 100.0*float(pardict['h']),
    # "omega_b": float(pardict['omega_b']),
    # "omega_cdm": float(pardict['omega_cdm']),
    # "N_ur": float(pardict["N_ur"]),
    # "N_ncdm": int(pardict["N_ncdm"]),
    # "m_ncdm": float(pardict["m_ncdm"]),
    # "Omega_k": float(pardict["Omega_k"]),
    #  })
    
    # if "w" in pardict.keys():
    #     template.set(
    #         {"Omega_Lambda": 0.0,
    #          "w0_fld": float(pardict["w"])
    #             })
    #     if "wa" in pardict.keys():
    #         template.set(
    #             {"wa_fld": float(pardict["wa"])
    #             })
    
    # template.set({
    #     "output": "mPk",
    #     "P_k_max_1/Mpc": float(pardict["P_k_max_h/Mpc"]),
    #     "z_max_pk": z_data,
    # })
    
    # template.compute()
    
    # # #This is in h/Mpc. 
    # # kmpiv = 0.03
    
    # # kvec = np.logspace(-2.0, 0.0, 300)
    # # # kvec = fittingdata.data['x_data'][0][0]
    
    
    # # h_fid = template.h()
    # # H_z_fid = template.Hubble(z_data)*conts.c/1000.0
    # # r_d_fid = template.rs_drag()*h_fid
    # # DM_fid = template.angular_distance(z_data)*(1.0+z_data)
    
    # # cosmo_fid = cosmology.cosmology.Cosmology(h = float(pardict['h']), Omega0_b = float(pardict['omega_b'])/float(pardict['h'])**2, 
    # #                                       Omega0_cdm = float(pardict['omega_cdm'])/float(pardict['h'])**2, 
    # #                                       N_ur = float(pardict["N_ur"]), P_k_max = float(pardict["P_k_max_h/Mpc"]), n_s = float(pardict["n_s"]), 
    # #                                       A_s = np.exp(float(pardict["ln10^{10}A_s"]))/1e10, N_ncdm = int(pardict["N_ncdm"]), 
    # #                                       Omega_k = float(pardict["Omega_k"]), 
    # #                                       tau_reio = float(pardict["tau_reio"]), P_z_max = z_data)
    
    # # fAmp_fid = template.scale_independent_growth_factor_f(z_data)*np.sqrt(template.pk_lin(kmpiv*h_fid,z_data)*h_fid**3)
    
    # # transfer_fid = cosmology.power.transfers.NoWiggleEisensteinHu(cosmo_fid, z_data)(kvec)**2
    
    # # transfer_fid = EH98_transfer(kvec, z_data, 1.0, cosmo = template)
    
    # # f = M.scale_independent_growth_factor_f(z_data)
    # # sigma8 = M.sigma(8.0 / M.h(), z_data)
    
    # fsigma8_fid = template.scale_independent_growth_factor_f(z_data)*template.sigma(8.0/template.h(), z_data)
    
    fsigma8_fid = float(pardict['fsigma8'])
    
    print('Start converting')

    # Loop over the parameters in the chain and use the grids to compute the derived parameters
    chainvals = []
    # for i, (vals, loglike) in enumerate(zip(burntin, like)):
    #     ln10As, h, omega_cdm, omega_b = vals[:4]
    for i in range(start, end):
        if with_w0 == True:
            ln10As, h, omega_cdm, omega_b, w = burntin[i, :5]
        elif with_w0_wa == True:
            ln10As, h, omega_cdm, omega_b, w, wa = burntin[i, :6]
        elif with_omegak == True:
            ln10As, h, omega_cdm, omega_b, omega_k = burntin[i, :5]
        else:
            ln10As, h, omega_cdm, omega_b = burntin[i, :4]
            
        if with_w0 == True:
            if np.any(np.less([ln10As, h, omega_cdm, omega_b, w], lower_bounds)) or np.any(
                np.greater([ln10As, h, omega_cdm, omega_b, w], upper_bounds)
            ):
                continue
        elif with_w0_wa == True:
            if np.any(np.less([ln10As, h, omega_cdm, omega_b, w, wa], lower_bounds)) or np.any(
                np.greater([ln10As, h, omega_cdm, omega_b, w, wa], upper_bounds)
            ):
                continue
        if with_omegak == True:
            if np.any(np.less([ln10As, h, omega_cdm, omega_b, omega_k], lower_bounds)) or np.any(
                np.greater([ln10As, h, omega_cdm, omega_b, omega_k], upper_bounds)
            ):
                continue
        else:
            if np.any(np.less([ln10As, h, omega_cdm, omega_b], lower_bounds)) or np.any(
                np.greater([ln10As, h, omega_cdm, omega_b], upper_bounds)
            ):
                continue
        
        for j in range(nz):
            if with_w0 == True or with_omegak == True:
                # print(interpolation_functions[j](burntin[i, :5])[0][:4])
                theo_aperp, theo_apara, theo_fAmp, theo_mslope = interpolation_functions[j](burntin[i, :5])[0][:4]
            elif with_w0_wa == True:
                theo_aperp, theo_apara, theo_fAmp, theo_mslope = interpolation_functions[j](burntin[i, :6])[0][:4]
            else:
                theo_aperp, theo_apara, theo_fAmp, theo_mslope = interpolation_functions[j](burntin[i, :4])[0][:4]
        
            if i % 10000 == 0:
                # print(i, alpha_par, alpha_par, f*sigma8, burntin[i, 4], burntin[i, 5])
                if with_w0 == True:
                    print(i, theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope, ln10As, h, omega_cdm, omega_b, w)
                elif with_w0_wa == True:
                    print(i, theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope, ln10As, h, omega_cdm, omega_b, w, wa)
                elif with_omegak == True:
                    print(i, theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope, ln10As, h, omega_cdm, omega_b, omega_k)
                else:
                    print(i, theo_aperp, theo_apara, theo_fAmp*fsigma8_fid, theo_mslope, ln10As, h, omega_cdm, omega_b)
            # p
            
            
            if int(pardict['vary_c4']) == 1:
                chainvals.append(
                    (
                        # alpha_perp,
                        # alpha_par,
                        # f*sigma8,
                        # burntin[i, 4],
                        # burntin[i, 5],
                        theo_aperp,
                        theo_apara,
                        theo_fAmp*fsigma8_fid, 
                        theo_mslope,
                        burntin[i, -3],
                        burntin[i, -2],
                        burntin[i, -1]
                    )
                )
            else:
                chainvals.append(
                    (
                        # alpha_perp,
                        # alpha_par,
                        # f*sigma8,
                        # burntin[i, 4],
                        # burntin[i, 5],
                        theo_aperp,
                        theo_apara,
                        theo_fAmp*fsigma8_fid, 
                        theo_mslope,
                        burntin[i, -2],
                        burntin[i, -1]
                    )
                )

    np.savetxt(newfile, np.array(chainvals))
