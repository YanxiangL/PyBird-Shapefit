# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:16:06 2023

@author: s4479813
"""

import numpy as np
import os
import sys
import copy
from configobj import ConfigObj
from scipy.linalg import lapack, block_diag

sys.path.append("../")
from pybird_dev import pybird
from tbird.Grid import grid_properties, run_camb, run_class
from fitting_codes.fitting_utils import format_pardict, FittingData

from fitting_codes.fitting_utils import (
    FittingData,
    BirdModel,
    create_plot,
    update_plot,
    format_pardict,
    do_optimization,
)

def find_bestfit_marg(params):
    b3, cct, cr1, cr2, ce1, cemono, cequad = params
    
    priors = 0.0
    
    # Gaussian prior for b3 of width 2 centred on 0
    priors += -0.5 * b3 ** 2/birdmodel_all[0].eft_priors[0]**2

    # Gaussian prior for cct of width 2 centred on 0
    priors += -0.5 * cct ** 2/birdmodel_all[0].eft_priors[1]**2

    # Gaussian prior for cr1 of width 4 centred on 0
    priors += -0.5 * cr1 ** 2 / birdmodel_all[0].eft_priors[2]**2

    # Gaussian prior for cr1 of width 4 centred on 0
    priors += -0.5 * cr2 ** 2 / birdmodel_all[0].eft_priors[3]**2

    # Gaussian prior for ce1 of width 2 centred on 0
    priors += -0.5 * ce1 ** 2 / birdmodel_all[0].eft_priors[4]**2

    # Gaussian prior for cemono of width 2 centred on 0
    priors += -0.5 * cemono ** 2 / birdmodel_all[0].eft_priors[5]**2

    # Gaussian prior for cequad of width 2 centred on 0
    priors += -0.5 * cequad ** 2/birdmodel_all[0].eft_priors[6]**2
    
    
    
    bs = np.array(
        [
            bestfit[4],
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
    
    P_model_lin, P_model_loop, P_model_interp = birdmodel_all[0].compute_model(
        bs.reshape((-1, 1)), Plin, Ploop, fittingdata.data["x_data"][0]
    )
    
    chi_squared = birdmodel_all[0].compute_chi2(P_model_interp, fittingdata.data)
    
    return -0.5*chi_squared[0] + priors


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


bestfit = np.array([ 0.98735055,  1.00273368,  0.37848122, -0.00906751,  2.12157187,
        1.49645345])
# bestfit = np.array([0.99709665,  1.00871029,  0.41892234, -0.01453929,  1.22791322,
#         0.41936005])
# bestfit = np.array([0.9973149 ,  1.00530617,  0.41853926, -0.01643777,  1.22659964,
#         0.42956858])

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
    



if len(bestfit) != 14:
    
    shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
    
    alpha_perp, alpha_par, fsigma8, m = bestfit[:4]
    
    bias = bestfit[4:]
    
    # b1, c2, c4 = bias
    
    # b2 = (c2+c4)/np.sqrt(2.0)
    # b4 = (c2-c4)/np.sqrt(2.0)
    
    # if int(pardict['vary_c4']) == 1:
    #     b1, bp, bd = bias
        
    #     b4 = bd
    #     b2 = (bp - bd)/0.86
        
    # else:
    #     b1, c2 = bias
    #     b2 = c2/np.sqrt(2.0)
    #     b4 = b2
    
    # margb = 0.0
    # bs = np.array(
    #     [
    #         bestfit[4],
    #         b2,
    #         margb,
    #         b4,
    #         margb,
    #         margb,
    #         margb,
    #         margb,
    #         margb,
    #         margb,
    #     ]
    # )
    
    margb = 0.0
    
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
    
    bs = bs.reshape((-1, 1))
    
    Plin_i, Ploop_i = birdmodel_all[0].modify_template([alpha_perp, alpha_par, fsigma8], fittingdata=fittingdata, factor_m=m, 
                                                       redindex=redindex, one_nz = one_nz, resum = True)
    Plin = np.array([Plin_i])
    Ploop = np.array([Ploop_i])
    
    Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
    Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])
    
    
    P_model_lin, P_model_loop, P_model_interp = birdmodel_all[0].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][0])
    Pi = birdmodel_all[0].get_Pi_for_marg(Ploop, bs[0], shot_noise_ratio, fittingdata.data["x_data"][0])
    
    # print(birdmodels[0].get_Pi_for_marg(
    #     Ploop, bs.reshape((-1, 1)), shot_noise_ratio, fittingdata.data["x_data"][0]
    # ))
    
    chi_squared = birdmodel_all[0].compute_chi2_marginalised(P_model_interp, Pi, fittingdata.data, onebin = onebin)
    print(chi_squared)
    
    # result = do_optimization(lambda *args: -find_bestfit_marg(*args), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # print(result["x"])
    
    # Pi_new = Pi[:, :, 0]
    
    # print(birdmodel_all[0].compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin)[0, :])
    
    # Covbi = np.dot(Pi_new, Pi_new.T) + block_diag(*1.0/model.eft_priors)
    # Cinvbi = np.linalg.inv(Covbi)
    # vectorbi = np.dot(P_model_interp.T, Pi_new.T) - np.dot(fittingdata.data['invcovdata'], Pi_new.T)
    
    # bg = -np.dot(Cinvbi, vectorbi.T)
    # print(bg[:, 0])
    
    # bs_analytic = result['x']
    
    bs_analytic = birdmodel_all[0].compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin, MinF = MinF)[0, :]
    # # np.save("bs.npy", bs_analytic)
    # print(bs_analytic)
    
    if MinF == False:
        if int(pardict['do_hex']) == 0:
            if pardict['prior'] == 'BOSS_MaxF':
                bs_analytic = np.array([bs_analytic[0], bs_analytic[1], bs_analytic[2], 0.0, bs_analytic[3], bs_analytic[4], bs_analytic[5]])
            else:
                bs_analytic = np.array([bs_analytic[0], bs_analytic[1], bs_analytic[2], 0.0, bs_analytic[3], 0.0, bs_analytic[4]])
    else:
        if int(pardict['do_hex']) == 0:
            bs_analytic = np.array([0.0, bs_analytic[0], bs_analytic[1], 0.0, bs_analytic[2], 0.0, bs_analytic[3]])
            
        else:
            bs_analytic = np.array([0.0, bs_analytic[0], bs_analytic[1], bs_analytic[2], bs_analytic[3], 0.0, bs_analytic[4]])
    
    if int(pardict['vary_c4']) == 1:
        bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], bestfit[6], bs_analytic[1], 
                            bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
    else:
        bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], 0.0, bs_analytic[1], 
                            bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
    
    # bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], bestfit[6], bs_analytic[1], 
    #                         bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
    print(bestfit)

# b2 = (bestfit[5] + bestfit[7]) / np.sqrt(2.0)
# b4 = (bestfit[5] - bestfit[7]) / np.sqrt(2.0)
# b2 = bestfit[5] - bestfit[7]
# b4 = bestfit[7]

pardict['xfit_min'] = [0.0, 0.0, 0.0]
pardict['xfit_max'] = [0.80, 0.80, 0.80]

fittingdata = FittingData(pardict)

# if int(pardict['vary_c4']) == 1:
#     b4 = bestfit[7]
#     b2 = (bestfit[5] - bestfit[7])/0.86
# else:
#     b2 = (bestfit[5] + bestfit[7])/np.sqrt(2.0)
#     b2 = (bestfit[5] - bestfit[7])/np.sqrt(2.0)

# bs = np.array(
#     [
#         bestfit[4],
#         b2,
#         bestfit[6],
#         b4,
#         bestfit[8],
#         bestfit[9],
#         bestfit[10],
#         # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
#         # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
#         # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
#         bestfit[11] * shot_noise_ratio,
#         bestfit[12] * shot_noise_ratio,
#         bestfit[13] * shot_noise_ratio,
#     ]
# )

if int(pardict['vary_c4']) == 0:
    b4 = bestfit[5]/np.sqrt(2.0)
    b2 = bestfit[5]/np.sqrt(2.0)
else:
    b2 = (bestfit[5] + bestfit[7])/np.sqrt(2.0)
    b4 = (bestfit[5] - bestfit[7])/np.sqrt(2.0)

bs = np.array(
    [
        bestfit[4],
        b2,
        bestfit[6],
        b4,
        bestfit[8],
        bestfit[9],
        bestfit[10],
        # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
        # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
        # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
        bestfit[11] * shot_noise_ratio,
        bestfit[12] * shot_noise_ratio,
        bestfit[13] * shot_noise_ratio,
    ]
)

bs = bs.reshape((-1, 1))
Plin_i, Ploop_i = birdmodel_all[0].modify_template([alpha_perp, alpha_par, fsigma8], fittingdata=fittingdata, factor_m=m, 
                                                   redindex=redindex, one_nz = one_nz, resum = True)
Plin = np.array([Plin_i])
Ploop = np.array([Ploop_i])

Plin = np.transpose(Plin, axes=[1, 2, 3, 0])
Ploop = np.transpose(Ploop, axes=[1, 3, 2, 0])

# np.save("Plin_MB.npy", Plin)
# np.save("Ploop_MB.npy", Ploop)

P_model_lin, P_model_loop, P_model_interp = birdmodel_all[0].compute_model(bs, Plin, Ploop, fittingdata.data["x_data"][0])

np.save("SF_pk_QSO_0p20.npy", [fittingdata.data["x_data"][0][0], P_model_interp])
# np.save("FS_pk_fiducial_ELG.npy", [fittingdata.data["x_data"][0][0], P_model_interp])
# np.save("SF_bestfit.npy", P_model_interp)
# print(np.float64(pardict['xfit_max']), np.int16(pardict['do_marg']))

chi_squared = birdmodel_all[0].compute_chi2(P_model_interp, fittingdata.data)
print(chi_squared)