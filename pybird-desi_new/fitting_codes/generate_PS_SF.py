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


configfile = '../config/tbird_KP4_LRG.ini'
pardict = ConfigObj(configfile)
pardict = format_pardict(pardict)
keyword = str("Shapefit_mock_mean")
pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
onebin = True
redindex = 0
Shapefit = True
one_nz = True

try:
    mock_num = int(sys.argv[5])
    print("Using one of the mocks.")
    datafiles = np.loadtxt(pardict['datafile'], dtype=str)
    mockfile = str(datafiles) + str(mock_num) + '.dat'
    newfile = '../config/data_mock_' + str(mock_num) + '.txt'
    text_file = open(newfile, "w")
    n = text_file.write(mockfile)
    text_file.close()
    pardict['datafile'] = newfile
    single_mock = True
except:
    print("Using the mean of mocks of the same redshift")
    single_mock = False
    pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
    pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"

# Set up the data
fittingdata = FittingData(pardict)

# n_sims = [978, 1000, 1000]
n_sims = [1000, 1000, 1000]

if onebin == False:
    hartlap = [(ns - fittingdata.data["ndata"][i] - 2.0) / (ns - 1.0) for i, ns in enumerate(n_sims)]
    cov_inv_new = copy.copy(fittingdata.data["cov_inv"])
    for i, (nd, ndcum) in enumerate(
        zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
    ):
        cov_inv_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= hartlap[i]
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
    
    hartlap = (n_sims[redindex] - fittingdata.data["ndata"][redindex] - 2.0) / (n_sims[redindex] - 1.0)
    print(hartlap)
    
    nparams = 7.0
    percival_A = 2.0/((n_sims[redindex] - fittingdata.data["ndata"][redindex]-1.0)*(n_sims[redindex] - fittingdata.data['ndata'][redindex]-4.0))
    percival_B = percival_A/2.0*(n_sims[redindex] - fittingdata.data['ndata'][redindex]-2.0)
    percival_m = (1.0+percival_B*(fittingdata.data['ndata'][redindex] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
    print(percival_m)
    
    cov_part = fittingdata.data['cov'][length_start:length_end, length_start:length_end]*percival_m
    fitdata_part = fittingdata.data['fit_data'][length_start:length_end]
    
    cov_lu, pivots, cov_part_inv, info = lapack.dgesv(cov_part, np.eye(len(cov_part)))
    
    cov_part_inv = cov_part_inv*hartlap
    
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
        
# fittingdata = FittingData(pardict)
xdata = [max(x, key=len) for x in fittingdata.data["x_data"]]


bestfit = np.array([1.00109181e+00, 9.91803004e-01, 4.54658474e-01, 1.08556290e-03,
       2.00486628e+00, 1.81208867e+00, 8.35103662e+00
])

birdmodel_all = []
for i in range(len(pardict["z_pk"])): 
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
        # print(shot_noise_ratio_prior)
    
    # birdmodel_i.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio])
    # birdmodel_i.eft_priors = np.array([2.0, 2.0, 10.0, 10.0, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio, 10.0*shot_noise_ratio])
    # birdmodel_i.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24*shot_noise_ratio_prior, 4.0, 20.0])
    # birdmodel_i.eft_priors = np.array([4.0, 10.0, 400.0, 400.0, 0.24*shot_noise_ratio_prior, 4.0, 20.0])
    # birdmodel_i.eft_priors = np.array([4.0, 10.0, 10.0, 10.0, 0.24*shot_noise_ratio_prior, 4.0, 10.0])
    # birdmodel_i.eft_priors = np.array([4.0, 10.0, 100.0, 100.0, 0.24*shot_noise_ratio_prior, 20.0, 20.0])
    # birdmodel_i.eft_priors = np.array([10.0, 200.0, 2000.0, 2000.0, 2.4*shot_noise_ratio_prior, 50.0, 20.0])
    # birdmodel_i.eft_priors = np.array([2.0, 20.0, 400.0, 400.0, 0.24*shot_noise_ratio_prior, 1.0*shot_noise_ratio_prior, 5.0*shot_noise_ratio_prior])
    birdmodel_i.eft_priors = np.array([2.0, 20.0, 100.0, 100.0, 0.24*shot_noise_ratio_prior, 2.0*shot_noise_ratio_prior, 4.0*shot_noise_ratio_prior])


    print(birdmodel_i.eft_priors)
    birdmodel_all.append(birdmodel_i)
    print(i)
    



if len(bestfit) != 14:
    
    shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
    
    alpha_perp, alpha_par, fsigma8, m = bestfit[:4]
    
    bias = bestfit[4:]
    
    # b1, c2, c4 = bias
    
    # b2 = (c2+c4)/np.sqrt(2.0)
    # b4 = (c2-c4)/np.sqrt(2.0)
    
    b1, bp, bd = bias
    
    b4 = bd
    b2 = (bp - bd)/0.86
    
    margb = 0.0
    bs = np.array(
        [
            bestfit[4],
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
    
    result = do_optimization(lambda *args: -find_bestfit_marg(*args), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    print(result["x"])
    
    # Pi_new = Pi[:, :, 0]
    
    # print(birdmodel_all[0].compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin)[0, :])
    
    # Covbi = np.dot(Pi_new, Pi_new.T) + block_diag(*1.0/model.eft_priors)
    # Cinvbi = np.linalg.inv(Covbi)
    # vectorbi = np.dot(P_model_interp.T, Pi_new.T) - np.dot(fittingdata.data['invcovdata'], Pi_new.T)
    
    # bg = -np.dot(Cinvbi, vectorbi.T)
    # print(bg[:, 0])
    
    # bs_analytic = result['x']
    
    bs_analytic = birdmodel_all[0].compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin)[0, :]
    # # np.save("bs.npy", bs_analytic)
    # print(bs_analytic)
    
    bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], bestfit[6], bs_analytic[1], 
                        bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
    print(bestfit)

# b2 = (bestfit[5] + bestfit[7]) / np.sqrt(2.0)
# b4 = (bestfit[5] - bestfit[7]) / np.sqrt(2.0)
# b2 = bestfit[5] - bestfit[7]
# b4 = bestfit[7]
b4 = bestfit[7]
b2 = (bestfit[5] - bestfit[7])/0.86

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

# np.save("SF_bestfit.npy", P_model_interp)
print(np.float64(pardict['xfit_max']), np.int16(pardict['do_marg']))

chi_squared = birdmodel_all[0].compute_chi2(P_model_interp, fittingdata.data)
print(chi_squared)