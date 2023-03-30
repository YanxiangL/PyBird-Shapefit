# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:39:14 2023

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
    
    P_model_lin, P_model_loop, P_model_interp = birdmodels[0].compute_model(
        bs.reshape((-1, 1)), Plin, Ploop, fittingdata.data["x_data"][0]
    )
    
    chi_squared = birdmodels[0].compute_chi2(P_model_interp, fittingdata.data)
    
    return -0.5*chi_squared[0]


# configfile = '../config/tbird_KP4_LRG.ini'
# pardict = ConfigObj(configfile)
# pardict = format_pardict(pardict)
# keyword = str("Shapefit_mock_mean")
# pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
# pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
# onebin = True
# redindex = 0

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

# # Set up the data
# fittingdata = FittingData(pardict)

# # n_sims = [978, 1000, 1000]
# n_sims = [1000, 1000, 1000]

# if onebin == False:
#     hartlap = [(ns - fittingdata.data["ndata"][i] - 2.0) / (ns - 1.0) for i, ns in enumerate(n_sims)]
#     cov_inv_new = copy.copy(fittingdata.data["cov_inv"])
#     for i, (nd, ndcum) in enumerate(
#         zip(fittingdata.data["ndata"], np.cumsum(fittingdata.data["ndata"]) - fittingdata.data["ndata"][0])
#     ):
#         cov_inv_new[ndcum : ndcum + nd, ndcum : ndcum + nd] *= hartlap[i]
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
    
#     hartlap = (n_sims[redindex] - fittingdata.data["ndata"][redindex] - 2.0) / (n_sims[redindex] - 1.0)
#     print(hartlap)
    
#     nparams = 7.0
#     percival_A = 2.0/((n_sims[redindex] - fittingdata.data["ndata"][redindex]-1.0)*(n_sims[redindex] - fittingdata.data['ndata'][redindex]-4.0))
#     percival_B = percival_A/2.0*(n_sims[redindex] - fittingdata.data['ndata'][redindex]-2.0)
#     percival_m = (1.0+percival_B*(fittingdata.data['ndata'][redindex] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
#     print(percival_m)
    
#     cov_part = fittingdata.data['cov'][length_start:length_end, length_start:length_end]*percival_m
#     fitdata_part = fittingdata.data['fit_data'][length_start:length_end]
    
#     cov_lu, pivots, cov_part_inv, info = lapack.dgesv(cov_part, np.eye(len(cov_part)))
    
#     cov_part_inv = cov_part_inv*hartlap
    
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
        
# # fittingdata = FittingData(pardict)
# xdata = [max(x, key=len) for x in fittingdata.data["x_data"]]

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

# Nl = 3

# kin, Pin, Om, Da_fid, Hz_fid, DN_fid, fN_fid, sigma8_fid, sigma8_0_fid, sigma12, r_d = run_class(pardict)

# correlator = pybird.Correlator()
# correlatorcf = pybird.Correlator()

# correlator.set(
#     {
#         "output": "bPk",
#         "multipole": Nl,
#         "skycut": len(pardict["z_pk"]),
#         "z": pardict["z_pk"],
#         "optiresum": False,
#         "with_bias": False,
#         "kmax": 0.35,
#         "xdata": xdata,
#         "with_AP": True,
#         "DA_AP": Da_fid,
#         "H_AP": Hz_fid,
#         "with_fibercol": False,
#         "with_window": False,
#         "with_stoch": True,
#         "with_resum": True,
#         "with_binning": True,
#     }
# )


# bestfit = [ 3.00294686e+00,  6.83625653e-01,  1.27453552e-01,  2.24029667e-02,
#   2.03616865e+00, -2.68598351e-01, -1.64552423e+00, -2.53827825e+01,
#   -3.21231087e+01,  3.10916139e+02, -3.73389539e+02,  6.78887388e-01,
#   6.29634020e+00, -4.08443338e+00]

# bestfit = [3.0422, 0.6771, 0.1226, 0.0223, 1.9938, 0.4090, -11.3269]

bestfit = np.array([3.00633643,  0.67674525,  0.12302006,  0.02230698,  2.05235021,
        2.00107713, 10.68159108])

# parameters = copy.deepcopy(pardict)
# for k, var in enumerate(pardict["freepar"]):
#     parameters[var] = bestfit[k]
    
# kin, Pin, Om, Da, Hz, DN, fN, sigma8, sigma8_0, sigma12, r_d = run_class(parameters)

# first = len(pardict["z_pk"]) // 2
# correlator.compute({"k11": kin, "P11": Pin[first], "Omega0_m": Om, "D": DN[0], "f": fN[0], "DA": Da[0], "H": Hz[0]})

# Plin, Ploop = correlator.bird.formatTaylorPs(kdata=xdata[0])

# Plin = np.swapaxes(np.reshape(Plin, (correlator.co.Nl, Plin.shape[-2]//correlator.co.Nl, Plin.shape[-1]))[:, :, 1:], axis1=1, axis2=2)

# Ploop = np.swapaxes(np.reshape(Ploop, (correlator.co.Nl, Ploop.shape[-2]//correlator.co.Nl, Ploop.shape[-1]))[:, :, 1:], axis1=1, axis2=2)

# Plin = Plin.reshape((3, 3, 36, 1))
# Ploop = Ploop.reshape((3, 36, 21, 1))

# print(np.shape(Plin), np.shape(Ploop))

# np.save("Plin_best.npy", Plin)
# np.save("Ploop_best.npy", Ploop)

birdmodels = []
for i in range(1):
    #This is the default shot-noise in the pybird.py 
    shot_noise_fid = (1.0/3e-4)
    print('redshift bin ' + str(i))
    shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
        
    model = BirdModel(pardict, redindex=i)
    # model.eft_priors = np.array([0.002, 0.002, 0.002, 2.0, 0.2, 0.0002, 0.0002])
    # model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio, 2.0*shot_noise_ratio])
    # model.eft_priors = np.array([2.0, 2.0, 10.0, 10.0, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio, 10.0/shot_noise_ratio])
    # model.eft_priors = np.array([2.0, 8.0, 10.0, 10.0, 0.012/shot_noise_ratio, 10.0, 10.0])
    # model.eft_priors = np.array([10.0, 500.0, 5000.0, 5000.0, 40.0/shot_noise_ratio, 40.0, 100.0])
    # model.eft_priors = np.array([4.0, 40.0, 400.0, 400.0, 0.24/shot_noise_ratio, 4.0, 20.0])
    # model.eft_priors = np.array([4.0, 10.0, 10.0, 10.0, 0.24/shot_noise_ratio, 4.0, 10.0])
    # model.eft_priors = np.array([4.0, 40.0, 400.0, 400.0, 2.4/shot_noise_ratio, 20.0, 20.0])
    # model.eft_priors = np.array([4.0, 10.0, 400.0, 400.0, 0.24/shot_noise_ratio, 4.0, 20.0])
    model.eft_priors = np.array([2.0, 20.0, 100.0, 100.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 4.0/shot_noise_ratio])
    print(model.eft_priors)
    birdmodels.append(model)
    

eft_priors_all = birdmodels[0].eft_priors
    
shot_noise_fid = (1.0/3e-4)

shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid



if len(bestfit) != 14:
    
    bias = bestfit[4:]
    
    # b1, c2, c4 = bias
    
    # b2 = (c2+c4)/np.sqrt(2.0)
    # b4 = (c2-c4)/np.sqrt(2.0)
    
    b1, bp, bd = bias
    
    b4 = bd
    b2 = (bp - bd)/0.86
    
    # b2 = (bestfit[5] + bestfit[6]) / np.sqrt(2.0)
    # b4 = (bestfit[5] - bestfit[6]) / np.sqrt(2.0)
    
    print(b1, b2, b4)
    
    margb = 0.0
    bs = np.array(
        [
            b1,
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
    
    Plin, Ploop = birdmodels[0].compute_pk(np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3]]).reshape(-1, 1))
    # print(np.shape(Plin), np.shape(Ploop))
    P_model_lin, P_model_loop, P_model_interp = birdmodels[0].compute_model(
        bs.reshape((-1, 1)), Plin, Ploop, fittingdata.data["x_data"][0]
    )
    
    Pi = birdmodels[0].get_Pi_for_marg(
        Ploop, bs.reshape((-1, 1))[0], shot_noise_ratio, fittingdata.data["x_data"][0]
    )
    
    # print(birdmodels[0].get_Pi_for_marg(
    #     Ploop, bs.reshape((-1, 1)), shot_noise_ratio, fittingdata.data["x_data"][0]
    # ))
    
    chi_squared = birdmodels[0].compute_chi2_marginalised(np.concatenate([P_model_interp]), Pi, fittingdata.data, onebin=onebin, eft_priors=eft_priors_all)
    print(chi_squared)
    
    # result = do_optimization(lambda *args: -find_bestfit_marg(*args), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # print(result["x"])
    
    # Pi_new = Pi[:, :, 0]
    
    # Covbi = np.dot(Pi_new, Pi_new.T) + block_diag(*1.0/model.eft_priors)
    # Cinvbi = np.linalg.inv(Covbi)
    # vectorbi = np.dot(P_model_interp.T, Pi_new.T) - np.dot(fittingdata.data['invcovdata'], Pi_new.T)
    
    # bg = -np.dot(Cinvbi, vectorbi.T)
    # print(bg[:, 0])
    
    # bs_analytic = result['x']
    
    bs_analytic = birdmodels[0].compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin, eft_priors= eft_priors_all)[0, :]
    # # np.save("bs.npy", bs_analytic)
    # print(bs_analytic)
    
    bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], bestfit[6], bs_analytic[1], 
                        bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
    print(bestfit)

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
Plin, Ploop = birdmodels[0].compute_pk(np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3]]).reshape(-1, 1))

# np.save("Plin_MB.npy", Plin)
# np.save("Ploop_MB.npy", Ploop)

P_model_lin, P_model_loop, P_model_interp = birdmodels[0].compute_model(
    bs.reshape((-1, 1)), Plin, Ploop, fittingdata.data["x_data"][0]
)

# np.save("FS_bestfit.npy", P_model_interp)

chi_squared = birdmodels[0].compute_chi2(P_model_interp, fittingdata.data)
print(chi_squared)