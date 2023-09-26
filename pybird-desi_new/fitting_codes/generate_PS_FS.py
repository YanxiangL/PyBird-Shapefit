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

def percival_factor(redindex):
    if int(pardict['do_marg']) == 1:
        nparams = len(pardict['freepar']) + 3 
    else:
        nparams = len(pardict['freepar']) + 10
        
    # print(nparams)
        
    if onebin == False:
        percival_A = 2.0/((n_sims[redindex] - fittingdata.data["ndata"][0]-1.0)*(n_sims[redindex] - fittingdata.data['ndata'][0]-4.0))
        percival_B = percival_A/2.0*(n_sims[redindex] - fittingdata.data['ndata'][0]-2.0)
        percival_m = (1.0+percival_B*(fittingdata.data['ndata'][0] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
    else:
        percival_A = 2.0/((n_sims - fittingdata.data["ndata"][0]-1.0)*(n_sims - fittingdata.data['ndata'][0]-4.0))
        percival_B = percival_A/2.0*(n_sims - fittingdata.data['ndata'][0]-2.0)
        percival_m = (1.0+percival_B*(fittingdata.data['ndata'][0] - nparams))/(1.0+percival_A + percival_B*(nparams+1.0))
    return percival_m

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

# configfile = '../config/tbird_KP4_LRG.ini'
# pardict = ConfigObj(configfile)
# pardict = format_pardict(pardict)
# keyword = str("Shapefit_mock_mean")
# pardict['datafile'] = '../config/datafiles_KP4_LRG_mean.txt'
# pardict['covfile'] = "../../data/Cov_dk0.005/cov_mean.txt"
# onebin = True
# redindex = 0
# Shapefit = True
# one_nz = True

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

configfile = sys.argv[1]
plot_flag = int(sys.argv[2])
# try:
#     redindex = int(sys.argv[4])
#     print('Using redshift bin '+ str(redindex))
#     onebin = True
# except:
#     print('Using all redshift bins')
#     onebin = False

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

nz = len(pardict['z_pk'])
if nz > 1.5:
    onebin = False
    print('Using all redshift bins')
else:
    onebin = True
    redindex = np.int32(pardict['red_index'])
    print('Using redshift bin '+ str(redindex))
try:
    mock_num = int(sys.argv[5])
    if nz == 1:
        datafiles = np.loadtxt(pardict['datafile'] + '.txt', dtype=str)
        mockfile = str(datafiles) + str(mock_num) + '.dat'
        newfile = '../config/data_mock_' + str(mock_num) + '.txt'
        text_file = open(newfile, "w")
        n = text_file.write(mockfile)
        text_file.close()
        pardict['datafile'] = newfile
        # pardict['covfile'] = pardict['covfile'] + '.txt'
        if pardict['constrain'] == 'Single':
            pardict['covfile'] = pardict['covfile'] + '.txt'
        elif pardict['constrain'] == 'Mean':
            pardict['covfile'] = pardict['covfile'] + '_mean.txt'
        else:
            raise ValueError('Enter either "Single" or "Mean" to use the normal or reduced covariance matrix. ')
        single_mock = True
    else: 
        raise ValueError('Cannot fit multiple redshift for one single mock at the moment.')
except:
    if nz == 1:
        pardict['datafile'] = pardict['datafile'] + '_mean.txt'
        # pardict['covfile'] = pardict['covfile'] + '_mean.txt'
        # pardict['covfile'] = pardict['covfile'] + '.txt'
        
        if pardict['constrain'] == 'Single':
            pardict['covfile'] = pardict['covfile'] + '.txt'
        elif pardict['constrain'] == 'Mean':
            pardict['covfile'] = pardict['covfile'] + '_mean.txt'
        else:
            raise ValueError('Enter either "Single" or "Mean" to use the normal or reduced covariance matrix. ')
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

# Set up the data
fittingdata = FittingData(pardict)
        
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

#LRG mock mean Gaussian
# bestfit = [3.0548781106577394, 0.6758093169448686, 0.12023366922404688 , 0.022308800199294716, 1.9793469444861695,
#        1.691363454233598, 6.566199216204451]

#LRG mock mean flat
# bestfit = [3.008533836794631, 0.6816588361174459, 0.12555636292101166, 0.02236057005075636, 
#            2.030432786561079, 2.141468614681582, 17.11241690446963]

#LRG DE wide
# bestfit = [2.99886825,  0.68977488,  0.12146323,  0.0223228 , -1.05688687,
#         2.0358308 ,  1.77087725,  8.31371555]

#LRG DE narrow
# bestfit = [ 3.04104421,  0.68055944,  0.1182929 ,  0.02226876, -1.03753666,
#         1.98239148,  1.27168939, -1.85843498]

# LRG Mark Gaussian
# bestfit = [3.067925270865796, 0.6719907921075584, 0.1184284919241066, 0.02224803858919172, 1.9580310270266494,
#            1.3379855384698862, -1.6928771167948806]

#LRG Mark flat
# bestfit = [3.08367644864868, 0.6721340355384011, 0.11736423459275985, 0.02231513620233875, 1.9393855942247895, 
#            1.2590778208448699, -3.6415450899945014]

#Fiducial flat prior
# bestfit = [3.0364, 0.6736, 0.1200, 0.02237, 2.06103964e+00,  2.39747306e+00, -7.49324561e+00,  1.13361194e+01,
#         -5.24585670e+00, -1.58389772e+01, -1.68687548e-09, -2.26496615e+00,
#         1.38384981e+01, -5.17223154e-09]

# bestfit = [3.0364, 0.6736, 0.1200, 0.02237, 1.98866159,   0.94786764,   3.16135093,  -5.38572144,
#          2.11842905,  -0.23023721, 0.0,   0.67113864,  -7.33575874,
#        -10.6802009]

#Fiducial flat prior kmax = 0.25
# bestfit = np.array([3.0364, 0.6736, 0.12, 0.02237, 2.04973726,   2.25430457,  -3.13967641,   6.41162702,
#     -1.34190849, -13.12405747, 0.0, -1.89736831,   4.05387642,
#      -1.63381558])

# bestfit = [3.10245501, 0.67165473, 0.11529609, 0.02229233, 1.91738637,
#        1.19557049, 2.63990362]
# bestfit = [ 3.01245146e+00,  6.77394731e-01,  1.19686894e-01,  2.21357835e-02,
#         2.09043032e+00,  2.52056425e+00, -8.30423069e+00,  1.34075740e+01,
#        -6.01932976e+00, -1.68115562e+01,  2.35823442e-06, -2.39499744e+00,
#         1.61882552e+01, -4.98798051e-08]

# bestfit = [3.03282544, 0.69681266, 0.11952176, 0.02254068, 1.22348942,
#         1.41181921]

# bestfit = [3.0702652686897376,
#   0.6744527776052058,
#   0.11843977436751986,
#   0.022361107351628877, 1.22938409,
#           0.47674245]

# bestfit = [3.09707434, 0.6736643 , 0.11702418, 0.02242518, 1.17252148,
#         0.30752356]

# bestfit = [3.07131686, 0.68386496, 0.12056747, 0.02255762, 1.19071212,
#        1.56628003]

#ELG bestfit to kmax = 0.25 h/Mpc
# bestfit = [3.0364, 0.6736, 0.12, 0.02237,  1.24017956,  0.79312347, -4.38116497]

#QSO bestfit to kmax = 0.25 h/Mpc
bestfit = [3.0364, 0.6736, 0.12, 0.02237,  2.10754623,  3.86825614, -3.88109069]

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

if pardict['prior'] == 'MinF':
    MinF = True
elif pardict['prior'] == 'MaxF':
    MinF = False
elif pardict['prior'] == 'BOSS_MaxF':
    MinF = False
elif pardict['prior'] == 'BOSS_MinF':
    MinF = True
else:
    raise ValueError('Enter either "MinF", "MaxF", "BOSS_MaxF", "BOSS_MinF" to determine to prior for the marginalized parameters. ')

birdmodels = []
for i in range(nz):
    #This is the default shot-noise in the pybird.py 
    shot_noise_fid = (1.0/3e-4)
    
    if onebin ==True:
        i = redindex
        shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid
    else:
        shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])[i]/shot_noise_fid
    
    print('redshift bin ' + str(i))
        
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
    # model.eft_priors = np.array([2.0, 20.0, 100.0, 100.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 4.0/shot_noise_ratio])
    # if flatprior == False:
    #     if fixedbias == False:
    #         # model.eft_priors = np.array([2.0, 20.0, 100.0, 100.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 4.0/shot_noise_ratio])
    #         # model.eft_priors = np.array([2.0, 20.0, 10.0, 10.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 4.0/shot_noise_ratio])
    #         model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])
            
    #     else:
    #         model.eft_priors = np.array([1e-10, 20.0, 100.0, 100.0, 1e-10/shot_noise_ratio, 2.0/shot_noise_ratio, 4.0/shot_noise_ratio])
    #     print(model.eft_priors)
    # else:
    #     print('Flat prior')
    if pardict['prior'] == 'BOSS_MaxF':
        if int(pardict['do_hex']) == 1:
            model.eft_priors = np.array([2.0, 2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])
        else:
            model.eft_priors = np.array([2.0, 2.0, 8.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio, 2.0/shot_noise_ratio])

        # model.eft_priors = np.array([1e-10, 2.0, 4.0, 1e-10, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
        
    elif pardict['prior'] == 'BOSS_MinF':
        if int(pardict['do_hex']) == 1:
            # model.eft_priors = np.array([1e-10, 2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
            model.eft_priors = np.array([2.0, 4.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio])
        else:
        # model.eft_priors = np.array([2.0, 4.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio])
            # model.eft_priors = np.array([1e-10, 2.0, 4.0, 1e-10, 0.24/shot_noise_ratio, 1e-10, 2.0/shot_noise_ratio])
            model.eft_priors = np.array([2.0, 8.0, 0.24/shot_noise_ratio, 2.0/shot_noise_ratio])
    else:
        print('Flat prior')
    birdmodels.append(model)
    
# if flatprior == False:
#     eft_priors_all = birdmodels[0].eft_priors
# else:
#     eft_priors_all = None

if pardict['prior'] == 'BOSS_MaxF' or pardict['prior'] == 'BOSS_MinF':
    if onebin == False:
        eft_priors_all = np.hstack([birdmodels[i].eft_priors for i in range(nz)])
        # print(eft_priors_all)
    else:
        eft_priors_all = birdmodels[0].eft_priors
else:
    eft_priors_all = None
    
if MinF == True:
    pardict['vary_c4'] = 0
    
shot_noise_fid = (1.0/3e-4)

shot_noise_ratio = np.float64(fittingdata.data["shot_noise"])/shot_noise_fid



if len(bestfit) != 14:
    if "w" in pardict.keys():
        bias = bestfit[5:]
    else:
        bias = bestfit[4:]
    
    # b1, c2, c4 = bias
    
    # b2 = (c2+c4)/np.sqrt(2.0)
    # b4 = (c2-c4)/np.sqrt(2.0)
    
    if int(pardict['vary_c4']) == 1:
        # b1, bp, bd = bias
        
        # b4 = bd
        # b2 = (bp - bd)/0.86
        
        b1, c2, c4 = bias
        
        b2 = (c2+c4)/np.sqrt(2.0)
        b4 = (c2-c4)/np.sqrt(2.0)
        
    else:
        b1, c2 = bias
        b2 = c2/np.sqrt(2.0)
        b4 = b2
    
    # b2 = (bestfit[5] + bestfit[6]) / np.sqrt(2.0)
    # b4 = (bestfit[5] - bestfit[6]) / np.sqrt(2.0)
    
    print(b1, b2, b4, shot_noise_ratio)
    
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
    
    # Om_0 = (bestfit[2] + bestfit[3])/bestfit[1]**2
    # redshift = np.float64(pardict['z_pk'])
    # Om = Om_0*(1+redshift)**3/(Om_0*(1+redshift)**3 + (1-Om_0))
    # growth = Om**(6.0/11.0)
    
    if "w" in pardict.keys():
        Plin, Ploop = birdmodels[0].compute_pk(np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4]]).reshape(-1, 1))
    else:
        Plin, Ploop = birdmodels[0].compute_pk(np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3]]).reshape(-1, 1))
    # print(np.shape(Plin), np.shape(Ploop))
    
    # print(np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3]]).reshape(-1, 1))
    P_model_lin, P_model_loop, P_model_interp = birdmodels[0].compute_model(
        bs.reshape((-1, 1)), Plin, Ploop, fittingdata.data["x_data"][0]
    )
    
    Pi = birdmodels[0].get_Pi_for_marg(
        Ploop, bs.reshape((-1, 1))[0], shot_noise_ratio, fittingdata.data["x_data"][0], MinF=MinF
    )
    
    
    chi_squared = birdmodels[0].compute_chi2_marginalised(np.concatenate([P_model_interp]), Pi, fittingdata.data, onebin=onebin, eft_priors=eft_priors_all)
    print(chi_squared)
    
    # P_model_lin, P_model_loop, P_model_interp = birdmodels[i].compute_model(
    #     bs, Plin, Ploop, fittingdata.data["x_data"][i]
    # )
    # Pi = birdmodels[i].get_Pi_for_marg(
    #     Ploop, bs[0], float(fittingdata.data["shot_noise"][i]), fittingdata.data["x_data"][i]
    # )
    

    # result = do_optimization(lambda *args: -find_bestfit_marg(*args), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # print(result["x"])
    
    # Pi_new = Pi[:, :, 0]
    
    # Covbi = np.dot(Pi_new, Pi_new.T) + block_diag(*1.0/model.eft_priors)
    # Cinvbi = np.linalg.inv(Covbi)
    # vectorbi = np.dot(P_model_interp.T, Pi_new.T) - np.dot(fittingdata.data['invcovdata'], Pi_new.T)
    
    # bg = -np.dot(Cinvbi, vectorbi.T)
    # print(bg[:, 0])
    
    # bs_analytic = result['x']
    
    bs_analytic = birdmodels[0].compute_bestfit_analytic(P_model_interp, Pi, fittingdata.data, onebin=onebin, eft_priors= eft_priors_all, MinF=MinF)[0, :]
    # # np.save("bs.npy", bs_analytic)
    print(bs_analytic)
    
    if MinF == False:
        if int(pardict['do_hex']) == 0:
            bs_analytic = np.array([bs_analytic[0], bs_analytic[1], bs_analytic[2], 0.0, bs_analytic[3], 0.0, bs_analytic[4]])
    else:
        if int(pardict['do_hex']) == 0:
            bs_analytic = np.array([0.0, bs_analytic[0], bs_analytic[1], 0.0, bs_analytic[2], 0.0, bs_analytic[3]])
            
        else:
            bs_analytic = np.array([0.0, bs_analytic[0], bs_analytic[1], bs_analytic[2], bs_analytic[3], 0.0, bs_analytic[4]])
    
    pardict['xfit_max'] = [0.30, 0.30, 0.30]
    print(pardict['xfit_max'])
    fittingdata = FittingData(pardict)
    
    if "w" in pardict.keys():
        if int(pardict['vary_c4']) == 1:
            bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bestfit[6], bs_analytic[0], bestfit[7], bs_analytic[1], 
                            bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
        else:
            bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bestfit[6], bs_analytic[0], 0.0, bs_analytic[1], 
                            bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
        if int(pardict['vary_c4']) == 0:
            # b4 = bestfit[8]
            # b2 = (bestfit[6] - bestfit[8])/0.86
            b2 = bestfit[6]/np.sqrt(2.0)
            b4 = bestfit[6]/np.sqrt(2.0)
        else:
            b2 = (bestfit[6] + bestfit[8])/np.sqrt(2.0)
            b4 = (bestfit[6] - bestfit[8])/np.sqrt(2.0)
        
        bs = np.array(
            [
                bestfit[5],
                b2,
                bestfit[7],
                b4,
                bestfit[9],
                bestfit[10],
                bestfit[11],
                # params[counter + 7] * float(fittingdata.data["shot_noise"][i]),
                # params[counter + 8] * float(fittingdata.data["shot_noise"][i]),
                # params[counter + 9] * float(fittingdata.data["shot_noise"][i]),
                bestfit[12] * shot_noise_ratio,
                bestfit[13] * shot_noise_ratio,
                bestfit[14] * shot_noise_ratio,
            ]
        )
    else:
        try:
            if int(pardict['vary_c4']) == 1:
                bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], bestfit[6], bs_analytic[1], 
                                bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
            else:
                bestfit = np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], 0.0, bs_analytic[1], 
                                bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5], bs_analytic[6]])
        except:
            bestfit = [bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4], bestfit[5], bs_analytic[0], bestfit[6], bs_analytic[1], 
                            bs_analytic[2], bs_analytic[3], bs_analytic[4], bs_analytic[5]]
            
            # print(2.0/5.0/growth*bs_analytic[5])

            # bestfit.append((2.0/5.0/growth*bs_analytic[5])[0])
            print('Fix ce3')
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
        
    print(bestfit)
else:
    # if int(pardict['vary_c4']) == 1:
    #     b4 = bestfit[7]
    #     b2 = (bestfit[5] - bestfit[7])/0.86
    # else:
    #     b2 = (bestfit[5] + bestfit[7])/np.sqrt(2.0)
    #     b2 = (bestfit[5] - bestfit[7])/np.sqrt(2.0)
    
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

if "w" in pardict.keys():
    Plin, Ploop = birdmodels[0].compute_pk(np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3], bestfit[4]]).reshape(-1, 1))
else:
    Plin, Ploop = birdmodels[0].compute_pk(np.array([bestfit[0], bestfit[1], bestfit[2], bestfit[3]]).reshape(-1, 1))

# np.save("Plin_MB.npy", Plin)
# np.save("Ploop_MB.npy", Ploop)

# print(np.shape(Plin))

P_model_lin, P_model_loop, P_model_interp = birdmodels[0].compute_model(
    bs.reshape((-1, 1)), Plin, Ploop, fittingdata.data["x_data"][0]
)

# print(fittingdata.data["x_data"][0][0])

# np.save("FS_pk_ELG_0p28.npy", [fittingdata.data["x_data"][0][0], P_model_interp])
# np.save("FS_pk_fiducial_cosmo_ELG.npy", [fittingdata.data["x_data"][0][0], P_model_interp])
np.save("FS_pk_fiducial_cosmo_QSO.npy", [fittingdata.data["x_data"][0][0], P_model_interp])


chi_squared = birdmodels[0].compute_chi2(P_model_interp, fittingdata.data)
print(chi_squared)