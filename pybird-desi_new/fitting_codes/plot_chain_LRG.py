# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:29:35 2022

@author: s4479813
"""

import numpy as np
import copy
import emcee
import sys
from chainconsumer import ChainConsumer
from scipy import interpolate
import os
from scipy.stats import iqr, median_absolute_deviation, anderson
import matplotlib.pyplot as plt
import pylab


def read_chain_backend(chainfile):

    reader = emcee.backends.HDFBackend(chainfile)

    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin, flat=True)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples

mock_num = int(sys.argv[1])
kmax = 0.15
kmax = "{:.2f}".format(kmax)
hex_key = True
if hex_key == False:
    key = 'nohex'
else:
    key = 'hex'

dx = [0.25, 0.03, 0.01, 0.001]
truth = [3.0364, 0.6736, 0.1200, 0.02237]
# truth = [1.0, 1.0, 0.45017, 0.0]

a_perp_all = []
a_para_all = []
fsigma8_all = []
m_all = []

a_perp_up = []
a_perp_down = []

a_para_up = []
a_para_down = []

fsigma8_up = []
fsigma8_down = []

m_up = []
m_down = []


for i in range(mock_num):
    c = ChainConsumer()

    # chainfile = str('fit_pec_vel_6dFGSv_k0p%03d_0p%03d_gridcorr_%d_%d_full_norsd_sigmau_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, i+1, sigma_u))
    
    # chainfile = '../../data/DESI_KP4_LRG_pk_0.20hex0.20_3order_hex_marg_Shapefit_mock_'+str(i)+'_bin_0.hdf5'
    
    # chainfile = '../../data/DESI_KP4_LRG_pk_0.20hex0.20_3order_nohex_marg_Shapefit_mock_'+str(i)+'_bin_0.hdf5'
    
    # chainfile = '../../data/DESI_KP4_LRG_pk_'+str(kmax)+'hex'+str(kmax)+'_grid_'+key+'_marg_Shapefit_planck_mock_'+str(i)+'.hdf5'
    
    chainfile = '../../data/DESI_KP4_LRG_pk_'+str(kmax)+'hex'+str(kmax)+'_grid_'+key+'_marg_kmin0p02_fewerbias_bin_0_mock_'+str(i)+'.hdf5'
    # print(chainfile)
    
    try:
        sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
        
        # if os.path.isfile(chainfile) is False:
        #     print('Mock ' +str(i) + ' has not finished.')
        #     continue
        # try:
        #     sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
        # except:
        #     print('Parameter not constraint in mock ' + str(i))
        #     continue
        
        # c.add_chain(sample_new, parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"], name = 'Shapefit old')    
        
        # data = c.analysis.get_summary(parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"])
        
        c.add_chain(sample_new[:, :4], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"], name = 'Full shape fit')
        
        parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"]
        
        data = c.analysis.get_summary()
        
        for j in range(len(truth)):
            try: 
                test = data[parameters[j]][2] - data[parameters[j]][0]
            except:
                data[parameters[j]][2] = truth[j] + 4*dx[j]
                data[parameters[j]][0] = truth[j] - 4*dx[j]
        
        # a_perp_all.append(data[r"$\alpha_{\perp}$"][1])
        # a_perp_down.append(data[r"$\alpha_{\perp}$"][0])
        # a_perp_up.append(data[r"$\alpha_{\perp}$"][2])
        
        # a_para_all.append(data[r"$\alpha_{\parallel}$"][1])
        # a_para_down.append(data[r"$\alpha_{\parallel}$"][0])
        # a_para_up.append(data[r"$\alpha_{\parallel}$"][2])
        
        # fsigma8_all.append(data["$f\sigma_8$"][1])
        # fsigma8_down.append(data["$f\sigma_8$"][0])
        # fsigma8_up.append(data["$f\sigma_8$"][2])
        
        # m_all.append(data["$m$"][1])
        # m_down.append(data["$m$"][0])
        # m_up.append(data["$m$"][2])
        
        a_perp_all.append(data[r"$\ln(10^{10} A_s)$"][1])
        a_perp_down.append(data[r"$\ln(10^{10} A_s)$"][0])
        a_perp_up.append(data[r"$\ln(10^{10} A_s)$"][2])
        
        a_para_all.append(data[r"$h$"][1])
        a_para_down.append(data[r"$h$"][0])
        a_para_up.append(data[r"$h$"][2])
        
        fsigma8_all.append(data[r"$\Omega_{\mathrm{cdm}} h^2$"][1])
        fsigma8_down.append(data[r"$\Omega_{\mathrm{cdm}} h^2$"][0])
        fsigma8_up.append(data[r"$\Omega_{\mathrm{cdm}} h^2$"][2])
        
        m_all.append(data[r"$\Omega_bh^2$"][1])
        m_down.append(data[r"$\Omega_bh^2$"][0])
        m_up.append(data[r"$\Omega_bh^2$"][2])
        
        c.remove_chain()
    except:
        print('Mock ' + str(i) + ' is unconstrained.')
        a_perp_all.append(truth[0])
        a_perp_down.append(truth[0] - 4*dx[0])
        a_perp_up.append(truth[0] + 4*dx[0])
        
        a_para_all.append(truth[1])
        a_para_down.append(truth[1] - 4*dx[1])
        a_para_up.append(truth[1] + 4*dx[1])
        
        fsigma8_all.append(truth[2])
        fsigma8_down.append(truth[2] - 4*dx[2])
        fsigma8_up.append(truth[2] + 4*dx[2])
        
        m_all.append(truth[3])
        m_down.append(truth[3] - 4*dx[3])
        m_up.append(truth[3] + 4*dx[3])
    
    print(i)
    
# a_perp_all_camb = np.array(a_perp_all)
# a_perp_down_camb = np.array(a_perp_down)
# a_perp_up_camb = np.array(a_perp_up)

# a_para_all_camb = np.array(a_para_all)
# a_para_down_camb = np.array(a_para_down)
# a_para_up_camb = np.array(a_para_up)

# fsigma8_all_camb = np.array(fsigma8_all)
# fsigma8_down_camb = np.array(fsigma8_down)
# fsigma8_up_camb = np.array(fsigma8_up)

# m_all_camb = np.array(m_all)
# m_down_camb = np.array(m_down)
# m_up_camb = np.array(m_up)

# a_perp_unc = np.vstack((a_perp_all_camb - a_perp_down_camb, a_perp_up_camb - a_perp_all_camb))
# a_para_unc = np.vstack((a_para_all_camb - a_para_down_camb, a_para_up_camb - a_para_all_camb))
# fsigma8_unc = np.vstack((fsigma8_all_camb - fsigma8_down_camb, fsigma8_up_camb - fsigma8_all_camb))
# m_unc = np.vstack((m_all_camb - m_down_camb, m_up_camb - m_all_camb))

# a_perp_up_mean = np.mean(a_perp_up_camb-a_perp_all_camb)
# a_perp_down_mean = np.mean(a_perp_all_camb - a_perp_down_camb)
# a_perp_mean = np.mean(a_perp_all_camb)

# a_para_up_mean = np.mean(a_para_up_camb - a_para_all_camb)
# a_para_down_mean = np.mean(a_para_all_camb - a_para_down_camb)
# a_para_mean = np.mean(a_para_all_camb)

# fsigma8_up_mean = np.mean(fsigma8_up_camb - fsigma8_all_camb)
# fsigma8_down_mean = np.mean(fsigma8_all_camb - fsigma8_down_camb)
# fsigma8_mean = np.mean(fsigma8_all_camb)

# m_up_mean = np.mean(m_up_camb - m_all_camb)
# m_down_mean = np.mean(m_all_camb - m_down_camb)
# m_mean = np.mean(m_all_camb)

# a_perp_up_unc = np.sqrt(np.sum((a_perp_up_camb - a_perp_mean)**2))/mock_num
# a_perp_down_unc = np.sqrt(np.sum((a_perp_mean - a_perp_down_camb)**2))/mock_num

# a_para_up_unc = np.sqrt(np.sum((a_para_up_camb - a_para_mean)**2))/mock_num
# a_para_down_unc = np.sqrt(np.sum((a_para_mean - a_para_down_camb)**2))/mock_num

# fsigma8_up_unc = np.sqrt(np.sum((fsigma8_up_camb - fsigma8_mean)**2))/mock_num
# fsigma8_down_unc = np.sqrt(np.sum((fsigma8_mean - fsigma8_down_camb)**2))/mock_num

# m_up_unc = np.sqrt(np.sum((m_up_camb - m_mean)**2))/mock_num
# m_down_unc = np.sqrt(np.sum((m_mean - m_down_camb)**2))/mock_num


# a_perp_all = []
# a_para_all = []
# fsigma8_all = []
# m_all = []

# a_perp_up = []
# a_perp_down = []

# a_para_up = []
# a_para_down = []

# fsigma8_up = []
# fsigma8_down = []

# m_up = []
# m_down = []

# bound_up = [1.1, 1.1, 1.0, 1.0]
# bound_low = [0.9, 0.9, 0.0, -1.0]


# for i in range(mock_num):
#     c = ChainConsumer()

#     # chainfile = str('fit_pec_vel_6dFGSv_k0p%03d_0p%03d_gridcorr_%d_%d_full_norsd_sigmau_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, i+1, sigma_u))
    
#     # chainfile = '../../data/DESI_KP4_LRG_pk_0.20hex0.20_3order_nohex_marg_Shapefit_mock_'+str(i)+'_bin_0.hdf5'
    
#     # chainfile = '../../data/chainfile/DESI_KP4_LRG_pk_0.20hex0.20_3order_nohex_marg_Shapefit_mock_'+str(i)+'_bin_0.hdf5'
    
#     # chainfile = '../../data/DESI_KP4_LRG_pk_0.20hex0.20_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mock_'+str(i)+'.hdf5'
    
#     # chainfile = '../../data/DESI_KP4_LRG_pk_'+str(kmax)+'hex'+str(kmax)+'_grid_'+key+'_marg_kmin0p02_fewerbias_bin_0_mock_'+str(i)+'.hdf5'
    
#     chainfile = '../../data/DESI_KP4_LRG_pk_'+kmax+'hex'+kmax+'_3order_'+key+'_marg_Shapefit_mock_' + str(i) +'_bin_0.hdf5'
    
#     sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
    
#     # if os.path.isfile(chainfile) is False:
#     #     print('Mock ' +str(i) + ' has not finished.')
#     #     continue
#     # try:
#     #     sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
#     # except:
#     #     print('Parameter not constraint in mock ' + str(i))
#     #     continue
    
#     c.add_chain(sample_new, parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"], name = 'Shapefit old')    
    
#     parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$", "$b_1$", "$c_2$"]    
    
#     # c.add_chain(sample_new[:, :4], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"], name = 'Full shape fit')
    
#     # parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"]
    
#     data = c.analysis.get_summary()
#     truth = [1.0, 1.0, 0.45017, 0.0]
#     # truth = [3.0364, 0.6736, 0.1200, 0.02237]
    
#     for j in range(len(truth)):
#         try: 
#             test = data[parameters[j]][2] - data[parameters[j]][0]
#         except:
#             data[parameters[j]][2] = bound_up[j]
#             data[parameters[j]][0] = bound_low[j]
    
#     # c.plotter.plot(truth=truth, filename='LRG_mock_' + str(i) + '_Shapefit.png')
    
#     a_perp_all.append(data[r"$\alpha_{\perp}$"][1])
#     a_perp_down.append(data[r"$\alpha_{\perp}$"][0])
#     a_perp_up.append(data[r"$\alpha_{\perp}$"][2])
    
#     a_para_all.append(data[r"$\alpha_{\parallel}$"][1])
#     a_para_down.append(data[r"$\alpha_{\parallel}$"][0])
#     a_para_up.append(data[r"$\alpha_{\parallel}$"][2])
    
#     fsigma8_all.append(data["$f\sigma_8$"][1])
#     fsigma8_down.append(data["$f\sigma_8$"][0])
#     fsigma8_up.append(data["$f\sigma_8$"][2])
    
#     m_all.append(data["$m$"][1])
#     m_down.append(data["$m$"][0])
#     m_up.append(data["$m$"][2])
    
#     # a_perp_all.append(data[r"$\ln(10^{10} A_s)$"][1])
#     # a_perp_down.append(data[r"$\ln(10^{10} A_s)$"][0])
#     # a_perp_up.append(data[r"$\ln(10^{10} A_s)$"][2])
    
#     # a_para_all.append(data[r"$h$"][1])
#     # a_para_down.append(data[r"$h$"][0])
#     # a_para_up.append(data[r"$h$"][2])
    
#     # fsigma8_all.append(data[r"$\Omega_{\mathrm{cdm}} h^2$"][1])
#     # fsigma8_down.append(data[r"$\Omega_{\mathrm{cdm}} h^2$"][0])
#     # fsigma8_up.append(data[r"$\Omega_{\mathrm{cdm}} h^2$"][2])
    
#     # m_all.append(data[r"$\Omega_bh^2$"][1])
#     # m_down.append(data[r"$\Omega_bh^2$"][0])
#     # m_up.append(data[r"$\Omega_bh^2$"][2])
    
#     c.remove_chain()
    
#     print(i)

#Save the mean of the Shapefit/Fullshapefit parameters and their respective uncertainties. 
a_perp_all_class = np.array(a_perp_all)
a_perp_down_class = np.array(a_perp_down)
a_perp_up_class = np.array(a_perp_up)

a_para_all_class = np.array(a_para_all)
a_para_down_class = np.array(a_para_down)
a_para_up_class = np.array(a_para_up)

fsigma8_all_class = np.array(fsigma8_all)
fsigma8_down_class = np.array(fsigma8_down)
fsigma8_up_class = np.array(fsigma8_up)

m_all_class = np.array(m_all)
m_down_class = np.array(m_down)
m_up_class = np.array(m_up)

#These uncertainties will be used later to plot the error bar. 
a_perp_unc_class = np.vstack((a_perp_all_class - a_perp_down_class, a_perp_up_class - a_perp_all_class))
a_para_unc_class = np.vstack((a_para_all_class - a_para_down_class, a_para_up_class - a_para_all_class))
fsigma8_unc_class = np.vstack((fsigma8_all_class - fsigma8_down_class, fsigma8_up_class - fsigma8_all_class))
m_unc_class = np.vstack((m_all_class - m_down_class, m_up_class - m_all_class))

a_perp_up_mean_class = np.mean(a_perp_up_class-a_perp_all_class)
a_perp_down_mean_class = np.mean(a_perp_all_class - a_perp_down_class)
a_perp_mean_class = np.mean(a_perp_all_class)

a_para_up_mean_class = np.mean(a_para_up_class - a_para_all_class)
a_para_down_mean_class = np.mean(a_para_all_class - a_para_down_class)
a_para_mean_class = np.mean(a_para_all_class)

fsigma8_up_mean_class = np.mean(fsigma8_up_class - fsigma8_all_class)
fsigma8_down_mean_class = np.mean(fsigma8_all_class - fsigma8_down_class)
fsigma8_mean_class = np.mean(fsigma8_all_class)

m_up_mean_class = np.mean(m_up_class - m_all_class)
m_down_mean_class = np.mean(m_all_class - m_down_class)
m_mean_class = np.mean(m_all_class)

#Assuming each measurement is indepedent of each other, we use the uncertainty propagation formula to calculate the uncertainty for the mean of the 
#mocks. 
a_perp_up_unc_class = np.sqrt(np.sum((a_perp_up_class - a_perp_mean_class)**2))/mock_num
a_perp_down_unc_class = np.sqrt(np.sum((a_perp_mean_class - a_perp_down_class)**2))/mock_num

a_para_up_unc_class = np.sqrt(np.sum((a_para_up_class - a_para_mean_class)**2))/mock_num
a_para_down_unc_class = np.sqrt(np.sum((a_para_mean_class - a_para_down_class)**2))/mock_num

fsigma8_up_unc_class = np.sqrt(np.sum((fsigma8_up_class - fsigma8_mean_class)**2))/mock_num
fsigma8_down_unc_class = np.sqrt(np.sum((fsigma8_mean_class - fsigma8_down_class)**2))/mock_num

m_up_unc_class = np.sqrt(np.sum((m_up_class - m_mean_class)**2))/mock_num
m_down_unc_class = np.sqrt(np.sum((m_mean_class - m_down_class)**2))/mock_num

x = np.arange(mock_num)

# # summary = np.loadtxt('../../data/chainfile/Summary.csv', delimiter = ',', dtype=str)
# # summary[0][0] = '0'
# # summary = np.array(summary, dtype=float)

# # summary = np.load('summary.npy')

# # index = np.where(summary[:, 9] < summary[:, 10])[0]

# # useful = summary[index, :]

# # mean_a_perp_useful = np.mean(useful[:, 1])
# # mean_a_para_useful = np.mean(useful[:, 3])
# # mean_fsigma8_useful = np.mean(useful[:, 5])
# # mean_m_useful = np.mean(useful[:, 7])

# # a_perp_up_unc_class_useful = np.sqrt(np.sum((a_perp_up_class[index] - a_perp_mean_class)**2))/len(index)
# # a_perp_down_unc_class_useful = np.sqrt(np.sum((a_perp_mean_class - a_perp_down_class[index])**2))/len(index)

# # a_para_up_unc_class_useful = np.sqrt(np.sum((a_para_up_class[index] - a_para_mean_class)**2))/len(index)
# # a_para_down_unc_class_useful = np.sqrt(np.sum((a_para_mean_class - a_para_down_class[index])**2))/len(index)

# # fsigma8_up_unc_class_useful = np.sqrt(np.sum((fsigma8_up_class[index] - fsigma8_mean_class)**2))/len(index)
# # fsigma8_down_unc_class_useful = np.sqrt(np.sum((fsigma8_mean_class - fsigma8_down_class[index])**2))/len(index)

# # m_up_unc_class_useful = np.sqrt(np.sum((m_up_class[index] - m_mean_class)**2))/len(index)
# # m_down_unc_class_useful = np.sqrt(np.sum((m_mean_class - m_down_class[index])**2))/len(index)

# # c = ChainConsumer()
# # chainfile = '../../data/DESI_KP4_LRG_pk_0.20hex0.20_3order_nohex_marg_kmin0p02_fewerbias_bin_0_mean.hdf5'
# # sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
# # c.add_chain(sample_new[:, :4], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"], name = 'Full shape fit')
# # data = c.analysis.get_summary()
# # c.plotter.plot(truth = truth, filename='Mean_LRG_fullShape.png')

#Reads in the fitting result with the mean of the mocks and rescaled covariance matrix. 
# c = ChainConsumer()
# chainfile = '../../data/DESI_KP4_LRG_pk_'+str(kmax) + 'hex'+str(kmax)+'_3order_'+key+'_marg_Shapefit_mock_mean_bin_0.hdf5'
# sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
# c.add_chain(sample_new[:, :4], parameters = [r"$\alpha_{\perp}$", r"$\alpha_{\parallel}$", "$f\sigma_8$", "$m$"], name = 'Shapefit old')
# data = c.analysis.get_summary()
# truth = [1.0, 1.0, 0.45017, 0.0]
# c.plotter.plot(truth = truth, filename='Mean_LRG_mean_'+key+'_Shapefit_'+kmax+'.png')

# plt.figure(1)
c = ChainConsumer()
# chainfile = '../../data/DESI_KP4_LRG_pk_'+str(kmax) + 'hex'+str(kmax)+'_3order_'+key+'_marg_Shapefit_planck_mock_mean.hdf5'
chainfile = '../../data/DESI_KP4_LRG_pk_'+str(kmax)+'hex'+str(kmax)+'_grid_'+key+'_marg_kmin0p02_fewerbias_bin_0_mean.hdf5'
sample_new, log_likelihood_new, max_log_likelihood_new = read_chain_backend(chainfile)
c.add_chain(sample_new[:, :4], parameters = [r"$\ln(10^{10} A_s)$", r"$h$", r"$\Omega_{\mathrm{cdm}} h^2$", r"$\Omega_bh^2$"], name = 'Full shape fit')
data = c.analysis.get_summary()
# # c.plotter.plot(truth = truth, filename='Mean_LRG_mean_'+key+'_Shapefit_'+str(kmax)+'.png')

const = 4

# plt.figure(1)

# f, (ax1, ax2) = plt.subplots(2, 1)

# # p1 = plt.errorbar(x, a_perp_all_camb, yerr=a_perp_unc, fmt='o', label='Shapefit', c = 'blue')
# ax2.hlines(truth[0], -10-const, -const, linestyles='dashed')
# ax1.hlines(0.0, -1.5, x[-1] + 0.5, linestyles='dashed')

# ax2.errorbar(-1-const, a_perp_mean, yerr=np.vstack((a_perp_down_mean, a_perp_up_mean)), fmt = 'o', c = 'orange')
# ax2.errorbar(-2-const, a_perp_mean, yerr=np.vstack((a_perp_down_unc, a_perp_up_unc)), fmt='o', c = 'green')
# ax2.errorbar(-3.0-const, data_shapefit[parameters[0]][1], yerr=np.vstack((data_shapefit[parameters[0]][1] - data_shapefit[parameters[0]][0], data_shapefit[parameters[0]][2] - data_shapefit[parameters[0]][1])), fmt='o', c = 'red')

# # p4 = plt.errorbar(x+0.5, a_perp_all_class, yerr=a_perp_unc_class, fmt='o', label='Mock with Full-Shape fit', c = 'red')
# # plt.errorbar(-1.5, mean_a_perp_useful, yerr=np.vstack((a_perp_down_unc_class_useful, a_perp_up_unc_class_useful)), fmt='o', label='Cut outliers')
# ax2.errorbar(-0.5-const, a_perp_mean_class, yerr=np.vstack((a_perp_down_mean_class, a_perp_up_mean_class)), fmt = 'o', c = 'purple')
# ax2.errorbar(-1.5-const, a_perp_mean_class, yerr=np.vstack((a_perp_down_unc_class, a_perp_up_unc_class)), fmt='o', c = 'brown')
# ax2.errorbar(-2.5-const, data[parameters[0]][1], yerr=np.vstack((data[parameters[0]][1] - data[parameters[0]][0], data[parameters[0]][2] - data[parameters[0]][1])), fmt='o', c = 'blue')

# ax1.errorbar(x, a_perp_all_camb - a_perp_all_class, yerr = np.sqrt(a_perp_unc**2 + a_perp_unc_class**2), fmt = 'o', c = 'r', label = 'Shapefit - Full-Shape fit')
# ax1.errorbar(-0.5, np.mean(a_perp_all_camb - a_perp_all_class), yerr = np.mean(np.sqrt(a_perp_unc**2 + a_perp_unc_class**2)), fmt = 'o', c = 'b', label = 'Mean difference')
# ax1.errorbar(-1.5, np.mean(a_perp_all_camb - a_perp_all_class), yerr = np.mean(np.sqrt(a_perp_unc**2 + a_perp_unc_class**2))/5.0, fmt = 'o', c = 'k', label = 'Standard error')

# ax1.set_xlabel('Mocks')
# ax1.set_ylabel(parameters[0] + ' Differences')
# ax2.set_ylabel(parameters[0])
# # plt.legend()
# ax1.legend(loc=4, prop={'size': 8})
# ax2.legend(['Expect', 'Shapefit mean', 'Shapefit standard error', 'Shapefit with mean of mocks', 'Full-Shape fit mean', 'Full-Shape fit standard error', 'Full-Shape fit with mean of mocks'], loc = 2, prop={'size': 8})
# # plt.savefig('a_perp_mean_LRG.png', dpi=1200)
# # plt.savefig('A_s_mean_LRG.png', dpi=1200)
# plt.savefig('A_s_mean_LRG_compare.png', dpi=1200)

# plt.figure(2)

# f, (ax1, ax2) = plt.subplots(2, 1)

# # p1 = plt.errorbar(x, a_perp_all_camb, yerr=a_perp_unc, fmt='o', label='Shapefit', c = 'blue')
# ax2.hlines(truth[1], -10-const, -const, linestyles='dashed')
# ax1.hlines(0.0, -1.5, x[-1] + 0.5, linestyles='dashed')

# ax2.errorbar(-1-const, a_para_mean, yerr=np.vstack((a_para_down_mean, a_para_up_mean)), fmt = 'o', c = 'orange')
# ax2.errorbar(-2-const, a_para_mean, yerr=np.vstack((a_para_down_unc, a_para_up_unc)), fmt='o', c = 'green')
# ax2.errorbar(-3.0-const, data_shapefit[parameters[1]][1], yerr=np.vstack((data_shapefit[parameters[1]][1] - data_shapefit[parameters[1]][0], data_shapefit[parameters[1]][2] - data_shapefit[parameters[1]][1])), fmt='o', c = 'red')

# # p4 = plt.errorbar(x+0.5, a_perp_all_class, yerr=a_perp_unc_class, fmt='o', label='Mock with Full-Shape fit', c = 'red')
# # plt.errorbar(-1.5, mean_a_perp_useful, yerr=np.vstack((a_perp_down_unc_class_useful, a_perp_up_unc_class_useful)), fmt='o', label='Cut outliers')
# ax2.errorbar(-0.5-const, a_para_mean_class, yerr=np.vstack((a_para_down_mean_class, a_para_up_mean_class)), fmt = 'o', c = 'purple')
# ax2.errorbar(-1.5-const, a_para_mean_class, yerr=np.vstack((a_para_down_unc_class, a_para_up_unc_class)), fmt='o', c = 'brown')
# ax2.errorbar(-2.5-const, data[parameters[1]][1], yerr=np.vstack((data[parameters[1]][1] - data[parameters[1]][0], data[parameters[1]][2] - data[parameters[1]][1])), fmt='o', c = 'blue')

# ax1.errorbar(x, a_para_all_camb - a_para_all_class, yerr = np.sqrt(a_para_unc**2 + a_para_unc_class**2), fmt = 'o', c = 'r', label = 'Shapefit - Full-Shape fit')
# ax1.errorbar(-0.5, np.mean(a_para_all_camb - a_para_all_class), yerr = np.mean(np.sqrt(a_para_unc**2 + a_para_unc_class**2)), fmt = 'o', c = 'b', label = 'Mean difference')
# ax1.errorbar(-1.5, np.mean(a_para_all_camb - a_para_all_class), yerr = np.mean(np.sqrt(a_para_unc**2 + a_para_unc_class**2))/5.0, fmt = 'o', c = 'k', label = 'Standard error')

# ax1.set_xlabel('Mocks')
# ax1.set_ylabel(parameters[1] + ' Differences')
# ax2.set_ylabel(parameters[1])
# # plt.legend()
# ax1.legend(loc=4, prop={'size': 8})
# ax2.legend(['Expect', 'Shapefit mean', 'Shapefit standard error', 'Shapefit with mean of mocks', 'Full-Shape fit mean', 'Full-Shape fit standard error', 'Full-Shape fit with mean of mocks'], loc = 2, prop={'size': 8})
# # plt.savefig('a_perp_mean_LRG.png', dpi=1200)
# # plt.savefig('A_s_mean_LRG.png', dpi=1200)
# plt.savefig('h_mean_LRG_compare.png', dpi=1200)

# plt.figure(3)

# f, (ax1, ax2) = plt.subplots(2, 1)

# # p1 = plt.errorbar(x, a_perp_all_camb, yerr=a_perp_unc, fmt='o', label='Shapefit', c = 'blue')
# ax2.hlines(truth[2], -10-const, -const, linestyles='dashed')
# ax1.hlines(0.0, -1.5, x[-1] + 0.5, linestyles='dashed')

# ax2.errorbar(-1-const, fsigma8_mean, yerr=np.vstack((fsigma8_down_mean, fsigma8_up_mean)), fmt = 'o', c = 'orange')
# ax2.errorbar(-2-const, fsigma8_mean, yerr=np.vstack((fsigma8_down_unc, fsigma8_up_unc)), fmt='o', c = 'green')
# ax2.errorbar(-3.0-const, data_shapefit[parameters[2]][1], yerr=np.vstack((data_shapefit[parameters[2]][1] - data_shapefit[parameters[2]][0], data_shapefit[parameters[2]][2] - data_shapefit[parameters[2]][1])), fmt='o', c = 'red')

# # p4 = plt.errorbar(x+0.5, a_perp_all_class, yerr=a_perp_unc_class, fmt='o', label='Mock with Full-Shape fit', c = 'red')
# # plt.errorbar(-1.5, mean_a_perp_useful, yerr=np.vstack((a_perp_down_unc_class_useful, a_perp_up_unc_class_useful)), fmt='o', label='Cut outliers')
# ax2.errorbar(-0.5-const, fsigma8_mean_class, yerr=np.vstack((fsigma8_down_mean_class, fsigma8_up_mean_class)), fmt = 'o', c = 'purple')
# ax2.errorbar(-1.5-const, fsigma8_mean_class, yerr=np.vstack((fsigma8_down_unc_class, fsigma8_up_unc_class)), fmt='o', c = 'brown')
# ax2.errorbar(-2.5-const, data[parameters[2]][1], yerr=np.vstack((data[parameters[2]][1] - data[parameters[2]][0], data[parameters[2]][2] - data[parameters[2]][1])), fmt='o', c = 'blue')

# ax1.errorbar(x, fsigma8_all_camb - fsigma8_all_class, yerr = np.sqrt(fsigma8_unc**2 + fsigma8_unc_class**2), fmt = 'o', c = 'r', label = 'Shapefit - Full-Shape fit')
# ax1.errorbar(-0.5, np.mean(fsigma8_all_camb - fsigma8_all_class), yerr = np.mean(np.sqrt(fsigma8_unc**2 + fsigma8_unc_class**2)), fmt = 'o', c = 'b', label = 'Mean difference')
# ax1.errorbar(-1.5, np.mean(fsigma8_all_camb - fsigma8_all_class), yerr = np.mean(np.sqrt(fsigma8_unc**2 + fsigma8_unc_class**2))/5.0, fmt = 'o', c = 'k', label = 'Standard error')

# ax1.set_xlabel('Mocks')
# ax1.set_ylabel(parameters[2] + ' Differences')
# ax2.set_ylabel(parameters[2])
# # plt.legend()
# ax1.legend(loc=4, prop={'size': 8})
# ax2.legend(['Expect', 'Shapefit mean', 'Shapefit standard error', 'Shapefit with mean of mocks', 'Full-Shape fit mean', 'Full-Shape fit standard error', 'Full-Shape fit with mean of mocks'], loc = 2, prop={'size': 8})
# # plt.savefig('a_perp_mean_LRG.png', dpi=1200)
# # plt.savefig('A_s_mean_LRG.png', dpi=1200)
# plt.savefig('omega_cdm_mean_LRG_compare.png', dpi=1200)

# plt.figure(4)

# f, (ax1, ax2) = plt.subplots(2, 1)

# # p1 = plt.errorbar(x, a_perp_all_camb, yerr=a_perp_unc, fmt='o', label='Shapefit', c = 'blue')
# ax2.hlines(truth[3], -10-const, -const, linestyles='dashed')
# ax1.hlines(0.0, -1.5, x[-1] + 0.5, linestyles='dashed')

# ax2.errorbar(-1-const, m_mean, yerr=np.vstack((m_down_mean, m_up_mean)), fmt = 'o', c = 'orange')
# ax2.errorbar(-2-const, m_mean, yerr=np.vstack((m_down_unc, m_up_unc)), fmt='o', c = 'green')
# ax2.errorbar(-3.0-const, data_shapefit[parameters[3]][1], yerr=np.vstack((data_shapefit[parameters[3]][1] - data_shapefit[parameters[3]][0], data_shapefit[parameters[3]][2] - data_shapefit[parameters[3]][1])), fmt='o', c = 'red')

# # p4 = plt.errorbar(x+0.5, a_perp_all_class, yerr=a_perp_unc_class, fmt='o', label='Mock with Full-Shape fit', c = 'red')
# # plt.errorbar(-1.5, mean_a_perp_useful, yerr=np.vstack((a_perp_down_unc_class_useful, a_perp_up_unc_class_useful)), fmt='o', label='Cut outliers')
# ax2.errorbar(-0.5-const, m_mean_class, yerr=np.vstack((m_down_mean_class, m_up_mean_class)), fmt = 'o', c = 'purple')
# ax2.errorbar(-1.5-const, m_mean_class, yerr=np.vstack((m_down_unc_class, m_up_unc_class)), fmt='o', c = 'brown')
# ax2.errorbar(-2.5-const, data[parameters[3]][1], yerr=np.vstack((data[parameters[3]][1] - data[parameters[3]][0], data[parameters[3]][2] - data[parameters[3]][1])), fmt='o', c = 'blue')

# ax1.errorbar(x, m_all_camb - m_all_class, yerr = np.sqrt(m_unc**2 + m_unc_class**2), fmt = 'o', c = 'r', label = 'Shapefit - Full-Shape fit')
# ax1.errorbar(-0.5, np.mean(m_all_camb - m_all_class), yerr = np.mean(np.sqrt(m_unc**2 + m_unc_class**2)), fmt = 'o', c = 'b', label = 'Mean difference')
# ax1.errorbar(-1.5, np.mean(m_all_camb - m_all_class), yerr = np.mean(np.sqrt(m_unc**2 + m_unc_class**2))/5.0, fmt = 'o', c = 'k', label = 'Standard error')

# ax1.set_xlabel('Mocks')
# ax1.set_ylabel(parameters[3] + ' Differences')
# ax2.set_ylabel(parameters[3])
# # plt.legend()
# ax1.legend(loc=4, prop={'size': 8})
# ax2.legend(['Expect', 'Shapefit mean', 'Shapefit standard error', 'Shapefit with mean of mocks', 'Full-Shape fit mean', 'Full-Shape fit standard error', 'Full-Shape fit with mean of mocks'], loc = 2, prop={'size': 8})
# # plt.savefig('a_perp_mean_LRG.png', dpi=1200)
# # plt.savefig('A_s_mean_LRG.png', dpi=1200)
# plt.savefig('omega_b_mean_LRG_compare.png', dpi=1200)

#----------------------------------------------------------------------------------------------------------------------------

# plt.figure(2)
# plt.hlines(truth[0], -3.5, x[-1])
# plt.errorbar(x, a_perp_all_class, yerr=a_perp_unc_class, fmt='o')
# plt.errorbar(-1, a_perp_mean_class, yerr=np.vstack((a_perp_down_mean_class, a_perp_up_mean_class)), fmt = 'o', c = 'orange')
# plt.errorbar(-2, a_perp_mean_class, yerr=np.vstack((a_perp_down_unc_class, a_perp_up_unc_class)), fmt='o', c = 'green')
# plt.errorbar(-3, data[parameters[0]][1], yerr=np.vstack((data[parameters[0]][1] - data[parameters[0]][0], data[parameters[0]][2] - data[parameters[0]][1])), fmt='o', c = 'red')
# plt.xlabel('mocks')
# plt.ylabel(parameters[0])
# plt.legend(['Expect', 'Mock', 'Mean', 'Standard error', 'Mean of the mocks'])
# plt.savefig('a_perp_mean_'+kmax+'_'+key+'.png', dpi=300)
# # plt.savefig('As_mean_'+str(kmax)+'_'+key+'.png', dpi=300)

# plt.figure(3)
# plt.hlines(truth[1], -3.5, x[-1])
# plt.errorbar(x, a_para_all_class, yerr=a_para_unc_class, fmt='o')
# plt.errorbar(-1, a_para_mean_class, yerr=np.vstack((a_para_down_mean_class, a_para_up_mean_class)), fmt = 'o', c = 'orange')
# plt.errorbar(-2, a_para_mean_class, yerr=np.vstack((a_para_down_unc_class, a_para_up_unc_class)), fmt='o', c = 'green')
# plt.errorbar(-3, data[parameters[1]][1], yerr=np.vstack((data[parameters[1]][1] - data[parameters[1]][0], data[parameters[1]][2] - data[parameters[1]][1])), fmt='o', c = 'red')
# plt.xlabel('mocks')
# plt.ylabel(parameters[1])
# plt.legend(['Expect', 'Mock', 'Mean', 'Standard error', 'Mean of the mocks'])
# plt.savefig('a_para_mean_'+kmax+'_'+key+'.png', dpi=300)
# # plt.savefig('h_mean_'+str(kmax)+'_'+key+'.png', dpi=300)

# plt.figure(4)
# plt.hlines(truth[2], -3.5, x[-1])
# plt.errorbar(x, fsigma8_all_class, yerr=fsigma8_unc_class, fmt='o')
# plt.errorbar(-1, fsigma8_mean_class, yerr=np.vstack((fsigma8_down_mean_class, fsigma8_up_mean_class)), fmt = 'o', c = 'orange')
# plt.errorbar(-2, fsigma8_mean_class, yerr=np.vstack((fsigma8_down_unc_class, fsigma8_up_unc_class)), fmt='o', c = 'green')
# plt.errorbar(-3, data[parameters[2]][1], yerr=np.vstack((data[parameters[2]][1] - data[parameters[2]][0], data[parameters[2]][2] - data[parameters[2]][1])), fmt='o', c = 'red')
# plt.xlabel('mocks')
# plt.ylabel(parameters[2])
# plt.legend(['Expect', 'Mock', 'Mean', 'Standard error', 'Mean of the mocks'])
# plt.savefig('fsigma8_mean_'+kmax+'_'+key+'.png', dpi=300)
# # plt.savefig('omega_cdm_mean_'+str(kmax)+'_'+key+'.png', dpi=300)

# plt.figure(5)
# plt.hlines(truth[3], -3.5, x[-1])
# plt.errorbar(x, m_all_class, yerr=m_unc_class, fmt='o')
# plt.errorbar(-1, m_mean_class, yerr=np.vstack((m_down_mean_class, m_up_mean_class)), fmt = 'o', c = 'orange')
# plt.errorbar(-2, m_mean_class, yerr=np.vstack((m_down_unc_class, m_up_unc_class)), fmt='o', c = 'green')
# plt.errorbar(-3, data[parameters[3]][1], yerr=np.vstack((data[parameters[3]][1] - data[parameters[3]][0], data[parameters[3]][2] - data[parameters[3]][1])), fmt='o', c = 'red')
# plt.xlabel('mocks')
# plt.ylabel(parameters[3])
# plt.legend(['Expect', 'Mock', 'Mean', 'Standard error', 'Mean of the mocks'])
# plt.savefig('m_mean_'+kmax+'_'+key+'.png', dpi=300)
# # plt.savefig('m_mean_'+str(kmax)+'_'+key+'.png', dpi=300)

#-------------------------------------------------------------------------------------------------------------------------------
mean = np.array([a_perp_mean_class, a_para_mean_class, fsigma8_mean_class, m_mean_class])
uncertainty = np.array([np.vstack((a_perp_down_unc_class, a_perp_up_unc_class)), np.vstack((a_para_down_unc_class, a_para_up_unc_class)), 
                        np.vstack((fsigma8_down_unc_class, fsigma8_up_unc_class)), np.vstack((m_down_unc_class, m_up_unc_class))])

mock = np.array([data[parameters[0]][1], data[parameters[1]][1], data[parameters[2]][1], data[parameters[3]][1]])
unc = np.array([np.vstack((data[parameters[0]][1] - data[parameters[0]][0], data[parameters[0]][2] - data[parameters[0]][1])), 
                np.vstack((data[parameters[1]][1] - data[parameters[1]][0], data[parameters[1]][2] - data[parameters[1]][1])),
                np.vstack((data[parameters[2]][1] - data[parameters[2]][0], data[parameters[2]][2] - data[parameters[2]][1])),
                np.vstack((data[parameters[3]][1] - data[parameters[3]][0], data[parameters[3]][2] - data[parameters[3]][1]))])

# np.save('Shapefit_mean_'+str(kmax)+'_'+key+'.npy', [mean, mock])
# np.save('Shapefit_unc_'+str(kmax)+'_'+key+'.npy', [uncertainty, unc])
np.save('FullShapefit_mean_'+str(kmax)+'_'+key+'.npy', [mean, mock])
np.save('FullShapefit_unc_'+str(kmax)+'_'+key+'.npy', [uncertainty, unc])
