"""near-reference variability in ddm
"""
import os
import sys
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import iqr
sys.path.append('../..')
from src import utils
from src.utils import par

utils.download_dataset("data/outputs/behavior")
utils.download_dataset("models/ddm/full")
with open(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle", 'rb') as f:
    ssb = pickle.load(f)

# full model
models = {}
files  = sorted([f for f in os.listdir(f"{utils.ORIGIN}/models/ddm/full") if ('.pkl' in f)])
for v_f in files:
    with open(f"{utils.ORIGIN}/models/ddm/full/{v_f}", 'rb') as f:
        models[v_f[17:21]] = pickle.load(f)

# 
np.random.seed(2023)
par['n_trial'] = 10000
ntimes = int(par['T']*par['n_time'])
ts     = np.linspace(0, par['T'], ntimes) # time space
dt     = ts[1] - ts[0]
dmIs   = [abs(ts-6).argmin(), abs(ts-12).argmin()]
res_ce = np.nan*np.zeros((2,50,2,2,ntimes)) # E/L, cw/ccw
res_cb = np.nan*np.zeros((2,50,2,ntimes))
name_par = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_b', 'lam', 's']
res_iqr = np.nan*np.zeros([2,50,5]) # [w/wo, n_subj, n_ref]

for i_m, (k_m, model) in enumerate(models.items()):

    # extract parameters and exclude reference attraction
    fitted_params = dict(zip(name_par, model.fitted_params.copy()))

    # make K through interpolation
    kappa = np.concatenate([model.kappa,[model.kappa[0]]])
    kappa = interp1d(np.concatenate([model.m,[np.pi]]), kappa)

    # efficient coding
    p,F   = model.efficient_encoding(fitted_params['s'])
    m_    = np.concatenate([model.m, [np.pi]])
    F_fun = interp1d(m_, np.concatenate([F,[2*np.pi]]), kind='linear')
    F_inv = interp1d(np.concatenate([F,[2*np.pi]]), m_, kind='linear')

    # update parameters for euler simulation
    par.update(fitted_params)
    par['F'] = F_fun
    par['F_inv'] = F_inv
    par['K'] = kappa
    
    # 
    res_e_e, res_m_e, res_p_e, res_dm_e = utils.run_euler(dmT=6,  **par)
    res_e_l, res_m_l, res_p_l, res_dm_l = utils.run_euler(dmT=12, **par)
    error_e = utils.wrap(res_p_e - par['stims'].reshape((-1,1,1)))
    error_l = utils.wrap(res_p_l - par['stims'].reshape((-1,1,1)))

    iqrs = np.array([iqr(np.stack([error_e[:,i_r],error_l[:,i_r]]).flatten()*90/np.pi) for i_r in range(5)])
    res_iqr[0,i_m,:] = iqrs
    
    #
    fitted_params['w_b'] = 0
    par.update(fitted_params)
    
    res_e_e, res_m_e, res_p_e, res_dm_e = utils.run_euler(dmT=6,  **par)
    res_e_l, res_m_l, res_p_l, res_dm_l = utils.run_euler(dmT=12, **par)
    error_e = utils.wrap(res_p_e - par['stims'].reshape((-1,1,1)))
    error_l = utils.wrap(res_p_l - par['stims'].reshape((-1,1,1)))

    iqrs = np.array([iqr(np.stack([error_e[:,i_r],error_l[:,i_r]]).flatten()*90/np.pi) for i_r in range(5)])
    res_iqr[1,i_m,:] = iqrs

    print(i_m)

# save
utils.mkdir(f"{utils.ORIGIN}/data/outputs/ddm")
with open(f"{utils.ORIGIN}/data/outputs/ddm/results_near_reference_variability.pickle", 'wb') as f:
    pickle.dump(res_iqr, f)