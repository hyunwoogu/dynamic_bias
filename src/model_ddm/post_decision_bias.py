"""code for generating post-decision biases for demonstration purposes
"""
import os
import sys
import pickle
import numpy as np
from scipy.interpolate import interp1d
sys.path.append('../..')
from src import utils
from src.utils import par
from src.model_ddm.model import DynamicBiasModel

utils.download_dataset("data/outputs/behavior")
utils.download_dataset("models/ddm/full")
with open(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle", 'rb') as f:
    results = pickle.load(f)
weights = np.median(np.array([v for k,v in results['weights'].items()]),axis=0)
ssb     = lambda m: utils.stimulus_specific_bias(m*90/np.pi, weights, **results['info'])
model   = DynamicBiasModel(stimulus_specific_bias=ssb, weights=weights)

# full model
models = {}
files  = sorted([f for f in os.listdir(f"{utils.ORIGIN}/models/ddm/full") if ('.pkl' in f)])
for v_f in files:
    with open(f"{utils.ORIGIN}/models/ddm/full/{v_f}", 'rb') as f:
        models[v_f[17:21]] = pickle.load(f)

# average of the fitted params
m = model.m
params  = np.stack([model.fitted_params for _, model in models.items()])
drifts  = params[:,0]
diffus  = params[:,3]
npars   = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_b', 'lam', 's']
mparams = dict(zip(npars, np.mean(params,axis=0)))
kappa   = model.kappa
kappa   = np.concatenate([kappa,[kappa[0]]])
kappa   = interp1d(np.concatenate([m,[np.pi]]), kappa)

# efficient coding
p,F   = model.efficient_encoding(mparams['s'])
m_    = np.concatenate([m, [np.pi]])
F_fun = interp1d(m_, np.concatenate([F,[2*np.pi]]), kind='linear')
F_inv = interp1d(np.concatenate([F,[2*np.pi]]), m_, kind='linear')


# Schematic correlation of b_pre and b_post
drfts = np.linspace(min(drifts),max(drifts),num=11)
dffus = np.linspace(min(diffus),max(diffus),num=11)
par['K'] = kappa 
par['F'] = F_fun
par['F_inv'] = F_inv
mparams['w_r'] = 0 # exclude reference attraction

np.random.seed(2023)
par['n_trial'] = 10000
ntimes = int(par['T']*par['n_time'])
ts     = np.linspace(0, par['T'], ntimes) # time space
dt     = ts[1] - ts[0]
dmIs   = [abs(ts-6).argmin(), abs(ts-12).argmin()]
res_ce = np.nan*np.zeros((len(drfts),len(dffus),2,2,ntimes)) # E/L, cw/ccw
res_cb = np.nan*np.zeros((len(drfts),len(dffus),2,ntimes))

for i_w, w_K in enumerate(drfts):    
    for i_d, w_D in enumerate(dffus):
        # extract parameters & update parameters for euler simulation
        mparams['w_K'] = w_K    
        mparams['w_D'] = w_D
        par.update(mparams)

        # 
        res_e_e, res_m_e, res_p_e, res_dm_e = utils.run_euler(dmT=6,  **par)
        res_e_l, res_m_l, res_p_l, res_dm_l = utils.run_euler(dmT=12, **par)
        error_e = utils.wrap(res_m_e - par['stims'].reshape((-1,1,1,1)))
        error_l = utils.wrap(res_m_l - par['stims'].reshape((-1,1,1,1)))
        cw_e, ccw_e = [np.array([utils.circmean(error_e[:,1:4,t,:][res_dm_e[:,1:4,:]==d]) for t in range(ntimes)])*90/np.pi for d in [1,-1]]
        cw_l, ccw_l = [np.array([utils.circmean(error_l[:,1:4,t,:][res_dm_l[:,1:4,:]==d]) for t in range(ntimes)])*90/np.pi for d in [1,-1]]    
        m_e = np.array([utils.circmean(error_e[:,1:4,t,:]*res_dm_e[:,1:4,:]) for t in range(ntimes)])*90/np.pi
        m_l = np.array([utils.circmean(error_l[:,1:4,t,:]*res_dm_l[:,1:4,:]) for t in range(ntimes)])*90/np.pi

        res_cb[i_w, i_d, 0, :] = m_e
        res_cb[i_w, i_d, 1, :] = m_l
        res_ce[i_w, i_d, 0, 0, :], res_ce[i_w, i_d, 0, 1, :] = cw_e, ccw_e
        res_ce[i_w, i_d, 1, 0, :], res_ce[i_w, i_d, 1, 1, :] = cw_l, ccw_l

        print(w_K,w_D)


# estimates of b_pre and b_post ([full/reduced,n_subj,n_timing])
bpre_e = res_cb[:,:,0,dmIs[0]-1]
bpre_l = res_cb[:,:,1,dmIs[1]-1]
bpre   = np.stack([bpre_e,bpre_l],axis=-1)
bpost  = (res_ce[:,:,:,0,-1]-res_ce[:,:,:,1,-1])/2 - bpre
res    = {
    'w_K'   : drfts,
    'w_D'   : dffus,
    'bpre'  : bpre,
    'bpost' : bpost
    }

# save
utils.mkdir(f"{utils.ORIGIN}/data/outputs/ddm")
with open(f"{utils.ORIGIN}/data/outputs/ddm/demo_post_decision_bias.pickle", 'wb') as f:
    pickle.dump(res, f)