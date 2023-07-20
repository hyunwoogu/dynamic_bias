"""estimation of decision-consistent bias function in ddm
"""
import os
import sys
import pickle
import numpy as np
from scipy.interpolate import interp1d
sys.path.append('../..')
from src import utils
from src.utils import par
from src.model_ddm import model
utils.download_dataset("models/ddm/full")
utils.download_dataset("models/ddm/reduced")

# full model
models_full = {}
files  = sorted([f for f in os.listdir(f"{utils.ORIGIN}/models/ddm/full") if ('.pkl' in f)])
for v_f in files:
    with open(f"{utils.ORIGIN}/models/ddm/full/{v_f}", 'rb') as f:
        models_full[v_f[17:21]] = pickle.load(f)
    
# reduced model
models_rdcd = {}
files  = sorted([f for f in os.listdir(f"{utils.ORIGIN}/models/ddm/reduced") if ('.pkl' in f)])
for v_f in files:
    with open(f"{utils.ORIGIN}/models/ddm/reduced/{v_f}", 'rb') as f:
        models_rdcd[v_f[17:21]] = pickle.load(f)

def nll_EL(par, evi, dm, tim):
    m, s1, s2_add, lam = par
    s2        = np.sqrt(s1**2 + s2_add**2)
    probE     = utils.psi(evi[tim==1], m, s1, lam)
    probL     = utils.psi(evi[tim==2], m, s2, lam)
    sumE_ll   = np.sum(np.log(probE[dm[tim==1]==1])) + np.sum(np.log(1.-probE[dm[tim==1]==0]))
    sumL_ll   = np.sum(np.log(probL[dm[tim==2]==1])) + np.sum(np.log(1.-probL[dm[tim==2]==0]))
    return -sumE_ll - sumL_ll

# 
np.random.seed(2023)
par['n_trial'] = 10000
ntimes = int(par['T']*par['n_time'])
ts     = np.linspace(0, par['T'], ntimes) # time space
dt     = ts[1] - ts[0]
dmIs   = [abs(ts-6).argmin(), abs(ts-12).argmin()]
res_ce = np.nan*np.zeros((2,50,2,2,ntimes)) # E/L, cw/ccw
res_cb = np.nan*np.zeros((2,50,2,ntimes))

name_par = {}
name_par['full'] = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_b', 'lam', 's']
name_par['reduced'] = ['w_E', 'w_P', 'w_D', 'w_r', 'w_b', 'lam', 's']

for i_models, (k_models, models) in enumerate({'full': models_full, 'reduced': models_rdcd}.items()):
    
    for i_m, (k_m, model) in enumerate(models.items()):

        # extract parameters and exclude reference attraction
        fitted_params = dict(zip(name_par[k_models], model.fitted_params.copy()))
        fitted_params['w_r'] = 0
        
        # make K through interpolation
        kappa = np.concatenate([model.kappa,[model.kappa[0]]])
        kappa = interp1d(np.concatenate([model.m,[np.pi]]), kappa)

        # efficient coding
        p,F = model.efficient_encoding(fitted_params['s'])
        m_    = np.concatenate([model.m, [np.pi]])
        F_fun = interp1d(m_, np.concatenate([F,[2*np.pi]]), kind='linear')
        F_inv = interp1d(np.concatenate([F,[2*np.pi]]), m_, kind='linear')

        # update parameters for euler simulation
        par.update(fitted_params)
        par['F'] = F_fun
        par['F_inv'] = F_inv
        
        if   k_models == 'full':
            par['K'] = kappa
        elif k_models == 'reduced':
            par['K'] = lambda x: 0
        
        # 
        res_e_e, res_m_e, res_p_e, res_dm_e = utils.run_euler(dmT=6,  **par)
        res_e_l, res_m_l, res_p_l, res_dm_l = utils.run_euler(dmT=12, **par)
        error_e = utils.wrap(res_m_e - par['stims'].reshape((-1,1,1,1)))
        error_l = utils.wrap(res_m_l - par['stims'].reshape((-1,1,1,1)))

        cw_e, ccw_e = [np.array([utils.circmean(error_e[:,1:4,t,:][res_dm_e[:,1:4,:]==d]) for t in range(ntimes)])*90/np.pi for d in [1,-1]]
        cw_l, ccw_l = [np.array([utils.circmean(error_l[:,1:4,t,:][res_dm_l[:,1:4,:]==d]) for t in range(ntimes)])*90/np.pi for d in [1,-1]]    
        m_e = np.array([utils.circmean(error_e[:,1:4,t,:]*res_dm_e[:,1:4,:]) for t in range(ntimes)])*90/np.pi
        m_l = np.array([utils.circmean(error_l[:,1:4,t,:]*res_dm_l[:,1:4,:]) for t in range(ntimes)])*90/np.pi

        res_cb[i_models, i_m, 0, :] = m_e
        res_cb[i_models, i_m, 1, :] = m_l
        res_ce[i_models, i_m, 0, 0, :], res_ce[i_models, i_m, 0, 1, :] = cw_e, ccw_e
        res_ce[i_models, i_m, 1, 0, :], res_ce[i_models, i_m, 1, 1, :] = cw_l, ccw_l
        
        print(i_m)

bpre_e = res_cb[:,:,0,dmIs[0]-1]
bpre_l = res_cb[:,:,1,dmIs[1]-1]
bpre   = np.stack([bpre_e,bpre_l],axis=-1)
bpost  = (res_ce[:,:,:,0,-1]-res_ce[:,:,:,1,-1])/2 - bpre
res    = {
    'bpre'  : bpre,
    'bpost' : bpost
}
with open(f"{utils.ORIGIN}/data/outputs/ddm/results_decision_consistent_bias.pickle", 'wb') as f:
    pickle.dump(res, f)