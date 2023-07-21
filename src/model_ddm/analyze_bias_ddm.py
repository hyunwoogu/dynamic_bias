"""fMRI: bias measures in the bold dynamics prediction by ddms
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append('../..')
from scipy.interpolate import interp1d
from src import utils
from src.utils import par

with open(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle", 'rb') as f:
    ssb = pickle.load(f)

with open(f"{utils.ORIGIN}/data/outputs/fmri/params_visual_drive.pickle", 'rb') as f:
    bold_params = pickle.load(f)
    
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]

# 
models_full = {}
models_rdcd = {}

# full model
files  = sorted([f for f in os.listdir(f"{utils.ORIGIN}/models/ddm/full") if ('.pkl' in f)])
for v_f in files:
    with open(f"{utils.ORIGIN}/models/ddm/full/{v_f}", 'rb') as f:
        models_full[v_f[17:21]] = pickle.load(f)
    
# reduced model
files  = sorted([f for f in os.listdir(f"{utils.ORIGIN}/models/ddm/reduced") if ('.pkl' in f)])
for v_f in files:
    with open(f"{utils.ORIGIN}/models/ddm/reduced/{v_f}", 'rb') as f:
        models_rdcd[v_f[17:21]] = pickle.load(f)

# 
def simulate_vd(Ts_ex, data, conv, delay_condition=[1,2]):
    """simulate the time series of visual drives
        Ts: evaluation time points
        return z(b)
    """
    s_tgt, c_tgt   = np.sin(data['stim']), np.cos(data['stim'])
    s_ref, c_ref   = np.sin(data['ref']),  np.cos(data['ref'])
    
    idx_u   = (Ts_ex>=0) & (Ts_ex<1.5)
    idx_v_e = (Ts_ex>=6) & (Ts_ex<6.+bold_params['tau_D'])
    idx_v_l = (Ts_ex>=12) & (Ts_ex<12.+bold_params['tau_D'])

    h_u     = np.sum(conv[:,idx_u],axis=-1)
    h_v_e   = np.sum(conv[:,idx_v_e],axis=-1)
    h_v_l   = np.sum(conv[:,idx_v_l],axis=-1)
    h_v     = [h_v_e,h_v_l]

    z_u     = np.array([[c_tgt],[s_tgt]]) * h_u.reshape([1,-1,1])
    z_v     = np.zeros_like(z_u) * np.nan
    for i_delay, delay in enumerate(delay_condition):
        idx = (data['delay']==delay)
        z_v[:,:,idx] = np.array([[c_ref[idx]],[s_ref[idx]]]) * h_v[i_delay].reshape([1,-1,1])
        
    return z_u, z_v        

# =====================
# bootstrapping
# =====================
n_boot = 1000
Ts_ex  = np.arange(28.5, step=0.5)  # each scenario compared, and this one yielded the best score
Us     = np.arange(-1,100, step=2.) # make it sufficiently large
par['T'] = Ts_ex.max().astype(int)
par['n_time'] = 2
conv  = utils.conv_operator(Ts_ex,Us,ds=0.5)
data = {
    'stim'   : np.repeat(par['stims'], 2*5),
    'relref' : np.tile(np.repeat(par['relref'], 2),24),
    'delay'  : np.tile([1,2], 5*24)
}
data['ref'] = utils.wrap(data['stim']+data['relref'])
zu,zv = simulate_vd(Ts_ex, data, conv)

name_par = {}
name_par['full'] = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_b', 'lam', 's']
name_par['reduced'] = ['w_E', 'w_P', 'w_D', 'w_r', 'w_b', 'lam', 's']

np.random.seed(2023)
ntrial = 1000
par['n_trial'] = ntrial
ntimes = int(par['T']*par['n_time'])+1
ts     = np.linspace(0, par['T'], ntimes) # time space
dt     = ts[1] - ts[0]
dmIs   = [abs(ts-6).argmin(), abs(ts-12).argmin()]

#
res_m  = np.nan*np.zeros([2,50,24,5,2,ntimes,ntrial])
res_dm = np.nan*np.zeros([2,50,24,5,2,ntrial])

for i_models, (k_models, models) in enumerate({'full': models_full, 'reduced': models_rdcd}.items()):
    
    for i_m, (k_m, model) in enumerate(models.items()):

        # extract parameters and exclude reference attraction
        fitted_params = dict(zip(name_par[k_models], model.fitted_params.copy()))
        
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

        #
        res_m [i_models,i_m,:,:,0,:] = np.concatenate([res_e_e[:,:,None],res_m_e],axis=2)
        res_m [i_models,i_m,:,:,1,:] = np.concatenate([res_e_l[:,:,None],res_m_l],axis=2)
        res_dm[i_models,i_m,:,:,0]   = res_dm_e
        res_dm[i_models,i_m,:,:,1]   = res_dm_l
        
        print(k_models, i_m, 'completed')

# 
res_ssb_popm = np.nan*np.zeros([ntrial,2,12,2,24])
res_ssb_indv = np.nan*np.zeros([ntrial,2,12,2,24,50])
m_res        = np.nan*np.zeros([ntrial,2,12,2,2])
baseline_m   = np.nan*np.zeros([ntrial,2,len(Ts_ex),2,2])

# inner loop 
for i_trial in range(ntrial):

    #
    preds = np.nan*np.zeros((51,2,50,24,5,2))
    for i_models in range(2): 
        for i_m in range(50):
            m_c  = conv @ np.cos(res_m[i_models,i_m,:,:,:,:,i_trial].reshape(24*5*2,-1).T)
            m_s  = conv @ np.sin(res_m[i_models,i_m,:,:,:,:,i_trial].reshape(24*5*2,-1).T)
            pred = np.arctan2(
                m_s + bold_params['rho_t']*zu[1] + bold_params['rho_r']*zv[1],
                m_c + bold_params['rho_t']*zu[0] + bold_params['rho_r']*zv[0]
            )
            preds[:,i_models,i_m] = pred.reshape((-1,24,5,2))        


    # stimulus-specific bias
    for i_models in range(2):
        for i_s in range(24):
            for i_t in range(2):
                # population
                res_ssb_popm[i_trial,i_models,:,i_t,i_s] = utils.circmean(preds[3:15,i_models,:,i_s,:,i_t]*90/np.pi, axis=(1,2))

                # individuals
                for i_sub in range(50):                
                    res_ssb_indv[i_trial,i_models,:,i_t,i_s,i_sub] = utils.circmean(preds[3:15,i_models,i_sub,i_s,:,i_t]*90/np.pi, axis=1)


    # decision-consistent bias        
    preds_e = utils.wrap( preds-par['stims'].reshape((1,1,1,-1,1,1)) )
    preds_e = preds_e[3:15]
    res_me  = utils.wrap( res_m[:,:,:,:,:,:,i_trial]-par['stims'].reshape((1,1,-1,1,1,1)) )

    # 
    for i_models in range(2):
        for i_t in range(2):
            for i_tr in range(12):
                particle = preds_e[i_tr,i_models,:,:,1:4,i_t]
                dms      = res_dm[i_models,:,:,1:4,i_t,i_trial]
                m_res[i_trial,i_models,i_tr,i_t,0] = utils.circmean(particle[dms==-1]*90/np.pi)
                m_res[i_trial,i_models,i_tr,i_t,1] = utils.circmean(particle[dms== 1]*90/np.pi)

    for i_models in range(2):
        for i_t in range(2):
            for i_tt, v_tt in enumerate(Ts_ex):
                particle = res_me[i_models,:,:,1:4,i_t,i_tt]
                dms      = res_dm[i_models,:,:,1:4,i_t,i_trial]
                baseline_m[i_trial,i_models, i_tt, i_t, 0] = utils.circmean(particle[dms==-1]*90/np.pi)
                baseline_m[i_trial,i_models, i_tt, i_t, 1] = utils.circmean(particle[dms== 1]*90/np.pi)

    if (i_trial+1) % 100 == 0:
        print(i_trial, 'completed')
        
del res_m, res_dm


# dcb computation
m_res      = m_res @ np.array([-1/2,1/2])
baseline_m = baseline_m @ np.array([-1/2,1/2])

utils.mkdir(f"{utils.ORIGIN}/data/outputs/ddm")
with open(f'{utils.ORIGIN}/data/outputs/ddm/bootstrap_ddm_stimulus_conditioned.pickle', 'wb') as f:
    pickle.dump({
        'pop':   res_ssb_popm,
        'indiv': res_ssb_indv
    }, f)    
with open(f'{utils.ORIGIN}/data/outputs/ddm/bootstrap_ddm_decision_conditioned.pickle', 'wb') as f:
    pickle.dump({
        'm':     baseline_m,
        'convm': m_res
    }, f)