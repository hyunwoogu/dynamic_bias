"""fMRI-DDM correspondence score
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.linalg import expm
sys.path.append('../..')
from src import utils
utils.download_dataset("models/ddm/full")
utils.download_dataset("models/ddm/reduced")
utils.download_dataset("data/processed/behavior")
utils.download_dataset("data/processed/fmri")
utils.download_dataset("data/outputs/fmri")

with open(f"{utils.ORIGIN}/data/outputs/fmri/params_visual_drive.pickle", 'rb') as f:
    bold_params = pickle.load(f)

labels = np.linspace(0,np.pi,120,endpoint=False)
path_output = f"{utils.ORIGIN}/data/processed/fmri/decoding/"
behavior    = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")

#
pop_behavior = []
pop_channel = []  # population data of reconstructed channel responses
for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    with open(path_output+f'decoding_sub-{v_sub:04}.pickle', 'rb') as f: 
        chan_recon = pickle.load(f)
    pop_behavior.append(chan_recon['behav'])
    pop_channel.append(chan_recon['chan'])
    
pop_behavior  = pd.concat(pop_behavior)
pop_channel   = np.concatenate(pop_channel, axis=1)
id_data       = pop_behavior.ID
near_data     = np.isin(pop_behavior.ref, [-4,0,4]) * 1

pop_channel_p = np.zeros(pop_channel.shape[:2])*np.nan
for t in range(12):
    for i in range(pop_channel.shape[1]):
        pop_channel_p[t,i] = 2.*utils.pop_vector_decoder(pop_channel[t,i], labels, unit='radian')

#
def simulate_ts(model, Ts, data, model_type='full'):
    """simulate the time series
        Ts: evaluation time points
    """
    # setting
    sin_sc = np.sin(model.m)
    cos_sc = np.cos(model.m)
    
    if model_type == 'full':
        w_K = model.fitted_params[0]*np.pi/90.
        
    w_E, w_P, w_D, w_r, w_b = [p*np.pi/90. for p in model.fitted_params[(-7):(-2)]]
    lam, s = model.fitted_params[(-2):]
    
    # efficient encoding 
    kappa = model.kappa
    p,F = model.efficient_encoding(s)
    p_e = np.sqrt(1./np.clip(w_E,model.eps,None))
    F_interp = model.F_interp(F)
    P0  = np.exp(p_e*(np.cos((F.reshape(-1,1)-F_interp(data['stim'])))-1.))
    P0  = P0*p.reshape(-1,1)
    P0 /= P0.sum(axis=0,keepdims=True)

    # transition matrices
    Lp = model.Hdiffu/2.*np.power(w_P,2) # production transition

    if model_type == 'reduced':
        L = model.Hdiffu/2.*np.power(w_D,2)
    elif model_type == 'full':
        L = model.Hdiffu/2.*np.power(w_D,2) - model.Hdrift@np.diag(kappa*w_K)

    # density propagation
    P  = [np.zeros_like(P0)*np.nan, np.zeros_like(P0)*np.nan] # CW, CCW
    L1 = [L*model.delay[0][0], L*model.delay[0][1]] # 1st delay transitions for early and late DM conditions
    L2 = [L*model.delay[1][0], L*model.delay[1][1]] # 2nd delay transitions for early and late DM conditions

    # forward propagation to conditioned distribution
    t_seg = 6. # time segment
    dT    = [t*t_seg for t in model.delay_condition]
    dTn   = [np.sum(Ts<=dt) for dt in dT]
    mt    = np.zeros([len(Ts), P0.shape[-1], 2]) * np.nan
    Ld    = [L1, L2]
    Pd    = [np.zeros_like(P0)*np.nan, np.zeros_like(P0)*np.nan] # CW, CCW
    Pdd   = [np.zeros_like(P0)*np.nan, np.zeros_like(P0)*np.nan] # CW, CCW

    for i_delay, delay in enumerate(model.delay_condition):
        idx     = (data['delay']==delay)
        _P      = expm(Ld[0][i_delay]) @ P0
        _P_cw   = (1.-lam)*model.mask_cw*_P + lam*model.mask_ccw*_P
        _P_ccw  = lam*model.mask_cw*_P + (1.-lam)*model.mask_ccw*_P
        Pd[0][:,idx] = _P_cw[:,idx]
        Pd[1][:,idx] = _P_ccw[:,idx]

    # Pre-decision dynamics 
    for i_delay, delay in enumerate(model.delay_condition):
        idx = (data['delay']==delay)
        for i_T, v_T in enumerate(Ts[Ts<=dT[i_delay]]):
            m_pri     = expm(L*v_T/t_seg) @ P0[:,idx]
            m_lik_cw  = (expm(L*(dT[i_delay]-v_T)/t_seg)[:,:,np.newaxis] * model.mask_cw[np.newaxis,:,idx]).sum(axis=1)
            m_lik_ccw = (expm(L*(dT[i_delay]-v_T)/t_seg)[:,:,np.newaxis] * model.mask_ccw[np.newaxis,:,idx]).sum(axis=1)
            m_pos_cw  = ((1.-lam)*m_lik_cw  + lam*m_lik_ccw) * m_pri
            m_pos_ccw = ((1.-lam)*m_lik_ccw + lam*m_lik_cw)  * m_pri
            mt[i_T,idx,0] = np.arctan2(m_pos_cw.T  @ sin_sc, m_pos_cw.T  @ cos_sc)
            mt[i_T,idx,1] = np.arctan2(m_pos_ccw.T @ sin_sc, m_pos_ccw.T @ cos_sc)
            
    # decision-induced bias
    for i_delay, delay in enumerate(model.delay_condition):
        for r in model.relative_reference:
            idx = (data['delay']==delay) & (data['relref']==r)
            Pdd[0][:,idx] = expm(model.Hdrift*(-w_b*(np.abs(r)<model.near_cutoff)-r*w_r)) @ Pd[0][:,idx]
            Pdd[1][:,idx] = expm(model.Hdrift*( w_b*(np.abs(r)<model.near_cutoff)-r*w_r)) @ Pd[1][:,idx]
                
    # Post-decision dynamics
    for i_delay, delay in enumerate(model.delay_condition):
        idx = (data['delay']==delay)
        for i_T, v_T in enumerate(Ts[Ts>dT[i_delay]]):
            m_cw  = expm(L*(v_T-dT[i_delay])/t_seg) @ Pdd[0][:,idx]
            m_ccw = expm(L*(v_T-dT[i_delay])/t_seg) @ Pdd[1][:,idx]
            mt[dTn[i_delay]+i_T,idx,0] = np.arctan2(m_cw.T  @ sin_sc, m_cw.T  @ cos_sc)
            mt[dTn[i_delay]+i_T,idx,1] = np.arctan2(m_ccw.T @ sin_sc, m_ccw.T @ cos_sc)
            
    return mt

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

# 
models = {}
files  = sorted([f for f in os.listdir(f"{utils.ORIGIN}/models/ddm/full") if ('.pkl' in f)])
for v_f in files:
    with open(f"{utils.ORIGIN}/models/ddm/full/{v_f}", 'rb') as f:
        models[v_f[17:21]] = pickle.load(f)

delays = [1,4,7,10]
full_loss = np.nan*np.zeros([len(delays),len(models),12])
Us    = np.arange(-1,100, step=2.) # make it sufficiently large

for i_delay, v_delay in enumerate(delays):
    Ts_ex = np.arange(18.5+v_delay, step=0.5)  
    conv  = utils.conv_operator(Ts_ex,Us,ds=0.5)

    for i_id, (k_id, model) in enumerate(models.items()):

        # data definition
        idx  = id_data==int(k_id)
        data =  {
            'deg': {
                'stim'   : pop_behavior.stim.to_numpy()[idx],
                'ref'    : (pop_behavior.stim+pop_behavior.ref).to_numpy()[idx],
            },
            'relref' : pop_behavior.ref.to_numpy()[idx],
            'delay'  : pop_behavior.Timing.to_numpy()[idx],
            'dm'     : (2 - pop_behavior.choice.to_numpy())[idx],
        }

        #
        sub_dm      = 2 - pop_behavior.choice.to_numpy()[idx]
        sub_channel = pop_channel_p[:,idx]
        sub_near    = np.isin(pop_behavior.ref[idx], [-4,0,4]) * 1

        # model definition
        data = model.convert_unit(data)
        model.gen_mask(data)

        # model prediction
        mt_ex = simulate_ts(model, Ts_ex, data, model_type='full')
        zu,zv = simulate_vd(Ts_ex, data, conv)

        mt_dm = np.zeros(mt_ex.shape[:-1]) * np.nan
        for i_dm, v_dm in enumerate((1-sub_dm).astype(int)): 
            mt_dm[:,i_dm] = mt_ex[:,i_dm,v_dm]
        z_neu = np.array([np.cos(sub_channel),np.sin(sub_channel)])
        mt_c  = conv@np.cos(mt_dm) # conv can be extended to smooth the prediction
        mt_s  = conv@np.sin(mt_dm)

        # 
        zb = bold_params['rho_t']*zu+bold_params['rho_r']*zv
        L1 = np.sum(conv[3:15,:,np.newaxis]*np.cos(sub_channel[:,np.newaxis,:]-mt_dm[np.newaxis,:,:]),axis=1)
        L2 = (z_neu*zb[:,3:15]).sum(axis=0)
        ka = np.abs(conv).sum(axis=-1)[:,np.newaxis] + np.sqrt((zb**2).sum(axis=0))
        L = -np.sum( (L1+L2)[:,(sub_near==1)] /ka[3:15,sub_near==1], axis=-1) # only near data

        #
        full_loss[i_delay, i_id,:] = L
        
    print(i_delay, 'completed')


full_indv_bar = np.nan*np.zeros((len(delays), 50))
for i_id, (k_id, model) in enumerate(models.items()):    
    idx_this = (id_data==int(k_id)) & (near_data==1) # only near data
    full_indv_bar[:,i_id] = -np.mean(full_loss,axis=-1)[:,i_id]/sum(idx_this)
    
delay = delays[np.argmax(np.mean(full_indv_bar,axis=-1))]
print("delay : ", delay, "s")

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


# ========================
# scoring
# ========================
Ts_ex = np.arange(18.5+delay, step=0.5)  
conv  = utils.conv_operator(Ts_ex,Us,ds=0.5)
Us    = np.arange(-1,100, step=2.) # make it sufficiently large

correspondence_scores = np.nan*np.zeros([2,len(models)])

for i_models, (k_model, models) in enumerate({'full': models_full, 'reduced': models_rdcd}.items()):

    for i_id, (k_id, model) in enumerate(models.items()):

        # data definition
        idx  = id_data==int(k_id)
        data =  {
            'deg': {
                'stim'   : pop_behavior.stim.to_numpy()[idx],
                'ref'    : (pop_behavior.stim+pop_behavior.ref).to_numpy()[idx],
            },
            'relref' : pop_behavior.ref.to_numpy()[idx],
            'delay'  : pop_behavior.Timing.to_numpy()[idx],
            'dm'     : (2 - pop_behavior.choice.to_numpy())[idx],
        }

        #
        sub_dm      = 2 - pop_behavior.choice.to_numpy()[idx]
        sub_channel = pop_channel_p[:,idx]
        sub_near    = np.isin(pop_behavior.ref[idx], [-4,0,4]) * 1

        # model definition
        data = model.convert_unit(data)
        model.gen_mask(data)

        # model prediction
        mt_ex = simulate_ts(model, Ts_ex, data, model_type=k_model)
        zu,zv = simulate_vd(Ts_ex, data, conv)

        mt_dm = np.zeros(mt_ex.shape[:-1]) * np.nan
        for i_dm, v_dm in enumerate((1-sub_dm).astype(int)): 
            mt_dm[:,i_dm] = mt_ex[:,i_dm,v_dm]
        z_neu = np.array([np.cos(sub_channel),np.sin(sub_channel)])
        mt_c  = conv@np.cos(mt_dm) # conv can be extended to smooth the prediction
        mt_s  = conv@np.sin(mt_dm)

        # 
        zb = bold_params['rho_t']*zu+bold_params['rho_r']*zv
        L1 = np.sum(conv[3:15,:,np.newaxis]*np.cos(sub_channel[:,np.newaxis,:]-mt_dm[np.newaxis,:,:]),axis=1)
        L2 = (z_neu*zb[:,3:15]).sum(axis=0)
        ka = np.abs(conv).sum(axis=-1)[:,np.newaxis] + np.sqrt((zb**2).sum(axis=0))
        s  = np.mean( (L1+L2)[:,(sub_near==1)] /ka[3:15,sub_near==1] ) # only near data
        correspondence_scores[i_models,i_id] = s
        
    print(k_model, 'done')

# 
utils.mkdir(f"{utils.ORIGIN}/data/outputs/fmri")
with open(f'{utils.ORIGIN}/data/outputs/fmri/results_correspondence_score.pickle', 'wb') as f:
    pickle.dump({
        'full'    : correspondence_scores[0],
        'reduced' : correspondence_scores[1]
    }, f)

# ========================
# scoring: permutation
# ========================
n_perm = 1000
np.random.seed(2023)
scores_perm = np.nan*np.zeros([2,50,n_perm])
Ts_ex = np.arange(18.5+delay, step=0.5)
conv  = utils.conv_operator(Ts_ex,Us,ds=0.5)
Us    = np.arange(-1,100, step=2.) # make it sufficiently large

for i_perm in range(n_perm):

    for i_models, (k_model, models) in enumerate({'full': models_full, 'reduced': models_rdcd}.items()):

        for i_id, (k_id, model) in enumerate(models.items()):

            # shuffle prediction data
            idx_id   = id_data==int(k_id) 
            idx      = idx_id & (near_data==1) # only near data
            idx_perm = np.random.permutation(np.sum(idx))

            data =  {
                'deg': {
                    'stim'   : pop_behavior.stim.to_numpy()[idx][idx_perm],
                    'ref'    : (pop_behavior.stim+pop_behavior.ref).to_numpy()[idx][idx_perm],
                },
                'relref' : pop_behavior.ref.to_numpy()[idx][idx_perm],
                'delay'  : pop_behavior.Timing.to_numpy()[idx][idx_perm],
                'dm'     : (2 - pop_behavior.choice.to_numpy())[idx][idx_perm],
            }

            #
            sub_dm      = 2 - pop_behavior.choice.to_numpy()[idx][idx_perm]
            sub_channel = pop_channel_p[:,idx] # do not shuffle neural data
            sub_near    = np.isin(pop_behavior.ref[idx], [-4,0,4])[idx_perm] * 1

            # model definition
            data = model.convert_unit(data)
            model.gen_mask(data)

            # model prediction
            mt_ex = simulate_ts(model, Ts_ex, data, model_type=k_model)
            zu,zv = simulate_vd(Ts_ex, data, conv)

            mt_dm = np.zeros(mt_ex.shape[:-1]) * np.nan
            for i_dm, v_dm in enumerate((1-sub_dm).astype(int)): 
                mt_dm[:,i_dm] = mt_ex[:,i_dm,v_dm]
            z_neu = np.array([np.cos(sub_channel),np.sin(sub_channel)])
            mt_c  = conv@np.cos(mt_dm) # conv can be extended to smooth the prediction
            mt_s  = conv@np.sin(mt_dm)

            # 
            zb = bold_params['rho_t']*zu+bold_params['rho_r']*zv
            L1 = np.sum(conv[3:15,:,np.newaxis]*np.cos(sub_channel[:,np.newaxis,:]-mt_dm[np.newaxis,:,:]),axis=1)
            L2 = (z_neu*zb[:,3:15]).sum(axis=0)
            ka = np.abs(conv).sum(axis=-1)[:,np.newaxis] + np.sqrt((zb**2).sum(axis=0))
            s  = np.mean( (L1+L2)[:,(sub_near==1)] /ka[3:15,sub_near==1] ) # only near data
            scores_perm[i_models,i_id,i_perm] = s

    print(i_perm, scores_perm[0,:,i_perm].mean(), scores_perm[1,:,i_perm].mean())

#
with open(f'{utils.ORIGIN}/data/outputs/fmri/permutation_correspondence_score.pickle', 'wb') as f:
    pickle.dump(scores_perm, f)