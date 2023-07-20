"""estimation of visual-drive-related parameters
"""
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.append('../..')
from src import utils
utils.download_dataset("data/processed/fmri")

# 
path_output = f"{utils.ORIGIN}/data/processed/fmri/decoding/"
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]

#
bold_params = {}
bold_params['tau_D'] = np.round(np.median(behavior.DM_RT),3)
labels = np.linspace(0,np.pi,120,endpoint=False)
pop_behavior = []
pop_channel  = []  # population data of reconstructed channel responses
for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    with open(path_output+f'decoding_sub-{v_sub:04}.pickle', 'rb') as f: 
        chan_recon = pickle.load(f)
    pop_behavior.append(chan_recon['behav'])
    pop_channel.append(chan_recon['chan'])
pop_behavior = pd.concat(pop_behavior)
pop_channel  = np.concatenate(pop_channel, axis=1)

#  
def simulate_vd(Ts_ex, target_data, delay_data, ref_data, conv, delay_condition=[1,2]):
    """simulate the time series of visual drives
        Ts: evaluation time points
        return z(b)
    """
    s_tgt, c_tgt   = np.sin(target_data), np.cos(target_data)
    s_ref, c_ref   = np.sin(ref_data),    np.cos(ref_data)    
    
    idx_u   = (Ts_ex>=0) & (Ts_ex<1.5)
    idx_v_e = (Ts_ex>=6) & (Ts_ex<6+bold_params['tau_D'])
    idx_v_l = (Ts_ex>=12) & (Ts_ex<12+bold_params['tau_D'])

    h_u     = np.sum(conv[:,idx_u],axis=-1)
    h_v_e   = np.sum(conv[:,idx_v_e],axis=-1)
    h_v_l   = np.sum(conv[:,idx_v_l],axis=-1)
    h_v     = [h_v_e,h_v_l]

    z_u     = np.array([[c_tgt],[s_tgt]]) * h_u.reshape([1,-1,1])
    z_v     = np.zeros_like(z_u) * np.nan
    for i_delay, delay in enumerate(delay_condition):
        idx = (delay_data==delay)
        z_v[:,:,idx] = np.array([[c_ref[idx]],[s_ref[idx]]]) * h_v[i_delay].reshape([1,-1,1])
        
    return z_u, z_v

Ts_ex = np.arange(28.5, step=0.5) # extension: freeze dynamics during estimation epoch
Us    = np.arange(-1,100, step=2.)  # make it sufficiently large
conv  = utils.conv_operator(Ts_ex,Us,ds=0.5)

def forward(rho, ts, conv=conv, t_box=bold_params['tau_D'], r_deg=21): 
    """event-related response function"""
    rho_s, rho_r = rho

    # memory drive
    m_s = np.zeros(len(conv)) # mt_conv @ np.zeros(len(ts))
    m_c = conv.sum(axis=-1)   # mt_conv @ np.ones(len(ts))
    
    # stimulus drive
    idx_s    = (ts>=0)  & (ts<1.5)
    s_s      = np.zeros(len(conv))
    s_c      = np.sum(conv[:,idx_s],axis=-1)
    
    # reference drive
    r_rad    = r_deg*2*np.pi/180.
    s_r, c_r = np.sin(r_rad), np.cos(r_rad)
    idx_r_e  = (ts>=6)  & (ts<6+bold_params['tau_D'])
    idx_r_l  = (ts>=12) & (ts<12+bold_params['tau_D'])
    
    hr_e     = np.sum(conv[:,idx_r_e],axis=-1)
    hr_l     = np.sum(conv[:,idx_r_l],axis=-1)
    r_s_e    = s_r*hr_e
    r_c_e    = c_r*hr_e
    r_s_l    = s_r*hr_l
    r_c_l    = c_r*hr_l

    pred_e  = np.arctan2(m_s+rho_s*s_s+rho_r*r_s_e, m_c+rho_s*s_c+rho_r*r_c_e)
    pred_l  = np.arctan2(m_s+rho_s*s_s+rho_r*r_s_l, m_c+rho_s*s_c+rho_r*r_c_l)
    return pred_e, pred_l

def loss(params, pat_e, pat_l):

    # L2 loss
    rho = params
    prd_e, prd_l = forward(rho, Ts_ex)
    prd_e, prd_l = prd_e[3:15]*90/np.pi, prd_l[3:15]*90/np.pi
    
    # calculate loss
    l2 = np.sum((prd_e-pat_e)**2) + np.sum((prd_l-pat_l)**2)
    return l2


stim_shift = (pop_behavior.stim / 1.5).astype(int).to_numpy()
pop_channel_shift = np.nan*pop_channel
for t in range(12):
    for i_vec, vec in enumerate(pop_channel[t,:,:]):
        pop_channel_shift[t, i_vec, :] = np.roll(vec, -stim_shift[i_vec])


relref_data = pop_behavior.ref.to_numpy()
delay_data  = pop_behavior.Timing.to_numpy()

## 
sub_p21_e = pop_channel_shift[:,(delay_data==1)&(relref_data==21)]
sub_p21_l = pop_channel_shift[:,(delay_data==2)&(relref_data==21)]
sub_n21_e = pop_channel_shift[:,(delay_data==1)&(relref_data==-21)]
sub_n21_l = pop_channel_shift[:,(delay_data==2)&(relref_data==-21)]

##
sub_ep_pat = np.array([utils.pop_vector_decoder(_p, labels, unit='radian') for _p in np.mean(sub_p21_e,axis=1)])*180/np.pi
sub_lp_pat = np.array([utils.pop_vector_decoder(_p, labels, unit='radian') for _p in np.mean(sub_p21_l,axis=1)])*180/np.pi
sub_en_pat = np.array([utils.pop_vector_decoder(_p, labels, unit='radian') for _p in np.mean(sub_n21_e,axis=1)])*180/np.pi
sub_ln_pat = np.array([utils.pop_vector_decoder(_p, labels, unit='radian') for _p in np.mean(sub_n21_l,axis=1)])*180/np.pi

## decalcomania
sub_n21_e_d = np.zeros_like(sub_n21_e)*np.nan
sub_n21_l_d = np.zeros_like(sub_n21_l)*np.nan
for t in range(12):
    for e in range(sub_n21_e_d.shape[1]):
        sub_n21_e_d[t,e] = np.roll(sub_n21_e[t,e][::-1],1)
    for l in range(sub_n21_l_d.shape[1]):
        sub_n21_l_d[t,l] = np.roll(sub_n21_l[t,l][::-1],1)    

sub_e = np.concatenate([sub_p21_e, sub_n21_e_d],axis=1)
sub_l = np.concatenate([sub_p21_l, sub_n21_l_d],axis=1)
sub_e_pat = np.array([utils.pop_vector_decoder(_p, labels, unit='radian') for _p in np.mean(sub_e,axis=1)])*180/np.pi
sub_l_pat = np.array([utils.pop_vector_decoder(_p, labels, unit='radian') for _p in np.mean(sub_l,axis=1)])*180/np.pi

res = minimize(loss, x0=[1,1], args=(sub_e_pat, sub_l_pat), bounds=[[0,30],[0,30]])
bold_params['rho_t'] = res['x'][0]
bold_params['rho_r'] = res['x'][1]

utils.mkdir(f"{utils.ORIGIN}/data/outputs/fmri")
with open(f'{utils.ORIGIN}/data/outputs/fmri/params_visual_drive.pickle', 'wb') as f:
    pickle.dump(bold_params, f)