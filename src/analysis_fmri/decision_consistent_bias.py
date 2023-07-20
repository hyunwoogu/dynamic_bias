"""fMRI: estimation of decision-consistent bias function in bold responses
"""
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.append('../..')
from src import utils

utils.mkdir(f"{utils.ORIGIN}/data/outputs/fmri")

# load dataset
utils.download_dataset("data/processed/behavior")
utils.download_dataset("data/processed/fmri")
utils.download_dataset("data/outputs/fmri")
with open(f"{utils.ORIGIN}/data/outputs/fmri/params_visual_drive.pickle", 'rb') as f:
    bold_params = pickle.load(f)
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")

# parameters
Ts_ex = np.arange(28.5, step=0.5)  # each scenario compared, and this one yielded the best score
Us    = np.arange(-1,100, step=2.) # make it sufficiently large
dmIs  = [abs(Ts_ex-6).argmin(), abs(Ts_ex-12).argmin()]
conv  = utils.conv_operator(Ts_ex,Us,ds=0.5)
alph_u, alph_v = bold_params['rho_t'], bold_params['rho_r']

# functions
def underlying(tt,a0,b0,a1,b1,t_dm):
    tidx =np.where(tt>=t_dm)[0][0]
    # pre_dm
    ut        = a0+b0*tt
    # post_dm
    u_dm      = ut[tidx] + a1
    ut[tidx:] = b1*(tt[tidx:]-t_dm) + u_dm
    return ut

def forward(rho, ts, underlying_deg, conv=conv, r_deg=0): 
    """event-related response function, given underlying dynamics"""
    rho_s, rho_r = rho

    # memory drive
    m_s = conv@np.sin(underlying_deg*np.pi/90)
    m_c = conv@np.cos(underlying_deg*np.pi/90)
    
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

def joint_loss(params, patts):
    a0,b0,a1,b1 = params
    
    u_y_e = forward([alph_u,alph_v],Ts_ex,underlying(Ts_ex,a0,b0,a1,b1,6))[0][3:15]*90/np.pi
    u_y_l = forward([alph_u,alph_v],Ts_ex,underlying(Ts_ex,a0,b0,a1,b1,12))[1][3:15]*90/np.pi

    # calculate joint loss
    l2 = np.sum((patts[0]-u_y_e)**2) + np.sum((patts[1]-u_y_l)**2)
    return l2


# 
path_output = f"{utils.ORIGIN}/data/processed/fmri/decoding/"
pop_behavior = []
pop_channel  = []  # population data of reconstructed channel responses
for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    with open(path_output+f'decoding_sub-{v_sub:04}.pickle', 'rb') as f: 
        chan_recon = pickle.load(f)
    pop_behavior.append(chan_recon['behav'])
    pop_channel.append(chan_recon['chan'])
pop_behavior = pd.concat(pop_behavior)
pop_channel = np.concatenate(pop_channel, axis=1)


# decoding
labels   = np.linspace(0,np.pi,120,endpoint=False)
stim_shift = (pop_behavior.stim / 1.5).astype(int).to_numpy()
pop_channel_shift = np.nan*pop_channel
for t in range(12):
    for i_vec, vec in enumerate(pop_channel[t,:,:]):
        pop_channel_shift[t, i_vec, :] = np.roll(vec, -stim_shift[i_vec])

dyn_cond = np.nan*np.empty([2,12,2])
for i_timing, v_timing in enumerate([1,2]):
    for i_c, v_c in enumerate([1,2]):
        idx = (pop_behavior.Timing==v_timing) & (pop_behavior.choice==v_c) & np.isin(pop_behavior.ref, [-4,0,4])
        for t in range(12):
            dyn_cond[i_timing,t,i_c] = utils.pop_vector_decoder(np.mean(pop_channel_shift[t,idx],axis=0), labels, unit='radian')

sse_dyn = np.nan*np.empty((12,50,24,2))
for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    for i_s, v_s in enumerate(np.unique(behavior.stim)):
        for i_t, v_t in enumerate([1,2]):
            idx = (pop_behavior.ID==v_sub) & (behavior.stim==v_s) & (behavior.Timing==v_t)
            for i_tr in range(12):
                sse_dyn[i_tr,i_sub,i_s,i_t] = utils.pop_vector_decoder(np.mean(pop_channel_shift[i_tr,idx],axis=0), labels, unit='radian')

# decoding - save
with open(f"{utils.ORIGIN}/data/outputs/fmri/mean_bold_conditioned.pickle", 'wb') as f:
    pickle.dump({
        'indv_stim_cond': sse_dyn, 
        'pop_decision_cond': dyn_cond
    },f)
    

# permutation
n_perm   = 10000
dyn_cond_perm = np.zeros((2,12,2,n_perm)) # timing, dm, trs 
for i_timing, v_timing in enumerate([1,2]):    
    for t in range(12):
        idx        = (pop_behavior.Timing==v_timing) & np.isin(pop_behavior.ref,[-4,0,4])
        choice_vec = pop_behavior.choice[idx].to_numpy()
        for i_perm in range(n_perm):
            choice_permute = np.random.permutation(choice_vec)
            for i_c, v_c in enumerate([1,2]):
                dyn_cond_perm[i_timing,t,i_c,i_perm] = utils.pop_vector_decoder(np.mean(pop_channel_shift[t,idx][choice_permute==v_c],axis=0), labels, unit='radian')
        print(i_timing,i_c,t)

# permutation - save
with open(f'{utils.ORIGIN}/data/outputs/fmri/permutation_bold_decision_consistent.pickle', 'wb') as f:
    pickle.dump(dyn_cond_perm,f)



# dcb - population
pop_channel_shift_p = np.zeros(pop_channel_shift.shape[:2])*np.nan
for t in range(12):
    for i in range(pop_channel_shift.shape[1]):        
        pop_channel_shift_p[t,i] = utils.pop_vector_decoder(pop_channel_shift[t,i], labels, unit='radian')*180/np.pi

patt_pop = (dyn_cond[:,:,0]-dyn_cond[:,:,1])*90/np.pi
_res = minimize(joint_loss, x0=[0,0,0,0], args=(patt_pop), bounds=[[-20,20],[-3,3],[-20,20],[-3,3]])
b_pop = underlying(Ts_ex,*_res['x'],6)[dmIs[0]], underlying(Ts_ex,*_res['x'],12)[dmIs[1]]

# dcb - individual
dyn_cond_indv = np.nan*np.empty([50,2,12,2])
for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    for i_timing, v_timing in enumerate([1,2]):
        for i_c, v_c in enumerate([1,2]):
            idx = (pop_behavior.ID==v_sub) & (pop_behavior.Timing==v_timing) & (pop_behavior.choice==v_c) & np.isin(pop_behavior.ref, [-4,0,4])
            for t in range(12):
                dyn_cond_indv[i_sub,i_timing,t,i_c] = utils.pop_vector_decoder(np.mean(pop_channel_shift[t,idx],axis=0), labels, unit='radian')

# dcb - fit a functional form to participant-wise data
pattt_sub = (dyn_cond_indv[:,:,:,0]-dyn_cond_indv[:,:,:,1])*180/np.pi/2
params_el_sub = np.nan*np.zeros([50,2,4])
b_pre   = np.nan*np.zeros([50,2])
b_post  = np.nan*np.zeros([50,2])
b       = np.nan*np.zeros([50,2])
time_dm = np.array([6,12])
time_em = np.array([18,18]) 

for i_id, v_id in enumerate(np.unique(pop_behavior.ID)):
    # Early
    idx = (pop_behavior.ID==v_id) & (pop_behavior.Timing==1) & (np.abs(pop_behavior.ref)<5) 
    patt_sub_ccw = pop_channel_shift_p[:,idx&(pop_behavior.choice==2)]
    patt_sub_cw  = pop_channel_shift_p[:,idx&(pop_behavior.choice==1)]
    patt_sub = np.concatenate([patt_sub_cw,-patt_sub_ccw],axis=-1)
    pattt_sub[i_id,0] = np.mean(patt_sub,axis=-1) # yet another way
    
    # Late
    idx = (pop_behavior.ID==v_id) & (pop_behavior.Timing==2) & (np.abs(pop_behavior.ref)<5) 
    patt_sub_ccw = pop_channel_shift_p[:,idx&(pop_behavior.choice==2)]
    patt_sub_cw  = pop_channel_shift_p[:,idx&(pop_behavior.choice==1)]
    patt_sub = np.concatenate([patt_sub_cw,-patt_sub_ccw],axis=-1)
    pattt_sub[i_id,1] = np.mean(patt_sub,axis=-1) # yet another way
    
    
for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    _pat = pattt_sub[i_sub]
    _res = minimize(joint_loss, x0=[1,1,0.2,0.2], args=(_pat), 
                    bounds=[[-20,20],[-3,3],[-20,20],[-3,3]])
    
    params_el_sub[i_sub][0] = _res['x']
    params_el_sub[i_sub][1] = _res['x']
    
    _b_pre   = params_el_sub[i_sub][:,0]+params_el_sub[i_sub][:,1]*time_dm
    _b_post0 = _b_pre   + params_el_sub[i_sub][:,2]
    _b       = _b_post0 + params_el_sub[i_sub][:,3]*(time_em-time_dm)
    _b_post  = _b-_b_pre
    
    b_pre[i_sub]  = _b_pre
    b_post[i_sub] = _b_post
    b[i_sub]      = _b

# dcb - save
with open(f"{utils.ORIGIN}/data/outputs/fmri/results_decision_consistent_bias.pickle", 'wb') as f:
    pickle.dump({
        'pop'   : b_pop,
        'bpre'  : b_pre,
        'bpost' : b_post
    }, f)