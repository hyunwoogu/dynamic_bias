"""Aggregate model outputs from intermediate files"""
import sys
import pickle
import numpy as np
sys.path.append('../..')
from src import utils

from rnn_bias import *
import rnn_bias.train as dt

n_model = 50
n_timesteps = 455
MODEL_TYPE = 'heterogeneous' # homogeneous, heterogeneous, heterogeneous_emonly
MODEL_DIR  = 'het'           # hom, het, het_emonly
TIMING     = 'early'         # early, late
OUTPUT_DIR = f"{utils.ORIGIN}/data/interim/rnn/{MODEL_DIR}/{TIMING}" + '/network{:02d}.pkl'
utils.mkdir(f"{utils.ORIGIN}/data/outputs/rnn/{MODEL_DIR}")

# ==========================================
# Settings
# ==========================================
#
par['design'].update({'iti'  : (0, 0.1),
                      'stim' : (0.1, 0.7),                      
                      'decision': (1.0, 1.6),
                      'delay'   : ((0.7,1.0),(1.6,1.9)),
                      'estim' : (1.9, 2.0)})
par['strength_ref'] = 2. # 
par['gamma'] = 10.       #
par['kappa']          = 5
par['noise_sd']       = 0.

if MODEL_TYPE == 'heterogeneous_emonly':
    dt.hp['lam_decision'] = 0. # no decision cost

dt.hp['w_in_dm_fix']  = True  
dt.hp['w_in_em_fix']  = True  
dt.hp['w_out_dm_fix'] = True  # assume linear voting from two separate populations
dt.hp['w_out_em_fix'] = True  # assume circular voting from two separate populations

if 'heterogeneous' in MODEL_TYPE:
    noise_vec  = np.abs(np.sin(np.linspace(0,2*np.pi,24,endpoint=False)))
    noise_vec *= par['gamma']
    par['noise_center'] = noise_vec

dt.hp            = dt.update_hp(dt.hp)
par              = update_parameters(par)
stimulus         = Stimulus()
ti_spec          = dt.gen_ti_spec(stimulus.generate_trial())

#
dm_behav_agg = np.zeros([n_model, 24, 9, 128]) * np.nan
em_behav_agg = np.zeros([n_model, 24, 9, 128]) * np.nan
er_behav_agg = np.zeros([n_model, 24, 9, 128]) * np.nan

#
em_readout_m_agg = np.zeros([n_model, n_timesteps, 24]) * np.nan # previously rad -> now deg
cm_agg           = np.zeros([n_model, 2, n_timesteps]) * np.nan # deg space

for i_model in range(n_model):
    with open(OUTPUT_DIR.format(i_model), 'rb') as f:
        res_model = pickle.load(f)
        
    # nice, these are all in degree space (in data_meanfield)
    er_behav   = (res_model['em'] - np.arange(180,step=7.5).reshape((-1,1,1)) - 90.) % 180 - 90.
    er_readout = (res_model['em_readout'] - np.arange(180,step=7.5).reshape((1,-1,1,1)) - 90.) % 180 - 90.    
    
    em_readout_m = res_model['em_readout'][:,:,3:6,:].reshape((n_timesteps,24,-1))
    em_readout_m = utils.circmean(em_readout_m,axis=-1)
    
    #
    cm   = np.zeros((2,n_timesteps)) * np.nan
    for i_dm in range(2):
        for t in range(n_timesteps):
            _sin = np.mean(np.sin(er_readout[t,:,3:6,:][res_model['dm'][:,3:6,:]==i_dm]*np.pi/90.))
            _cos = np.mean(np.cos(er_readout[t,:,3:6,:][res_model['dm'][:,3:6,:]==i_dm]*np.pi/90.))
            cm[i_dm,t] = np.arctan2(_sin, _cos) * 90/np.pi
            
    dm_behav_agg[i_model] = res_model['dm']
    em_behav_agg[i_model] = res_model['em']
    er_behav_agg[i_model] = er_behav

    em_readout_m_agg[i_model] = em_readout_m
    cm_agg[i_model]           = cm
    
    print("model", i_model)


# save
with open(f"{utils.ORIGIN}/data/outputs/rnn/{MODEL_DIR}/agg_{TIMING}.pkl", 'wb') as f:
    pickle.dump({
        'dm_behav_agg': dm_behav_agg,
        'em_behav_agg': em_behav_agg,
        'er_behav_agg': er_behav_agg,
        'em_readout_m_agg': em_readout_m_agg,
        'cm_agg': cm_agg
    }, f)