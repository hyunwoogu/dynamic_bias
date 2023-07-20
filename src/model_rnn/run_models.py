"""Run models and aggregate outputs"""
import sys
import pickle
import numpy as np
import tensorflow as tf
sys.path.append('../..')
from src import utils

from rnn_bias import *
import rnn_bias.train as dt
import rnn_bias.analysis as da

MODEL_TYPE = 'heterogeneous' # homogeneous, heterogeneous, heterogeneous_emonly
MODEL_DIR  = 'het'           # hom, het, het_emonly
TIMING     = 'early'         # early, late

n_model  = 50
temp     = f"{utils.ORIGIN}/models/rnn/{MODEL_DIR}" + "/network{:02d}"
temp_out = f"{utils.ORIGIN}/data/interim/rnn/{MODEL_DIR}/{TIMING}" + '/network{:02d}.pkl'
utils.mkdir(f"{utils.ORIGIN}/data/interim/rnn/{MODEL_DIR}/{TIMING}")

# ==========================================
# Settings 1
# ==========================================
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

#
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


# ==========================================
# Settings 2
# ==========================================
if TIMING == 'early':
    # task structure for early DM condition
    par['design'].update({'iti'  : (0, 0.1),
                        'stim' : (0.1, 0.7),                      
                        'decision': (2.5, 3.1),
                        'delay'   : ((0.7,2.5),(3.1,7.3)),
                        'estim' : (7.3,9.1)})

elif TIMING == 'late':
    # task structure for late DM condition
    par['design'].update({'iti'  : (0, 0.1),
                          'stim' : (0.1, 0.7),                      
                          'decision': (4.9, 5.5),
                          'delay'   : ((0.7,4.9),(5.5,7.3)),
                          'estim' : (7.3,9.1)})

par['reference'] = np.arange(-4,4+1)

#
def get_circmean(pred_output_em):
    cenoutput    = da.softmax_pred_output(pred_output_em)
    post_support = np.linspace(0,np.pi,par['n_ori'],endpoint=False)
    post_sinr    = np.sin(2*post_support)
    post_cosr    = np.cos(2*post_support)
    pseudo_mean  = np.arctan2(cenoutput @ post_sinr, cenoutput @ post_cosr)/2
    return pseudo_mean

dt.hp            = dt.update_hp(dt.hp)
par              = update_parameters(par)
stimulus         = Stimulus()
ti_spec          = dt.gen_ti_spec(stimulus.generate_trial())
trial_info       = dt.tensorize_trial(stimulus.generate_trial())

n_ori, n_ref, n_batch, n_time = \
    par['n_ori'], len(par['reference']), par['batch_size'], par['n_timesteps']


# ==========================================
# Save model outputs into intermediate files
# ==========================================
for i_model in range(n_model): 
    model   = tf.saved_model.load(temp.format(i_model))
    
    # 
    dm_behav   = np.zeros([n_ori, n_ref, n_batch]) * np.nan
    em_behav   = np.zeros([n_ori, n_ref, n_batch]) * np.nan
    er_behav   = np.zeros([n_ori, n_ref, n_batch]) * np.nan
    em_readout = np.zeros([n_time, n_ori, n_ref, n_batch]) * np.nan
    er_readout = np.zeros([n_time, n_ori, n_ref, n_batch]) * np.nan
    
    # 
    for i_stim in range(par['n_ori']):
        v_stim    = i_stim * 180./par['n_ori']
        stim_dist = np.zeros(par['n_ori'])
        stim_dist[i_stim] = 1.
        par['stim_dist'] = stim_dist
        for i_ref, v_ref in enumerate(par['reference']):
            par['reference']  = par['reference']
            ref_dist          = np.zeros(len(par['reference']))
            ref_dist[i_ref]   = 1.
            par['ref_dist']   = ref_dist
            par               = update_parameters(par)
            stimulus          = Stimulus(par)
            ti_spec    = dt.gen_ti_spec(stimulus.generate_trial())
            trial_info = dt.tensorize_trial(stimulus.generate_trial())
            pred_output_dm, pred_output_em, _, _  = model.rnn_model(trial_info['neural_input1'], trial_info['neural_input2'], dt.hp)
            output_dm  = pred_output_dm.numpy()[par['design_rg']['decision'],:,:]
            output_em  = pred_output_em.numpy()[par['design_rg']['estim'],:,:]
            _, _, choice            = da.behavior_summary_dm(output_dm, par=par)
            _, estim_mean, error, _ = da.behavior_summary_em({'stimulus_ori': i_stim}, output_em, par=par)
            _circmean = get_circmean(pred_output_em) * 180./np.pi
            em_readout[:, i_stim, i_ref] = _circmean
            dm_behav[i_stim, i_ref] = choice * 1
            em_behav[i_stim, i_ref] = estim_mean * 180/np.pi

        if i_stim % 6 == 0:
            print(f"Network: {i_model+1}", "Stimulus", i_stim) # i_model + 1 makes no sense
    
        with open(temp_out.format(i_model), 'wb') as f:
            pickle.dump({
                'dm': dm_behav,
                'em': em_behav,
                'em_readout': em_readout,
            }, f)