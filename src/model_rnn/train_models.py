"""Train RNN models"""
import sys
sys.path.append('../..')
from src import utils

from rnn_bias import *
import rnn_bias.train as dt
import numpy as np
import tensorflow as tf

MODEL_TYPE = 'heterogeneous' # homogeneous, heterogeneous, heterogeneous_emonly
MODEL_DIR  = 'het'           # hom, het, het_emonly
OUTPUT_DIR = f"{utils.ORIGIN}/models/rnn/{MODEL_DIR}" + '/network{:02d}'
utils.mkdir(f"{utils.ORIGIN}/data/outputs/rnn/{MODEL_DIR}")

# ==========================================
# Settings
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
# Training loop!
# ==========================================
n_iter   = 300
n_model  = 50
n_print  = 10
dt.hp    = dt.update_hp(dt.hp)
par      = update_parameters(par)
stimulus = Stimulus()
ti_spec  = dt.gen_ti_spec(stimulus.generate_trial())

##
for i_model in range(n_model):    
    model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'loss_dm': [], 'loss_em': []}
    model             = dt.initialize_rnn(ti_spec)
    dt.hp             = dt.update_hp(dt.hp)
    for iter in range(n_iter):
        trial_info        = dt.tensorize_trial(stimulus.generate_trial())
        Y, Loss           = model(trial_info, dt.hp)
        model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
        
        # Print
        if iter % n_print == 0: dt.print_results(model_performance, iter)

    model.model_performance = dt.tensorize_model_performance(model_performance)
    tf.saved_model.save(model, OUTPUT_DIR.format(i_model))