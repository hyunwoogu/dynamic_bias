"""
train RNNs using pytorch
"""
import argparse
import numpy as np
import torch

from dynamic_bias import utils
import dynamic_bias.models.rnn.train.pytorch as rnnt
from dynamic_bias.models.rnn import \
    Stimulus, par, update_parameters, hp, update_hp

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='heterogeneous')
args = parser.parse_args()
MODEL_TYPE = args.model_type
print(f"Running RNN type: {MODEL_TYPE}")

SEED    = 27
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ITER  = 300
N_MODEL = 50

# ==========================================
# Settings
# ==========================================
OUTPUT_DIR = f"{utils.ORIGIN}/models/rnn/{MODEL_TYPE}" + '/network{:02d}'
utils.mkdir(f"{utils.ORIGIN}/models/rnn/{MODEL_TYPE}")

np.random.seed(SEED)
torch.manual_seed(SEED)

par['design'].update({'iti'     : (0, 0.1),
                      'stim'    : (0.1, 0.7),                      
                      'decision': (1.0, 1.6),
                      'delay'   : ((0.7,1.0),(1.6,1.9)),
                      'estim'   : (1.9, 2.0)})

if MODEL_TYPE.endswith('emonly'):
    hp['lam_decision'] = 0. # no decision cost

if MODEL_TYPE.endswith('d2e_ablation'):
    hp['DtoE_off'] = True # no DM to EM connection

if 'heterogeneous' in MODEL_TYPE:
    par['gamma']    = 10. # gamma_D
    par['noise_sd'] = 0.  # noise level, 0 when gamma > 0
    noise_vec  = np.abs(np.sin(np.linspace(0,2*np.pi,24,endpoint=False)))
    noise_vec *= par['gamma']
    par['noise_center'] = noise_vec

par      = update_parameters(par)
stimulus = Stimulus()
hp       = update_hp(hp)
hp       = rnnt.tensorize_hp(hp)
hp       = {k:v.to(DEVICE) for k,v in hp.items()}

# ==========================================
# Training
# ==========================================
for i_model in range(N_MODEL):    
    model_performance = rnnt.initialize_model_performance()
    model = rnnt.initialize_rnn().to(DEVICE)
    for iter in range(N_ITER):
        trial_info        = rnnt.tensorize_trial(stimulus.generate_trial(), DEVICE)
        Y, Loss           = model(trial_info, hp)
        model_performance = rnnt.append_model_performance(model_performance, trial_info, Y, Loss, par)
        rnnt.print_results(model_performance, iter)

    model.model_performance = rnnt.tensorize_model_performance(model_performance)
    torch.save(model, OUTPUT_DIR.format(i_model))