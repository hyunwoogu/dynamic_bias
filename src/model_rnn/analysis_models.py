"""aggregate the models
"""
import sys
sys.path.append('../..')
from src import utils

import pickle
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from rnn_bias import *
import rnn_bias.train as dt

# analysis trained Js 
for model_dir in ['hom', 'het']:
    output_dir = f"{utils.ORIGIN}/models/rnn/{model_dir}" + '/network{:02d}'
    J = np.nan*np.zeros((50,4,48,48))
    for i_model in range(50): 
        model   = tf.saved_model.load(output_dir.format(i_model))
        J[i_model, 0, :, :] = model.var_dict['w_rnn11'].numpy()
        J[i_model, 1, :, :] = model.var_dict['w_rnn12'].numpy()
        J[i_model, 2, :, :] = model.var_dict['w_rnn21'].numpy()
        J[i_model, 3, :, :] = model.var_dict['w_rnn22'].numpy()
    
    utils.mkdir(f"{utils.ORIGIN}/data/outputs/rnn/{model_dir}")
    with open(f"{utils.ORIGIN}/data/outputs/rnn/{model_dir}/trained_J.pickle", 'wb') as f:
        pickle.dump({
            'J11': J[:,0,:,:],
            'J12': J[:,1,:,:],
            'J21': J[:,2,:,:],
            'J22': J[:,3,:,:]
        },f)

# load J's
with open(f'{utils.ORIGIN}/data/outputs/rnn/hom/trained_J.pickle', 'rb') as f:
    J_hom = pickle.load(f)
with open(f'{utils.ORIGIN}/data/outputs/rnn/het/trained_J.pickle', 'rb') as f:
    J_het = pickle.load(f)

## "shorter" structure
par['strength_ref'] = 2.5 
par['batch_size']   = 1
par['kappa']    = 5
par['noise_sd'] = 0

dt.hp['w_in_dm_fix']  = True  
dt.hp['w_in_em_fix']  = True  
dt.hp['w_out_dm_fix'] = True  # assume linear voting from two separate populations
dt.hp['w_out_em_fix'] = True  # assume circular voting from two separate populations

dt.hp = dt.update_hp(dt.hp)
hp    = {k:dt.hp[k].numpy() for k in dt.hp if tf.is_tensor(dt.hp[k])}
hp['noise_rnn_sd'] = 0
hp['alpha_neuron'] = dt.hp['alpha_neuron']
hp['alpha_noise']  = dt.hp['alpha_noise']


## numpy version of forward pass
def rnn_model(input_data1, input_data2, hp, var_dict, with_noise):
    _h1 = np.zeros((par['batch_size'], 48))
    _h2 = np.zeros((par['batch_size'], 48))
    _n1 = _h1*0
    _n2 = _h2*0
    h1_stack    = []
    h2_stack    = []
    y_dm_stack  = []
    y_em_stack  = []
    for i_t in range(len(input_data1)):
        rnn_input1  = input_data1[i_t,:,:]
        rnn_input2  = input_data2[i_t,:,:]
        _n1, _n2, _h1, _h2 = rnn_cell(_n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp=hp, var_dict=var_dict, with_noise=with_noise)
        h1_stack.append(_h1.astype(np.float32))
        h2_stack.append(_h2.astype(np.float32))
        y_dm_stack.append(((_h1.astype(np.float32)+_h2.astype(np.float32)) @ hp['w_out_dm']).astype(np.float32))
        y_em_stack.append(((_h1.astype(np.float32)+_h2.astype(np.float32)) @ hp['w_out_em']).astype(np.float32))
        
    h1_stack   = np.stack(h1_stack) 
    h2_stack   = np.stack(h2_stack) 
    y_dm_stack = np.stack(y_dm_stack) 
    y_em_stack = np.stack(y_em_stack) 
    return y_dm_stack, y_em_stack, h1_stack, h2_stack

def rnn_cell(_n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp, var_dict, with_noise=False):

    _n1 = _n1.astype(np.float32)*np.exp(-hp['alpha_noise']) \
        + np.sqrt(1.-np.exp(-2.*hp['alpha_noise']))*np.random.normal(size=_n1.shape,loc=0,scale=hp['noise_rnn_sd'])
    _n2 = _n2.astype(np.float32)*np.exp(-hp['alpha_noise']) \
        + np.sqrt(1.-np.exp(-2.*hp['alpha_noise']))*np.random.normal(size=_n2.shape,loc=0,scale=hp['noise_rnn_sd'])

    _h1 = _h1.astype(np.float32) * (1. - hp['alpha_neuron']) \
        + hp['alpha_neuron'] * tf.nn.sigmoid(rnn_input1 @ hp['w_in1'] \
            + _h1.astype(np.float32) @ var_dict['J11'] \
            + _h2.astype(np.float32) @ var_dict['J21'] + with_noise*_n1 ).numpy()
    _h2 = _h2.astype(np.float32) * (1. - hp['alpha_neuron']) \
        + hp['alpha_neuron'] * tf.nn.sigmoid(rnn_input2 @ hp['w_in2'] \
            + _h1.astype(np.float32) @ var_dict['J12'] \
            + _h2.astype(np.float32) @ var_dict['J22'] + with_noise*_n2 ).numpy()
    
    return _n1, _n2, _h1, _h2

def run_rnn(par=par):
    """forward run and aggregate rnn outputs
    """
    n_ori, n_ref, n_batch, n_time = \
        par['n_ori'], len(par['reference']), par['batch_size'], par['n_timesteps']    

    r1  = np.zeros([n_time, n_ori, n_ref, n_batch, 48]) * np.nan
    r2  = np.zeros([n_time, n_ori, n_ref, n_batch, 48]) * np.nan
    
    #
    for i_stim in range(par['n_ori']):
        stim_dist = np.zeros(par['n_ori'])
        stim_dist[i_stim] = 1.
        par['stim_dist'] = stim_dist
        for i_ref, _ in enumerate(par['reference']):
            par['reference']  = par['reference']
            ref_dist          = np.zeros(len(par['reference']))
            ref_dist[i_ref]   = 1.
            par['ref_dist']   = ref_dist
            par               = update_parameters(par)
            stimulus          = Stimulus(par)
            trial_info = dt.tensorize_trial(stimulus.generate_trial())
            _, _, H1, H2  = rnn_model(trial_info['neural_input1'], trial_info['neural_input2'], hp, var_dict, with_noise=False)
            r1[:, i_stim, i_ref]    = H1
            r2[:, i_stim, i_ref]    = H2

        if i_stim % 6 == 0:
            print("Stimulus", i_stim)
            
    return r1, r2


# ================================================
# fit pca
# ================================================
dur = 0.6
par['design'].update({'iti'     : (0,   0.1),
                      'stim'    : (0.1, 0.7),
                      'decision': (0.8, 0.8+dur),
                      'delay'   : ((0.7, 0.8),(0.8+dur, 0.8+dur+0.1)),
                      'estim'   : (0.8+dur+0.1,0.8+dur+1.)})
par['reference'] = np.array([0,1])

par        = update_parameters(par)
stimulus   = Stimulus()
ti_spec    = dt.gen_ti_spec(stimulus.generate_trial())
trial_info = dt.tensorize_trial(stimulus.generate_trial())

lam = 0.
var_dict = {
    'J11': (1.-lam)*J_hom['J11'].mean(axis=0) + lam*J_het['J11'].mean(axis=0),
    'J12': (1.-lam)*J_hom['J12'].mean(axis=0) + lam*J_het['J12'].mean(axis=0),
    'J21': (1.-lam)*J_hom['J21'].mean(axis=0) + lam*J_het['J21'].mean(axis=0),
    'J22': (1.-lam)*J_hom['J22'].mean(axis=0) + lam*J_het['J22'].mean(axis=0),
}
r1, r2 = run_rnn()

# fit PCA to the stacked r_theta (for relative reference=0)
rndseed = 2023
r_theta = (r2[:,:,0,:,:24]+r2[:,:,0,:,24:])/2.
pca_shared = PCA(n_components=2, random_state=rndseed)
pca_shared.fit(r_theta.reshape((-1,24)))

# reference directions
ref_input = np.eye(24)
ref_input_trans = np.concatenate([pca_shared.transform(ref_input[s][:24].reshape((1,-1))) for s in range(24)])



# ================================================
# temporal localization 
# ================================================
r1_hom, r2_hom = {}, {}
r1_het, r2_het = {}, {}

par['reference'] = np.array([-1,1])
for dur, durn in zip([0.6, 0.9, 1.2], ['s', 'm', 'l']):
    par['design'].update({'iti'     : (0,   0.1),
                          'stim'    : (0.1, 0.7),
                          'decision': (0.8, 0.8+dur),
                          'delay'   : ((0.7, 0.8),(0.8+dur, 0.8+dur+0.1)),
                          'estim'   : (0.8+dur+0.1,3.)})
    par        = update_parameters(par)
    stimulus   = Stimulus()
    ti_spec    = dt.gen_ti_spec(stimulus.generate_trial())
    trial_info = dt.tensorize_trial(stimulus.generate_trial())

    #
    lam = 0
    var_dict = {
        'J11': (1.-lam)*J_hom['J11'].mean(axis=0) + lam*J_het['J11'].mean(axis=0),
        'J12': (1.-lam)*J_hom['J12'].mean(axis=0) + lam*J_het['J12'].mean(axis=0),
        'J21': (1.-lam)*J_hom['J21'].mean(axis=0) + lam*J_het['J21'].mean(axis=0),
        'J22': (1.-lam)*J_hom['J22'].mean(axis=0) + lam*J_het['J22'].mean(axis=0),
    }
    _r1, _r2   = run_rnn()
    
    r1_hom[durn]   = _r1.copy()
    r2_hom[durn]   = _r2.copy()
    
    # 
    lam = 1
    var_dict = {
        'J11': (1.-lam)*J_hom['J11'].mean(axis=0) + lam*J_het['J11'].mean(axis=0),
        'J12': (1.-lam)*J_hom['J12'].mean(axis=0) + lam*J_het['J12'].mean(axis=0),
        'J21': (1.-lam)*J_hom['J21'].mean(axis=0) + lam*J_het['J21'].mean(axis=0),
        'J22': (1.-lam)*J_hom['J22'].mean(axis=0) + lam*J_het['J22'].mean(axis=0),
    }
    _r1, _r2   = run_rnn()
    
    r1_het[durn]   = _r1.copy()
    r2_het[durn]   = _r2.copy()


utils.mkdir(f"{utils.ORIGIN}/data/outputs/rnn")
with open(f"{utils.ORIGIN}/data/outputs/rnn/results_state_space.pickle", 'wb') as f:
    pickle.dump({
        'pca'   : pca_shared,
        'r1_hom': r1_hom,
        'r2_hom': r2_hom,
        'r1_het': r1_het,
        'r2_het': r2_het
    }, f)
