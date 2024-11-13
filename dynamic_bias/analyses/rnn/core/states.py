"""utility functions for rnn state-space analyses
"""
import numpy as np
from scipy.special import expit
from .... import utils
from ....models.rnn import Stimulus, update_parameters

def tc(x, dtype=np.float32):
    """type conversion"""
    return x.astype(dtype)

def rnn_model(input_data1, input_data2, hp, var_dict, with_noise, par):
    """numpy version of forward pass

    inputs
    ------
        input_data1 : (time, batch, input_dim)
        input_data2 : (time, batch, input_dim)
        hp : hyperparameters
        var_dict : dictionary of variables
        with_noise : whether to add noise
    """
    _h1 = np.zeros((par['batch_size'], par['n_hidden1']))
    _h2 = np.zeros((par['batch_size'], par['n_hidden2']))
    _n1 = np.zeros((par['batch_size'], par['n_hidden1']))
    _n2 = np.zeros((par['batch_size'], par['n_hidden2']))

    # rnn forward pass and stack the outputs
    h1_stack    = []
    h2_stack    = []
    y_dm_stack  = []
    y_em_stack  = []
    for i_t in range(len(input_data1)):
        rnn_input1  = input_data1[i_t,:,:]
        rnn_input2  = input_data2[i_t,:,:]
        _n1, _n2, _h1, _h2 = rnn_cell(_n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp=hp, var_dict=var_dict, with_noise=with_noise)
        h1_stack.append(tc(_h1))
        h2_stack.append(tc(_h2))
        y_dm_stack.append( tc( (tc(_h1)+tc(_h2)) @ hp['w_out_dm'] ) )
        y_em_stack.append( tc( (tc(_h1)+tc(_h2)) @ hp['w_out_em'] ) )
    h1_stack   = np.stack(h1_stack) 
    h2_stack   = np.stack(h2_stack) 
    y_dm_stack = np.stack(y_dm_stack) 
    y_em_stack = np.stack(y_em_stack) 
    return y_dm_stack, y_em_stack, h1_stack, h2_stack


def rnn_cell(_n1, _n2, _h1, _h2, rnn_input1, rnn_input2, hp, var_dict, with_noise=False):

    alphan = hp['alpha_noise']
    alphah = hp['alpha_neuron']

    # Ornstein-Uhlenbeck noise model
    _n1 = tc(_n1)*np.exp(-alphan) + np.sqrt(1.-np.exp(-2.*alphan))*np.random.normal(size=_n1.shape,scale=hp['noise_rnn_sd'])
    _n2 = tc(_n2)*np.exp(-alphan) + np.sqrt(1.-np.exp(-2.*alphan))*np.random.normal(size=_n2.shape,scale=hp['noise_rnn_sd'])

    # RNN dynamics
    _h1 = tc(_h1)*(1.-alphah) + alphah*expit(
        rnn_input1@hp['w_in1'] + tc(_h1)@var_dict['J11'] + tc(_h2)@var_dict['J21'] + with_noise*_n1
    )
    _h2 = tc(_h2)*(1.-alphah) + alphah*expit(
        rnn_input2@hp['w_in2'] + tc(_h1)@var_dict['J12'] + tc(_h2)@var_dict['J22'] + with_noise*_n2
    )

    return _n1, _n2, _h1, _h2


def run_rnn(par, hp, var_dict, verbose=False): 
    """forward run and aggregate rnn outputs
    """
    n_ori, n_ref, n_batch, n_time = \
        par['n_ori'], len(par['reference']), par['batch_size'], par['n_timesteps']

    r1  = utils.nan([n_time, n_ori, n_ref, n_batch, 48])
    r2  = utils.nan([n_time, n_ori, n_ref, n_batch, 48])
    for i_stim in range(par['n_ori']):
        stim_dist = np.zeros(par['n_ori'])
        stim_dist[i_stim] = 1.
        par['stim_dist'] = stim_dist
        for i_ref, _ in enumerate(par['reference']):
            ref_dist         = np.zeros(len(par['reference']))
            ref_dist[i_ref]  = 1.
            par['ref_dist']  = ref_dist
            par              = update_parameters(par)
            stimulus         = Stimulus(par)
            trial_info       = stimulus.generate_trial()
            _, _, H1, H2     = rnn_model(trial_info['u_rho'], trial_info['u_the'], 
                                         hp, var_dict, with_noise=False, par=par)
            r1[:, i_stim, i_ref] = H1
            r2[:, i_stim, i_ref] = H2

        if (i_stim % 6 == 0) & verbose:
            print("Stimulus", i_stim)
            
    return r1, r2


def project_to_θ (r, J):
    """project the RNN reference population activities to the space of θ"""
    r_cw  = (r[...,:24] @ J[:24,:])
    r_ccw = (r[...,24:] @ J[24:,:])
    r = np.concatenate( [(r_cw [...,:24] + r_cw [...,24:]) / 2.,
                         (r_ccw[...,:24] + r_ccw[...,24:]) / 2.,], axis=-1 )
    return r
