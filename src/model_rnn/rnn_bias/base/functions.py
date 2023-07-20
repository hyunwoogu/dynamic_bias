import pickle
import numpy as np

__all__ = ['initialize', 'w_design', 'convert_to_rg']

def initialize(dims, gain=1., shape=0.1, scale=1.0):
    w = gain*np.random.gamma(shape, scale, size=dims).astype(np.float32)
    return np.float32(w)

def w_design(w_name, par):
    if ((w_name == 'w_in1') | (w_name == 'w_in2') | (w_name == 'w_out_em')):
        if w_name == 'w_in1':
            n     = par['n_tuned_input']
        elif w_name == 'w_in2':
            n     = par['n_tuned_input']
        else:
            n     = par['n_tuned_output']

        # repeat quo times, and evenly space by rem
        cos_vec   = np.cos(np.arange(2*np.pi, step=2*np.pi/n))
        if w_name == 'w_in1':
            quo, rem  = divmod(par['n_hidden1'], n)
        else:
            quo, rem  = divmod(par['n_hidden2'], n)
        w, w_apdx = np.empty((0,n)), np.empty((0,n))

        if quo > 0: 
            w      = np.tile(np.stack([np.roll(cos_vec, s) for s in range(n)]),quo).T
        if rem > 0:
            w_apdx = np.stack([np.roll(cos_vec, s) for s in np.arange(n, step=int(n/rem))[:rem]])

        if (w_name == 'w_in1') | (w_name == 'w_in2'):
            w = np.concatenate((w,w_apdx), axis=0).T
        else:
            w = np.concatenate((w,w_apdx), axis=0)   * 0.4
    
    elif w_name   == 'w_out_dm':
        w = np.kron(np.eye(2), np.ones(int(par['n_hidden1']/2))).T

    elif w_name   == 'w_rnn11':
        w = np.zeros((par['n_hidden1'],par['n_hidden1']))

    elif w_name   == 'w_rnn21':
        w  = np.zeros((par['n_hidden2'],par['n_hidden1']))

    elif w_name   == 'w_rnn22':
        w = np.zeros((par['n_hidden2'],par['n_hidden2']))

    return w.astype(np.float32)

def convert_to_rg(design, dt):
    """Convert range specs(dictionary) into time-step domain"""
    if type(design) == dict:
        rg_dict = {}
        for k,v in design.items():
            if len(np.shape(v)) == 1:
                start_step = round(v[0] / dt * 1000.)
                end_step = round(v[1] / dt * 1000.)
                rg_dict[k] = np.arange(start_step, end_step, dtype=np.int32)
            else:
                rg_dict[k] = np.concatenate([np.arange(round(i[0] / dt * 1000.),round(i[1] / dt * 1000.), dtype=np.int32) for i in v])
        return rg_dict

    elif type(design) in (tuple, list):
        if len(np.shape(design)) == 1:
            start_step = round(design[0] / dt * 1000.)
            end_step = round(design[1] / dt * 1000.)
            rg = np.arange(start_step, end_step, dtype=np.int32)
        else:
            rg = np.concatenate([np.arange(round(i[0] / dt * 1000.), round(i[1] / dt * 1000.), dtype=np.int32) for i in design])
        return rg