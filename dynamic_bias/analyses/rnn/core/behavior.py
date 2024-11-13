"""utility functions for rnn behavior analyses
"""
import numpy as np
from scipy.special import softmax
from .... import utils

def softmax_pred_output(pred_output, axis=2):
    # softmax the pred_output
    cenoutput = softmax(pred_output, axis=axis)
    return cenoutput

def posterior_mean(pred_output, axis=2):
    # posterior mean as a function of time
    post_prob = softmax_pred_output(pred_output, axis=axis) # pred_output is assumed to be estimation period
    post_prob = post_prob/(np.sum(post_prob, axis=axis, keepdims=True)+utils.EPS32) # Dirichlet normaliation
    return post_prob

def behavior_summary_dm(pred_output, axis=2):
    post_prob = posterior_mean(pred_output, axis=axis)
    
    # posterior probability collapsed along time
    dv_mean    = np.mean(post_prob,axis=0)
    dv_L       = dv_mean[:,1]
    dv_R       = dv_mean[:,0]
    choice     = dv_L < dv_R   # 1 for cw, 0 for ccw
    return choice

# def behavior_summary_em(trial_info, pred_output, par, axis=2):
def behavior_summary_em(pred_output, par, axis=2):
    post_prob = posterior_mean(pred_output, axis=axis)

    labels = np.linspace(0,180, num=par['n_ori'], endpoint=False)
    pseudo_mean = utils.pop_vector_decoder(post_prob, labels)
    estim_mean  = utils.circmean(pseudo_mean, axis=0) # collapse along time

    return estim_mean # , error, beh_perf
