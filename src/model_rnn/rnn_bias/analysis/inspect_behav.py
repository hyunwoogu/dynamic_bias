import numpy as np
import tensorflow as tf

__all__ = ['softmax_pred_output','behavior_summary_dm', 'behavior_summary_em']

def softmax_pred_output(pred_output):
    # softmax the pred_output
    cenoutput = tf.nn.softmax(pred_output, axis=2)
    cenoutput = cenoutput.numpy()
    return cenoutput

# Edited (in order to minimize overhead)
def behavior_summary_dm(pred_output, par):
    cenoutput  = softmax_pred_output(pred_output) # pred_output is assumed to be decision period
    
    # posterior probability as a function of time
    post_prob  = cenoutput
    post_prob  = post_prob/(np.sum(post_prob, axis=2, keepdims=True)+np.finfo(np.float32).eps) # Dirichlet normaliation
    
    # posterior probability collapsed along time
    dv_mean    = np.mean(post_prob,axis=0)
    dv_L       = dv_mean[:,1]
    dv_R       = dv_mean[:,0]
    choice     = dv_L > dv_R
    return dv_L, dv_R, choice


def behavior_summary_em(trial_info, pred_output, par):
    cenoutput = softmax_pred_output(pred_output) # pred_output is assumed to be estimation period
    
    # posterior mean as a function of time
    post_prob = cenoutput
    post_prob = post_prob/(np.sum(post_prob, axis=2, keepdims=True)+np.finfo(np.float32).eps) # Dirichlet normaliation
    post_support = np.linspace(0,np.pi,par['n_ori'],endpoint=False)
    post_sinr = np.sin(2*post_support)
    post_cosr = np.cos(2*post_support)
    pseudo_mean = np.arctan2(post_prob @ post_sinr, post_prob @ post_cosr)/2
    
    # posterior mean collapsed along time
    estim_sinr = (np.sin(2*pseudo_mean)).mean(axis=0)
    estim_cosr = (np.cos(2*pseudo_mean)).mean(axis=0)
    estim_mean = np.arctan2(estim_sinr, estim_cosr)/2
    
    ## Quantities for plotting
    ground_truth  = trial_info['stimulus_ori']
    ground_truth  = ground_truth * np.pi/par['n_ori']
    raw_error     = estim_mean - ground_truth
    error         = (raw_error - np.pi/2.) % (np.pi) - np.pi/2.
    beh_perf      = np.cos(2.*(ground_truth - estim_mean))

    return ground_truth, estim_mean, error, beh_perf