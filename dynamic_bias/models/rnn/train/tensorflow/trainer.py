import numpy as np
import tensorflow as tf
from .model import Model
from ..hyper import hp

__all__ = ['initialize_rnn', 'append_model_performance', 'print_results',
           'tensorize_hp', 'tensorize_trial', 'tensorize_model_performance', 'gen_ti_spec']

def tensorize_hp(hp):
    for k, v in hp.items():
        hp[k] = tf.constant(v, name=k)
    return hp

hp = tensorize_hp(hp)
hp_spec = {}
for k, v in hp.items():
	hp_spec[k] = tf.TensorSpec(v.numpy().shape, tf.dtypes.as_dtype(v.numpy().dtype), name=k)

def initialize_rnn(ti_spec,hp_spec=hp_spec):
    # in TensorFlow, explicit function compilation needed
    model = Model()
    model.__call__.get_concrete_function(
        trial_info=ti_spec,
        hp=hp_spec
    )
    model.rnn_model.get_concrete_function(
        input_data1=ti_spec['u_rho'],
        input_data2=ti_spec['u_the'],
        hp=hp_spec
    )
    return model

def append_model_performance(model_performance, trial_info, Y, Loss, par):
    perf_dm, perf_em = _get_eval(trial_info, Y, par)
    model_performance['loss'].append(Loss['loss'].numpy())
    model_performance['loss_dm'].append(Loss['loss_dm'].numpy())
    model_performance['loss_em'].append(Loss['loss_em'].numpy())
    model_performance['perf_dm'].append(perf_dm)
    model_performance['perf_em'].append(perf_em)
    return model_performance

def print_results(model_performance, iteration):
    print_res = 'Iter. {:4d}'.format(iteration)
    print_res += ' | Discrimination Performance {:0.4f}'.format(model_performance['perf_dm'][iteration]) + \
                 ' | Estimation Performance {:0.4f}'.format(model_performance['perf_em'][iteration]) + \
                 ' | Loss {:0.4f}'.format(model_performance['loss'][iteration])
    print(print_res)

def tensorize_trial(trial_info):
    for k, v in trial_info.items():
        trial_info[k] = tf.constant(v, name=k)
    return trial_info

def tensorize_model_performance(model_performance):
    tensor_mp = {'perf_dm': tf.Variable(model_performance['perf_dm'], trainable=False),
                 'perf_em': tf.Variable(model_performance['perf_em'], trainable=False),
                 'loss':    tf.Variable(model_performance['loss'],    trainable=False),
                 'loss_dm': tf.Variable(model_performance['loss_dm'], trainable=False),
                 'loss_em': tf.Variable(model_performance['loss_em'], trainable=False)}
    return tensor_mp

def gen_ti_spec(trial_info) :
    ti_spec = {}
    for k, v in trial_info.items():
        _shape = list(v.shape)
        if len(_shape) > 1: _shape[0] = None; _shape[1] = None
        ti_spec[k] = tf.TensorSpec(_shape, tf.dtypes.as_dtype(v.dtype), name=k)
    return ti_spec

#
def _get_eval(trial_info, output, par):
    argoutput = tf.math.argmax(output['dm'], axis=2).numpy()
    perf_dm   = np.mean(np.array([argoutput[t,:] == ((trial_info['reference_ori'].numpy() > 0)) for t in par['design_rg']['decision']]))

    cenoutput = tf.nn.softmax(output['em'], axis=2).numpy()
    post_prob = cenoutput
    post_prob = post_prob / (np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)  # Dirichlet normaliation
    post_support = np.linspace(0, np.pi, par['n_ori'], endpoint=False) + np.pi / par['n_ori'] / 2
    pseudo_mean = np.arctan2(post_prob @ np.sin(2 * post_support),
                             post_prob @ np.cos(2 * post_support)) / 2
    estim_sinr = (np.sin(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_cosr = (np.cos(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_mean = np.arctan2(estim_sinr, estim_cosr) / 2
    perf_em = np.mean(np.cos(2. * (trial_info['stimulus_ori'].numpy() * np.pi / par['n_ori'] - estim_mean)))

    return perf_dm, perf_em