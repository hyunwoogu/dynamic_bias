import numpy as np
import tensorflow as tf
from rnn_bias.base import par
from rnn_bias.base.functions import initialize, w_design

__all__ = ['hp', 'hp_spec', 'update_hp']

# Model hyperparameters(modifiable)
hp  = {
	'w_in_dm_fix'   : False,
	'w_in_em_fix'   : False,
	'w_out_dm_fix'  : False,
	'w_out_em_fix'  : False,
	'DtoE_off'      : False,
	'EtoD_off'      : False,

	'gain'          : 0.,
	'learning_rate' : 2e-2,
	'dt'            : 20.,
	'grad_max'      : 0.1,  # gradient clipping
	'noise_rnn_sd'  : 0.01, 
	'lam_decision'  : 1.,
	'lam_estim'     : 1.,
	'tau_neuron'    : 100.,
    'tau_noise'     : 200.,
	'batch_size_v'  : np.zeros(par['batch_size']),

	'w_in1'   : w_design('w_in1', par),
	'w_in2'   : w_design('w_in2', par),
	'w_rnn11' : w_design('w_rnn11', par),
	'w_rnn21' : w_design('w_rnn21', par),
	'w_rnn22' : w_design('w_rnn22', par),
	'w_out_dm': w_design('w_out_dm', par),
	'w_out_em': w_design('w_out_em', par)
}

def update_hp(hp):
	hp.update({
		'w_in10'    : initialize((par['n_input'],  par['n_hidden1']),   gain=hp['gain']),
		'w_in20'    : initialize((par['n_input'],  par['n_hidden2']),   gain=hp['gain']),
		'w_rnn110'  : initialize((par['n_hidden1'], par['n_hidden1']),   gain=hp['gain']),
		'w_rnn120'  : initialize((par['n_hidden1'], par['n_hidden2']),   gain=hp['gain']),
		'w_rnn210'  : initialize((par['n_hidden2'], par['n_hidden1']),   gain=hp['gain']),
		'w_rnn220'  : initialize((par['n_hidden2'], par['n_hidden2']),   gain=hp['gain']),

		'w_out_dm0' : initialize((par['n_hidden1'], par['n_output_dm']), gain=hp['gain']),
		'w_out_em0' : initialize((par['n_hidden2'], par['n_output_em']), gain=hp['gain'])
	})

	hp.update({
		'alpha_neuron': np.float32(hp['dt']/hp['tau_neuron']),
		'alpha_noise':  np.float32(hp['dt']/hp['tau_noise'])
	})
	return hp

hp = update_hp(hp)

# Tensorize hp
for k, v in hp.items():
    hp[k] = tf.constant(v, name=k)	

hp_spec = {}
for k, v in hp.items():
	hp_spec[k] = tf.TensorSpec(v.numpy().shape, tf.dtypes.as_dtype(v.numpy().dtype), name=k)