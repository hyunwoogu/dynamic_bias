import numpy as np
from .functions import convert_to_rg

__all__ = ['par', 'update_parameters']

# All the relevant parameters ========================================================================
par = {
	# Experiment design: unit: second(s)
	'design': {'iti' : (0, 1.5),
			   'stim': (1.5, 3.0),
			   'delay'   : ((3.0, 4.0),(5.5,6.0)),
			   'decision': (4.0, 5.5),
			   'estim'   : (6.0, 7.5)},
	'dm_output_range': 'design',  # decision period
	'em_output_range': 'design',  # estim period

	# Mask specs
	'mask_dm': {'iti': 0., 'stim': 0., 'decision': 1., 'delay': 0., 'estim': 0.},  # strengthe
	'mask_em': {'iti': 0., 'stim': 1., 'decision': 1., 'delay': 1., 'estim': 1.},  # strength

	# Decision specs
	'reference': [-4, -3, -2, -1, 1, 2, 3, 4],
	'strength_ref': 2.,
	'strength_decision': 1.,

	# stimulus distribution
	'stim_dist'				: 'uniform', # or a specific input
	'ref_dist'				: 'uniform', # or a specific input

	# Timings and rates
	'dt'                    : 20.,     # time discretization (ms)
	'tau'   			    : 100,     # time constant (ms)

	# Input and noise
	'noise_center'          : np.zeros(24),

	# Tuning function data
	'strength_input'        : 1., # magnitutde scaling factor for von Mises
	'strength_output'       : 1., # magnitutde scaling factor for von Mises
	'kappa'                 : 2,  # concentration scaling factor for von Mises

	# Loss parameters

	# Training specs
	# 'n_iterations'        : 300,

	# Neuronal settings
	'n_tuned_input'	 : 24,   # number of possible orientation-tuned neurons (input)
	'n_tuned_output' : 24,   # number of possible orientation-tuned neurons (input)
	'n_ori'	 	     : 24 ,  # number of possible orientaitons (output)
	'noise_sd'       : 0.01, 
	'n_hidden1' 	 : 48,  
	'n_hidden2' 	 : 48,  

	# Experimental settings
	'batch_size' 	: 128,
}


def update_parameters(par):
	# ranges and masks
	par.update({'design_rg': convert_to_rg(par['design'], par['dt'])})
	par.update({
		'n_timesteps' : sum([len(v) for _ ,v in par['design_rg'].items()]),
		'n_ref'       : len(par['reference']),
	})

	# default settings
	if par['dm_output_range'] == 'design':
		par['dm_output_rg'] = convert_to_rg(par['design']['decision'], par['dt'])
	else:
		par['dm_output_rg'] = convert_to_rg(par['em_output_range'], par['dt'])

	if par['em_output_range'] == 'design':
		_stim     = convert_to_rg(par['design']['stim'], par['dt'])
		_decision = convert_to_rg(par['design']['decision'], par['dt'])
		_delay    = convert_to_rg(par['design']['delay'], par['dt'])
		_delay1   = convert_to_rg(par['design']['delay'][1], par['dt'])
		_estim    = convert_to_rg(par['design']['estim'], par['dt'])
		em_output = np.concatenate((_stim,_decision,_delay,_estim))
		par['em_output_rg']  = em_output
	else:
		par['em_output_rg'] = convert_to_rg(par['em_output_range'], par['dt'])

	## set n_input
	par['n_input'] = par['n_tuned_input']
	par['n_output_dm'] = 2
	par['n_output_em'] = par['n_tuned_output']

	## stimulus distribution
	if par['stim_dist'] == 'uniform':
		par['stim_p'] = np.ones(par['n_ori'])
	else:
		par['stim_p'] = par['stim_dist']
	par['stim_p'] = par['stim_p' ] /np.sum(par['stim_p'])

	if par['ref_dist'] == 'uniform':
		par['ref_p'] = np.ones(par['n_ref'])
	else:
		par['ref_p'] = par['ref_dist']
	par['ref_p'] = par['ref_p' ] /np.sum(par['ref_p'])

	return par

par = update_parameters(par)