"""stimulus specific bias in behavior data
"""
from dynamic_bias import utils
from dynamic_bias.analyses.behavior import PsychometricFunction
from dynamic_bias.analyses.behavior import StimulusSpecificBias

OUT_DIR = f"{utils.ORIGIN}/data/outputs/behavior"

# load behavior
behavior  = utils.load_behavior()
stim_list = utils.exp_stim_list()
id_list   = sorted(behavior.ID.unique())

"""
1. Fitting stimulus-specific bias functions
    output : results_stimulus_specific_bias.pickle
"""
ssb = StimulusSpecificBias()
ssb_fits = {}
for i_id, v_id in enumerate(id_list):    
    sub_behav = behavior[behavior.ID == v_id]
    ssb.fit( [utils.circmean(sub_behav.error[sub_behav.stim==s]) for s in stim_list] )
    ssb_fits[v_id] = ssb.weights
    
utils.save(ssb_fits, f'{OUT_DIR}/results_stimulus_specific_bias.pickle')
print('results_stimulus_specific_bias.pickle saved')


"""
2. Fitting stimulus-specific bias modulation at DM timing
    output : results_stimulus_specific_bias_weight.pickle

Unlike DDM, we don't have access to the error patterns at DM timing in the human data. 
Instead, we can estimate the biases at DM timing using:
    (i) the previously fitted EM stimulus-specific bias pattern at EM timing and 
    (ii) choice data at DM timing
Specifically, we fit the amplitudes of the stimulus-specific bias pattern 
(i) in explaining the choice data (ii) using the psychometric functions.

Whenever possible (ex. DDM), we leverage access to the latent error patterns to estimate the biases.
"""
ssb = StimulusSpecificBias()
ssb_fits = utils.load(f"{OUT_DIR}/results_stimulus_specific_bias.pickle")

# fit stimulus-specific bias amplitudes at DM timing
ssb_weights = {}
for i_id, v_id in enumerate(id_list):
    sub_df = behavior[behavior.ID == v_id]
    data = {
        'evidence' : -sub_df.ref.to_numpy(),
        'choice'   : 2.-sub_df.choice.to_numpy(),
        'cond'     : sub_df.Timing.to_numpy(),
        'stim'     : sub_df.stim.to_numpy(),
    }
    pse_fun = lambda s: -ssb(s, ssb_fits[v_id])
    psi = PsychometricFunction(
        link='gaussian_pse',
        pse=pse_fun,
        sequential_inequality=['s']
    )
    psi.fit(data, constrain_params=['lam'])
    ssb_weights[v_id] = dict(zip(['E','L'], psi.fitted_params[:,0]))

utils.save(ssb_weights, f'{OUT_DIR}/results_stimulus_specific_bias_weight.pickle')
print('results_stimulus_specific_bias_weight.pickle saved')

"""
3. Finding fixed points of the stimulus-specific bias function
    output : results_stimulus_specific_bias_fixed_points.pickle
"""
ssb = StimulusSpecificBias()
ssb_fits = utils.load(f"{OUT_DIR}/results_stimulus_specific_bias.pickle")

ssb_fixed_points = dict(diverging={}, converging={})
for i_id, v_id in enumerate(id_list):
    ssb.weights = ssb_fits[v_id]
    critical_stimuli = utils.find_critical_stimuli(ssb, stim_list)
    ssb_fixed_points['diverging'][v_id] = critical_stimuli['diverging']
    ssb_fixed_points['converging'][v_id] = critical_stimuli['converging']

utils.save(ssb_fixed_points, f'{OUT_DIR}/results_stimulus_specific_bias_fixed_points.pickle')
print('results_stimulus_specific_bias_fixed_points.pickle saved')