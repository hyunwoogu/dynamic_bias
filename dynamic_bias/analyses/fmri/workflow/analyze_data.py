"""analyze the biases from BOLD decoding using resampling methods
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from dynamic_bias import utils
from dynamic_bias.analyses.behavior import StimulusSpecificBias

# load data
behavior = utils.load_behavior()
bold_channel = utils.load_bold().transpose((0,2,1))
ssb_fits = utils.load(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle")

# construct data
data = {
    'id'     : behavior.ID.to_numpy(),
    'stim'   : behavior.stim.to_numpy(),
    'cond'   : behavior.Timing.to_numpy(),
    'choice' : behavior.choice.to_numpy(),
    'relref' : behavior.ref.to_numpy(),
}
id_list = np.unique(data['id'])
n_trial = len(data['cond'])

# construct data : near reference
idx_near = np.abs(data['relref']) <= 8
data_near = {k : v[idx_near] for k, v in data.items()}
n_trial_near = len(data_near['cond'])
pop_channel_near = bold_channel[:,idx_near]

# labels for trajectory for error trajectory computation (stimulus-shifted)
label = utils.exp_stim_list(step=1.5)
labels = {stim : np.roll(label, shift) for stim, shift in zip( utils.exp_stim_list(), range(0,120,5) )} 
labels = np.stack([labels[s] for s in data['stim']], axis=0)
labels_near = labels[idx_near]


"""
1. Bootstrapping stimulus-conditioned trajectory
    bootstrapping to compute SEM of stimulus-conditioned trajectories
        for each timing / stimulus condition, resample data with replacement
    output : bootstrap_stimulus_conditioned.pickle
"""
N_BOOT = 10000
np.random.seed(2023)

stim_traj_boot = {'early': [], 'late': []}
for i_boot in range(N_BOOT):
    idxb = utils.resample_indices(n_trial, groups=[data['cond'], data['stim']], replace=True)
    traj_boot = utils.collapse( 
        bold_channel[:,idxb], 
        collapse_groups=[data['cond'][idxb], data['stim'][idxb]],
        collapse_func=utils.pop_vector_decoder, 
        collapse_kwargs={'labels': label[np.newaxis]}, 
        collapse_axis=1, return_list=True
    )
    stim_traj_boot['early'].append(traj_boot[0])
    stim_traj_boot['late'].append(traj_boot[1])

    if i_boot % 500 == 0:
        print(f'iteration {i_boot} done')

stim_traj_boot['early'] = np.array(stim_traj_boot['early'])
stim_traj_boot['late']  = np.array(stim_traj_boot['late'])

utils.save(stim_traj_boot, f'{utils.ORIGIN}/data/outputs/fmri/bootstrap_stimulus_conditioned.pickle')
print('bootstrap_stimulus_conditioned.pickle saved')


"""
2. Bootstrapping decision-conditioned trajectory 
    bootstrapping to compute SEM of near-reference decision-conditioned trajectories
    output : bootstrap_decision_conditioned.pickle
"""
decision_traj_boot = {'early': [], 'late': []}

for i_boot in range(N_BOOT):
    idxb = utils.resample_indices(n_trial_near, groups=[data_near['cond'], data_near['choice']], replace=True)
    traj_boot = utils.collapse( 
        (pop_channel_near[:,idxb], labels_near[idxb]), 
        collapse_groups=[data_near['cond'][idxb], data_near['choice'][idxb]],
        collapse_func=utils.pop_vector_decoder, 
        collapse_axis=-2, return_list=True
    )
    decision_traj_boot['early'].append(traj_boot[0]) # [cw/ccw, n_time]
    decision_traj_boot['late'].append(traj_boot[1])  # [cw/ccw, n_time]

    if i_boot % 500 == 0:
        print(f'iteration {i_boot} done')

decision_traj_boot['early'] = np.array(decision_traj_boot['early'])
decision_traj_boot['late']  = np.array(decision_traj_boot['late'])

utils.save(decision_traj_boot, f'{utils.ORIGIN}/data/outputs/fmri/bootstrap_decision_conditioned.pickle')
print('bootstrap_decision_conditioned.pickle saved')


"""
3. Permutation test of stimulus-specific bias
    permutation to compute the null distribution of stimulus-specific bias
        collapse decoding trajectories for each (timing, subject, stimulus)
        for each (timing and subject), shuffle stimulus labels and compute errors in decoding
        each bias is compared to subject-specific ssb estimated from behavior data
    output : permutation_stimulus_specific_bias.pickle
"""
N_PERM = 10000
np.random.seed(2023)

# (i) stimulus-specific bias - behavior
ssb = StimulusSpecificBias()
ssb_funs = {}
for v_id in id_list:
    ## some participants do not have all the stimuli for each condition, so we need filtering
    ssb.weights = ssb_fits[v_id]
    ssb_funs[v_id] = {
        'early' : ssb( np.unique(data['stim'][(data['id']==v_id) & (data['cond']==1)]) ),
        'late'  : ssb( np.unique(data['stim'][(data['id']==v_id) & (data['cond']==2)]) ),
    }

# (i) stimulus-specific bias - decoding trajectories
traj_ssb, traj_groups = utils.collapse(
    (bold_channel, labels),
    collapse_groups=[data['cond'], data['id'], data['stim']],
    collapse_func=utils.pop_vector_decoder,
    collapse_axis=-2, return_labels=True
)

# (iii) stimulus-specific bias - construct data
data_ssb = {
    'cond' : np.array( [g[0] for g in traj_groups] ),
    'id'   : np.array( [g[1] for g in traj_groups] ),
    'stim' : np.array( [g[2] for g in traj_groups] )
}
n_trial_ssb = len(data_ssb['cond'])

# (iv) stack decoding trajectories
traj_ssb = np.array([traj_ssb[cond][id][stim] for cond, id, stim in 
                     zip(data_ssb['cond'], data_ssb['id'], data_ssb['stim'])])

# (v) permutation
ssb_weights_perm = { 'obs'  : {'early': [], 'late': []},  'null' : {'early': [], 'late': []} }
for i_perm in range(1 + N_PERM):
    if i_perm == 0:
        ## original observation
        idxp = np.arange(n_trial_ssb)
    else:
        ## shuffle stimulus labels 
        idxp = utils.resample_indices(n_trial_ssb, groups=[data_ssb['cond'], data_ssb['id']], replace=False)

    # run linear regression on collapsed data against ssb curve
    subj_weights = {'early': [], 'late': []}

    for i_id, v_id in enumerate(id_list):
        subj_weights['early'].append(
            LinearRegression(fit_intercept=False).fit(
                ssb_funs[v_id]['early'][:,None],
                traj_ssb[idxp][(data_ssb['id']==v_id) & (data_ssb['cond']==1)],
            ).coef_[:,0]
        )
        subj_weights['late'].append(
            LinearRegression(fit_intercept=False).fit(
                ssb_funs[v_id]['late'][:,None],
                traj_ssb[idxp][(data_ssb['id']==v_id) & (data_ssb['cond']==2)],
            ).coef_[:,0]
        )
    
    # save
    if i_perm == 0:
        ssb_weights_perm['obs']['early'] = np.array(subj_weights['early'])
        ssb_weights_perm['obs']['late']  = np.array(subj_weights['late'])
    else:
        ssb_weights_perm['null']['early'].append( np.mean(subj_weights['early'],axis=0) )
        ssb_weights_perm['null']['late'] .append( np.mean(subj_weights['late'],axis=0) )
    
    if i_perm % 500 == 0:
        print(f'iteration {i_perm} done')

# save
ssb_weights_perm['null']['early'] = np.array(ssb_weights_perm['null']['early'])
ssb_weights_perm['null']['late']  = np.array(ssb_weights_perm['null']['late'])
utils.save(ssb_weights_perm, f"{utils.ORIGIN}/data/outputs/fmri/permutation_stimulus_specific_bias.pickle") 
print('permutation_stimulus_specific_bias.pickle saved')


"""
4. Permutation test of decision-conditioned trajectory
    permutation to compute the null distribution of decision-conditioned trajectory
        for each timing condition (under near references), shuffle choice labels to obtain null
    output : permutation_decision_conditioned.pickle
"""
np.random.seed(2023)

idx_near = np.abs(data['relref']) <= 8
data_near = {k : v[idx_near] for k, v in data.items()}
n_trial_near = len(data_near['cond'])
bold_channel_near = bold_channel[:,idx_near]

# 
decision_traj_perm = {'obs' : {}, 'null' : {'early': [], 'late': []}}

for i_perm in range(1 + N_PERM):
    # resample data
    if i_perm == 0:
        ## original observation 
        idxp = np.arange(n_trial_near)
    else:
        ## shuffle choice labels
        idxp = utils.resample_indices(n_trial_near, groups=data_near['cond'], replace=False)
    
    traj_perm = utils.collapse(
        (bold_channel_near, labels_near),
        collapse_groups=[data_near['cond'], data_near['choice'][idxp]],
        collapse_func=utils.pop_vector_decoder, 
        collapse_axis=-2, return_list=True
    )
    if i_perm == 0:
        decision_traj_perm['obs']['early'] = np.array(traj_perm[0])
        decision_traj_perm['obs']['late']  = np.array(traj_perm[1])
    else:
        decision_traj_perm['null']['early'].append(traj_perm[0])
        decision_traj_perm['null']['late'].append(traj_perm[1])

    if i_perm % 500 == 0:
        print(f'iteration {i_perm} done')

decision_traj_perm['null']['early'] = np.array(decision_traj_perm['null']['early'])
decision_traj_perm['null']['late']  = np.array(decision_traj_perm['null']['late'])

utils.save(decision_traj_perm, f'{utils.ORIGIN}/data/outputs/fmri/permutation_decision_conditioned.pickle')
print('permutation_decision_conditioned.pickle saved')