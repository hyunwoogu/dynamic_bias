"""estimation of visual-drive-related parameters
"""
import numpy as np
from dynamic_bias import utils
from dynamic_bias.analyses.fmri import HemodynamicModel

# load data
behavior = utils.load_behavior()
bold_channel = utils.load_bold()

# construct data
data = {
    'stim'   : behavior.stim.to_numpy(),
    'relref' : behavior.ref.to_numpy(),
    'cond'   : behavior.Timing.to_numpy(),
}
data['near'] = np.abs(data['relref']) <= 8
n_trial = len(data['cond'])

"""
1. Fitting visual drive parameters
    output : results_visual_drive.pickle
"""
# 1. labels for trajectory for error trajectory computation 
    # (stimulus-shifted / relative-reference-sign-flipped)
label = utils.exp_stim_list(step=1.5)
label = {stim : {1: np.roll(label, shift), 0: utils.reflect(np.roll(label, shift), shift)} 
         for stim, shift in zip( utils.exp_stim_list(), range(0,120,5) )}
labels = np.stack([label[s][r>=0] for s,r in zip(data['stim'], data['relref'])], axis=-1)

# 2. collapse data for timing conditions
idx_far = np.abs(data['relref']) > 8
traj = utils.collapse(
    (bold_channel[...,idx_far], labels[..., idx_far]), 
    collapse_groups=data['cond'][idx_far],
    collapse_func=utils.pop_vector_decoder, return_list=True
)
traj = np.array(traj)

# 3. fit the hemodynamic model
hdt = np.arange(4,28,step=2) # timing of interest
hdm = HemodynamicModel(onset_hemodynamic=hdt[0])

data_fit = {'stim'  : np.array([0,0]),
            'ref'   : np.array([21.,21.]),
            't_dms' : np.array([6,12]),}
hdm.fit(data = data_fit, traj = traj)
utils.save(hdm.fitted_params['visual'], f'{utils.ORIGIN}/data/outputs/fmri/results_visual_drive.pickle')
print('results_visual_drive.pickle saved')


"""
2. Bootstrapping visual drive parameters
    bootstrapping to compute SEM of parameters and r-squared
    output : bootrap_visual_drive.pickle
"""
N_BOOT = 10000
np.random.seed(2023)

# hemodynamic model
hdt = np.arange(4,28,step=2) # timing of interest
hdm = HemodynamicModel(onset_hemodynamic=hdt[0])

data_fit = {'stim'  : np.array([0,0]),
            'ref'   : np.array([21.,21.]),
            't_dms' : np.array([6,12]),}

# labels for trajectory for error trajectory computation 
label = utils.exp_stim_list(step=1.5)
label = {stim : {1: np.roll(label, shift), 0: utils.reflect(np.roll(label, shift), shift)} 
         for stim, shift in zip( utils.exp_stim_list(), range(0,120,5) )}
labels = np.stack([label[s][r>=0] for s,r in zip(data['stim'], data['relref'])], axis=-1)

#
res_boot = {'traj': [], 'beta' : [], 'rsq' : []}

i_boot = 0
while i_boot < N_BOOT:
    # resample data
    idxb = utils.resample_indices(n_trial, groups=[data['cond'], data['near']], replace=True)
    pop_channel_b = bold_channel[...,idxb]
    labels_b      = labels[...,idxb]
    idx_far_b     = ~data['near'][idxb]
    cond_b        =  data['cond'][idxb]

    # compute trajectories
    traj_b = utils.collapse( 
        (pop_channel_b[...,idx_far_b], labels_b[..., idx_far_b]), 
        collapse_groups=cond_b[idx_far_b],
        collapse_func=utils.pop_vector_decoder, return_list=True
    )

    # fit
    success = hdm.fit(data_fit, np.array(traj_b), return_success=True)
    traj_b_pred = hdm.predict(data_fit, model='visual')

    if not success:
        print(f'failed at iteration {i_boot}')
        continue
    
    # store
    res_boot['traj'].append(traj_b)
    res_boot['beta'].append(list(hdm.fitted_params['visual'].values()))
    res_boot['rsq'].append(hdm.score(traj_b, traj_b_pred, method='r_squared', axis=-1))
    i_boot += 1
    
    if i_boot % 500 == 0:
        print(f'iteration {i_boot} done')

res_boot['traj'] = np.array(res_boot['traj'])
res_boot['beta'] = np.array(res_boot['beta'])
res_boot['rsq']  = np.array(res_boot['rsq'])

utils.save(res_boot, f'{utils.ORIGIN}/data/outputs/fmri/bootstrap_visual_drive.pickle')
print('bootstrap_visual_drive.pickle saved')
