"""fMRI: estimation of decision-consistent bias function in bold responses
"""
import numpy as np
from dynamic_bias import utils
from dynamic_bias.analyses.fmri import HemodynamicModel

# load dataset
utils.download_dataset("data/outputs/behavior")
utils.download_dataset("data/outputs/fmri")

behavior = utils.load_behavior()
bold_channel  = utils.load_bold().transpose((0,2,1))
visual_params = utils.load(f"{utils.ORIGIN}/data/outputs/fmri/results_visual_drive.pickle")
fixed_points  = utils.load(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias_fixed_points.pickle")

# [1] construct data
data = {
    'id'     : behavior.ID.to_numpy(),
    'stim'   : behavior.stim.to_numpy(),
    'cond'   : behavior.Timing.to_numpy(),
    'choice' : behavior.choice.to_numpy(),
    'relref' : behavior.ref.to_numpy(),
}
id_list = np.unique(data['id'])
n_trial = len(data['cond'])

# [2] construct data : near reference
idx_near = np.abs(data['relref']) <= 8
data_near = {k : v[idx_near] for k, v in data.items()}

label = utils.exp_stim_list(step=1.5)
bold_channel_error = utils.pop_vector_decoder( bold_channel, labels=label )
bold_channel_error = utils.wrap(bold_channel_error - data['stim'], period=180)
bold_channel_error = bold_channel_error * (3-2*data['choice']) # sign-flip for choice

patterns_bold = utils.collapse(bold_channel_error[:,idx_near],
                               collapse_groups=[data_near['id'], data_near['cond']],
                               collapse_kwargs=dict(axis=-1))

## around fixed points
idx_stim = np.zeros(len(behavior.stim), dtype=int) 
for i_id, v_id in enumerate( id_list ):
    idv_idx = (behavior.ID==v_id)
    idx_stim[idv_idx] = np.select(
        [np.isin(behavior[idv_idx].stim, fixed_points['diverging'][v_id]),
         np.isin(behavior[idv_idx].stim, fixed_points['converging'][v_id])],
        [1, 2], default=0
    )
idx_near_fp = ((idx_stim==1) | (idx_stim==2)) & idx_near
data_near_fp = {k:v[idx_near_fp] for k,v in data.items()}
data_near_fp['group'] = idx_stim[idx_near_fp] 
bold_channel_error_near_fp = bold_channel_error[:,idx_near_fp]
n_near_trial = sum(idx_near)
n_near_fp_trial = sum(idx_near_fp)

# [3] construct variables related to fitting hemodynamic model
data_fit = {'stim'   : np.zeros(2),
            'ref'    : np.zeros(2),
            'choice' : np.ones(2),
            't_dms'  : np.array([6, 12]),}
hdt  = np.arange(4,28,step=2)
hdtv = np.arange(5,27,step=2)


"""
1. Estimate decision-consistent bias from BOLD data using hemodynamic model
    output : results_decision_consistent_bias.pickle
"""
hdm  = HemodynamicModel(onset_hemodynamic=hdtv[0], offset_hemodynamic=hdt[-1], visual_params=visual_params)
dcbs = {'d_bpre': [], 'd_bpost': [], 'b': {'early': [], 'late': []}}
for _, v_id in enumerate(id_list):
    traj_fit = np.stack([patterns_bold[v_id][1], patterns_bold[v_id][2]],axis=0)
    hdm.fit(data = data_fit, traj = hdm.interp(hdtv, hdt, traj_fit), model='piecewise_linear')
    b_decompose = hdm.decompose_b(return_b=True)
    dcbs['d_bpre'].append( *np.diff(b_decompose['b_pre']) )
    dcbs['d_bpost'].append( *np.diff(b_decompose['b_post']) )
    dcbs['b']['early'].append( b_decompose['b'][0] )
    dcbs['b']['late'].append( b_decompose['b'][1] )

utils.save(dcbs, f"{utils.ORIGIN}/data/outputs/fmri/results_decision_consistent_bias.pickle")
print('results_decision_consistent_bias.pickle saved')


"""
2. Bootstrapping
    bootstrapping to compute SEMs of converging / diverging biases
    output : bootstrap_decision_consistent_bias_fixed_points.pickle
"""
# bootstrapping
N_BOOT = 10000
np.random.seed(2023)
res_boot = {'d_bpre'  : {'diverging' : [], 'converging' : []}, 
            'd_bpost' : {'diverging' : [], 'converging' : []}}

hdm_div  = HemodynamicModel(onset_hemodynamic=hdtv[0], offset_hemodynamic=hdt[-1], visual_params=visual_params)
hdm_conv = HemodynamicModel(onset_hemodynamic=hdtv[0], offset_hemodynamic=hdt[-1], visual_params=visual_params)

for i_boot in range(N_BOOT):
    # resample data
    idxb = utils.resample_indices(n_near_fp_trial, groups=data_near_fp['group'], replace=True)
    data_b = {k: v[idxb] for k, v in data_near_fp.items()}
    bold_b = bold_channel_error_near_fp[:,idxb]

    # compute biases, separately for groups
    data_b_div  = {k: v[data_b['group']==1] for k, v in data_b.items()}
    data_b_conv = {k: v[data_b['group']==2] for k, v in data_b.items()}

    patterns_bold_div  = utils.collapse(bold_b[:,data_b['group']==1],
                                        collapse_groups=data_b_div['cond'],
                                        collapse_kwargs=dict(axis=-1),
                                        return_list=True)
    patterns_bold_conv = utils.collapse(bold_b[:,data_b['group']==2],
                                        collapse_groups=data_b_conv['cond'],
                                        collapse_kwargs=dict(axis=-1),
                                        return_list=True)
    hdm_div.fit(data = data_fit, traj = hdm_div.interp(hdtv, hdt, patterns_bold_div), model='piecewise_linear')
    hdm_conv.fit(data = data_fit, traj = hdm_conv.interp(hdtv, hdt, patterns_bold_conv), model='piecewise_linear')
    b_decomp_div  = hdm_div.decompose_b()
    b_decomp_conv = hdm_conv.decompose_b()
    for b_decomp, n_model in zip([b_decomp_div, b_decomp_conv], ['diverging', 'converging']):
        res_boot['d_bpre'][n_model].append( *np.diff(b_decomp['b_pre']) )
        res_boot['d_bpost'][n_model].append( *np.diff(b_decomp['b_post']) )

    if i_boot % 500 == 0:
        print(f'iteration {i_boot} done')

utils.save(res_boot, f"{utils.ORIGIN}/data/outputs/fmri/bootstrap_decision_consistent_bias_fixed_points.pickle")
print('bootstrap_decision_consistent_bias_fixed_points.pickle saved')


"""
3. Permutation
    permutation to compute the null distribution of converging / diverging biases
    output : permutation_decision_consistent_bias_fixed_points.pickle
"""
N_PERM = 10000
np.random.seed(2023)
res_perm = {'obs' : {'d_bpre' : {}, 'd_bpost' : {}},
            'null' : {'d_bpre'  : {'diverging' : [], 'converging' : []}, 
                      'd_bpost' : {'diverging' : [], 'converging' : []}}}

hdm_div  = HemodynamicModel(onset_hemodynamic=hdtv[0], offset_hemodynamic=hdt[-1], visual_params=visual_params)
hdm_conv = HemodynamicModel(onset_hemodynamic=hdtv[0], offset_hemodynamic=hdt[-1], visual_params=visual_params)

for i_perm in range(1 + N_PERM):
    # resample data
    if i_perm == 0:
        ## original observation 
        data_p = data_near_fp
    else:
        ## resample data
        idxp = utils.resample_indices(n_near_fp_trial, replace=False)
        data_p = {k: v[idxp] if k=='group' else v for k, v in data_near_fp.items()}

    # compute biases, separately for groups
    data_p_div  = {k: v[data_p['group']==1] for k, v in data_p.items()}
    data_p_conv = {k: v[data_p['group']==2] for k, v in data_p.items()}
    patterns_bold_div  = utils.collapse(bold_channel_error_near_fp[:,data_p['group']==1],
                                        collapse_groups=data_p_div['cond'],
                                        collapse_kwargs=dict(axis=-1),
                                        return_list=True)
    patterns_bold_conv = utils.collapse(bold_channel_error_near_fp[:,data_p['group']==2],
                                        collapse_groups=data_p_conv['cond'],
                                        collapse_kwargs=dict(axis=-1),
                                        return_list=True)
    hdm_div.fit(data = data_fit, traj = hdm_div.interp(hdtv, hdt, patterns_bold_div), model='piecewise_linear')
    hdm_conv.fit(data = data_fit, traj = hdm_conv.interp(hdtv, hdt, patterns_bold_conv), model='piecewise_linear')
    b_decomp_div  = hdm_div.decompose_b()
    b_decomp_conv = hdm_conv.decompose_b()

    for b_decomp, n_model in zip([b_decomp_div, b_decomp_conv], ['diverging', 'converging']):
        if i_perm == 0:
            res_perm['obs']['d_bpre'][n_model] = np.diff(b_decomp['b_pre'])
            res_perm['obs']['d_bpost'][n_model] = np.diff(b_decomp['b_post'])
        else:
            res_perm['null']['d_bpre'][n_model].append( *np.diff(b_decomp['b_pre']) )
            res_perm['null']['d_bpost'][n_model].append( *np.diff(b_decomp['b_post']) )

    if i_perm % 500 == 0:
        print(f'iteration {i_perm} done')

utils.save(res_perm, f"{utils.ORIGIN}/data/outputs/fmri/permutation_decision_consistent_bias_fixed_points.pickle")
print('permutation_decision_consistent_bias_fixed_points.pickle saved')