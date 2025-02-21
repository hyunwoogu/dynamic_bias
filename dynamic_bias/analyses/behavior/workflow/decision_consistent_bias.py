"""decision-consistent bias workflow
    statistical testing of decision-consistent biases using resampling methods.
"""
import numpy as np
from dynamic_bias import utils
from dynamic_bias.analyses.behavior import DecisionConsistentBias

OUT_DIR = f"{utils.ORIGIN}/data/outputs/behavior"
utils.download_dataset("data/outputs/behavior")

# (i) load dataset
behavior = utils.load_behavior()
fixed_points = utils.load(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias_fixed_points.pickle")

# (ii) converging / diverging indices at population level
# 0 : others / 1 : diverging / 2 : converging
id_list = np.unique(behavior.ID)
idx_stim = np.zeros(len(behavior.stim), dtype=int) 
for i_id, v_id in enumerate( id_list ):
    idv_idx = (behavior.ID==v_id)
    idx_stim[idv_idx] = np.select(
        [np.isin(behavior[idv_idx].stim, fixed_points['diverging'][v_id]), 
         np.isin(behavior[idv_idx].stim, fixed_points['converging'][v_id])],
        [1, 2], default=0
    )

# (iii) construct data, and subset the data around fixed points
data = {
    'evidence' : -behavior.ref.to_numpy(),
    'choice'   : 2.-behavior.choice.to_numpy(),
    'error'    : behavior.error.to_numpy(),
    'cond'     : behavior.Timing.to_numpy(),
    'id'       : behavior.ID.to_numpy(),
}
n_trial = len(data['cond'])

## subset around fixed points
idx_fp = (idx_stim==1) | (idx_stim==2)
data_fp = {k:v[idx_fp] for k,v in data.items()} 
data_fp['group'] = idx_stim[idx_fp]
n_fp_trial = sum(idx_fp)


"""
1. Participants pre- and post-decision biases
    output : results_decision_consistent_bias.pickle
    Upper bounds used to ensure convergence of the optimization. Results similar without explicit bounds.
"""
res = {'d_bpre' : [], 'd_bpost' : []}
dcb = DecisionConsistentBias(sequential_inequality=['s'])
for i_id, v_id in enumerate( id_list ):
    idv_idx = (data['id']==v_id)
    data_sub = {k:v[idv_idx] for k,v in data.items()}

    dcb.fit(data_sub)
    b_components  = dcb.decompose(data_sub)

    res['d_bpre'].append( np.squeeze(np.diff(b_components['b_pre'])) )
    res['d_bpost'].append( np.squeeze(np.diff(b_components['b_post'])) )

res['d_bpre']  = np.array(res['d_bpre'])
res['d_bpost'] = np.array(res['d_bpost'])

utils.save(res, f'{OUT_DIR}/results_decision_consistent_bias.pickle')
print('results_decision_consistent_bias.pickle saved')


"""
2. Bootstrapping
    bootstrapping to compute SEMs of converging / diverging biases
    output : bootstrap_decision_consistent_bias_fixed_points.pickle
"""
N_BOOT = 10000
np.random.seed(2023)

# population-level decision-consistent biases (for nuisance parameter m)
dcb_div  = DecisionConsistentBias(sequential_inequality=['s'])
dcb_conv = DecisionConsistentBias(sequential_inequality=['s'])
dcb_div.fit(data_fp)
dcb_conv.fit(data_fp)

# bootstrapping
res_boot = {'d_bpre'  : {'diverging' : [], 'converging' : []}, 
            'd_bpost' : {'diverging' : [], 'converging' : []}}

for i_boot in range(N_BOOT):
    # resample trials to estimate sampling distribution
    idxb = utils.resample_indices(n_fp_trial, groups=data_fp['group'], replace=True)
    data_b = {k: v[idxb] for k, v in data_fp.items()}

    # compute pre- and post-decision biases, separately for groups
    data_b_div  = {k: v[data_b['group']==1] for k, v in data_b.items()}
    data_b_conv = {k: v[data_b['group']==2] for k, v in data_b.items()}

    dcb_div.fit(data_b_div, inherit_params=['m'])
    b_components_div  = dcb_div.decompose(data_b_div)
    dcb_conv.fit(data_b_conv, inherit_params=['m'])
    b_components_conv = dcb_conv.decompose(data_b_conv)

    res_boot['d_bpre']['diverging'].append( *np.diff(*b_components_div['b_pre']) ) 
    res_boot['d_bpre']['converging'].append( *np.diff(*b_components_conv['b_pre']) ) 
    res_boot['d_bpost']['diverging'].append( *np.diff(*b_components_div['b_post']) )
    res_boot['d_bpost']['converging'].append( *np.diff(*b_components_conv['b_post']) )
    
    if i_boot % 500 == 0:
        print(f'iteration {i_boot} done')

# save
res_boot['d_bpre']['diverging'] = np.array(res_boot['d_bpre']['diverging'])
res_boot['d_bpre']['converging'] = np.array(res_boot['d_bpre']['converging'])
res_boot['d_bpost']['diverging'] = np.array(res_boot['d_bpost']['diverging'])
res_boot['d_bpost']['converging'] = np.array(res_boot['d_bpost']['converging'])

utils.save(res_boot, f'{OUT_DIR}/bootstrap_decision_consistent_bias_fixed_points.pickle')
print('bootstrap_decision_consistent_bias_fixed_points.pickle saved')


"""
3. Permutation
    permutation to compute the null distribution of converging / diverging biases
    output : permutation_decision_consistent_bias_fixed_points.pickle
"""
N_PERM = 10000
np.random.seed(2023)

# population-level decision-consistent biases
dcb_div  = DecisionConsistentBias(sequential_inequality=['s'])
dcb_conv = DecisionConsistentBias(sequential_inequality=['s'])
dcb_div.fit(data_fp)
dcb_conv.fit(data_fp)

# permutation
res_perm = {'obs' : {'d_bpre' : {}, 'd_bpost' : {}},
            'null' : {'d_bpre'  : {'diverging' : [], 'converging' : []}, 
                      'd_bpost' : {'diverging' : [], 'converging' : []}}}

for i_perm in range(1 + N_PERM):
    # resample data
    if i_perm == 0:
        ## original observation 
        data_p = data_fp
    else:
        ## resample data to estimate sampling distribution
        idxp = utils.resample_indices(n_fp_trial, replace=False)
        data_p = {k: v[idxp] if k=='group' else v for k, v in data_fp.items()}

    # compute pre- and post-decision biases
    data_p_div  = {k: v[data_p['group']==1] for k, v in data_p.items()}
    data_p_conv = {k: v[data_p['group']==2] for k, v in data_p.items()}

    dcb_div.fit(data_p_div, inherit_params=['m'])
    b_components_div  = dcb_div.decompose(data_p_div)
    dcb_conv.fit(data_p_conv, inherit_params=['m'])
    b_components_conv = dcb_conv.decompose(data_p_conv)

    if i_perm == 0:
        res_perm['obs']['d_bpre']['diverging']  = np.diff(*b_components_div['b_pre'])
        res_perm['obs']['d_bpre']['converging'] = np.diff(*b_components_conv['b_pre'])
        res_perm['obs']['d_bpost']['diverging'] = np.diff(*b_components_div['b_post'])
        res_perm['obs']['d_bpost']['converging'] = np.diff(*b_components_conv['b_post'])
    else:
        res_perm['null']['d_bpre']['diverging'].append( *np.diff(*b_components_div['b_pre']) ) 
        res_perm['null']['d_bpre']['converging'].append( *np.diff(*b_components_conv['b_pre']) ) 
        res_perm['null']['d_bpost']['diverging'].append( *np.diff(*b_components_div['b_post']) )
        res_perm['null']['d_bpost']['converging'].append( *np.diff(*b_components_conv['b_post']) )
    
    if i_perm % 500 == 0:
        print(f'iteration {i_perm} done')

utils.save(res_perm, f'{OUT_DIR}/permutation_decision_consistent_bias_fixed_points.pickle')
print('permutation_decision_consistent_bias_fixed_points.pickle saved')