"""analyze data from trained RNNs
"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import iqr

from dynamic_bias import utils
from dynamic_bias.analyses.behavior import PsychometricFunction
from dynamic_bias.analyses.behavior import StimulusSpecificBias

OUT_DIR = f"{utils.ORIGIN}/data/outputs/rnn"

# load aggregated RNN behaviors
utils.download_dataset("data/outputs/rnn")
models = ['heterogeneous', 'heterogeneous_emonly', 'heterogeneous_d2e_ablation']
datas = dict()
for v_model in models:
    datas[v_model] = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/results_{v_model}.pickle")

# behavior info
id_list   = np.arange(50)
stim_list = utils.exp_stim_list()
evi_list  = np.array([-22.5, -15, -7.5, 0, +7.5, +15, +22.5])
evi_near  = np.array([-7.5, 0, +7.5])
thres     = 8 # threshold for near-reference
ref_list  = np.array([-22.5, -7.5, 0, 7.5, 22.5])
ref_near  = np.array([-7.5, 0, +7.5])

"""
1. Growth of stimulus-specific and decision-consistent biases
    outputs :
        results_stimulus_specific_bias.pickle
        results_stimulus_specific_bias_weight.pickle
        results_stimulus_specific_bias_fixed_points.pickle
        results_decision_conditioned.pickle
"""
ssb = StimulusSpecificBias()
ssb_fits = {}
ssb_weights = {}
ssb_fixed_points = dict(diverging={}, converging={})
res_dcb = { 'refwise': {}, 'combined': {} }
for v_delay in ['early', 'late']:
    res_dcb['refwise'][v_delay] = {k:utils.nan((len(id_list),len(evi_list))) for k in ['cw','ccw']}
    res_dcb['combined'][v_delay] = {k:utils.nan((len(id_list))) for k in ['cw','ccw']}

for v_id in id_list:
    sub_idx =  datas['heterogeneous']['ID'] == v_id
    sub_tim =  datas['heterogeneous']['timing'][sub_idx]
    sub_stm =  datas['heterogeneous']['stimulus'][sub_idx]
    sub_etm =  datas['heterogeneous']['estim'][sub_idx]
    sub_evi = -datas['heterogeneous']['relref'][sub_idx]
    sub_chc =  datas['heterogeneous']['choice'][sub_idx]
    sub_err = utils.wrap( sub_etm-sub_stm, period=180.)

    # [1] Fitting stimulus-specific bias function
    ssb.fit( [utils.circmean(sub_err[sub_stm==s]) for s in stim_list] )
    ssb_fits[v_id] = ssb.weights

    # [2] Fitting stimulus-specific bias amplitudes at DM timing
    data = {
        'evidence' : sub_evi,
        'choice'   : sub_chc,
        'cond'     : sub_tim,
        'stim'     : sub_stm,
    }
    pse_fun = lambda s: -ssb(s)
    psi = PsychometricFunction(
        link='gaussian_pse',
        pse=pse_fun,
        sequential_inequality=['s']
    )
    psi.fit(data, constrain_params=['lam'])
    ssb_weights[v_id] = dict(zip(['E','L'], psi.fitted_params[:,0]))

    # [3] Finding fixed points of the stimulus-specific bias function
    critical_stimuli = utils.find_critical_stimuli(ssb, stim_list)
    ssb_fixed_points['diverging'][v_id] = critical_stimuli['diverging']
    ssb_fixed_points['converging'][v_id] = critical_stimuli['converging']

    # [4] Finding decision-consistent bias
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i_delay, v_delay in enumerate(['early', 'late']):
            # reference-wise
            res_dcb['refwise'][v_delay]['cw'][v_id] = np.array([
                utils.circmean(sub_err[(sub_tim==v_delay) & (sub_chc==1) & (sub_evi==evi)]) for evi in evi_list
            ])
            res_dcb['refwise'][v_delay]['ccw'][v_id] = np.array([
                utils.circmean(sub_err[(sub_tim==v_delay) & (sub_chc==0) & (sub_evi==evi)]) for evi in evi_list
            ])
            # combined
            res_dcb['combined'][v_delay]['cw'][v_id]  = utils.circmean(sub_err[(sub_tim==v_delay) & (sub_chc==1) & np.isin(sub_evi, evi_near)])
            res_dcb['combined'][v_delay]['ccw'][v_id] = utils.circmean(sub_err[(sub_tim==v_delay) & (sub_chc==0) & np.isin(sub_evi, evi_near)])

utils.save(ssb_fits, f'{OUT_DIR}/results_stimulus_specific_bias.pickle')
utils.save(ssb_weights, f'{OUT_DIR}/results_stimulus_specific_bias_weight.pickle')
utils.save(ssb_fixed_points, f'{OUT_DIR}/results_stimulus_specific_bias_fixed_points.pickle')
utils.save(res_dcb, f'{OUT_DIR}/results_decision_conditioned.pickle')
print('model predictions saved')


"""
2. RNN-wise stimulus-specific bias trajectory
    collapse trajectories for each (timing, network, stimulus)
    each bias is compared to subject-specific ssb estimated from behavior data
    output : stimulus_specific_bias_weight_traj.pickle
"""
ssb_fits  = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/results_stimulus_specific_bias.pickle")

# (i) stimulus-specific bias - behavior
ssb = StimulusSpecificBias()
ssb_funs = {}
for v_id in id_list:
    ## some participants do not have all the stimuli for each condition, so we need filtering
    ssb.weights = ssb_fits[v_id]
    ssb_funs[v_id] = ssb( stim_list )

# (ii) stimulus-specific bias - trajectories
traj_ssb = utils.collapse(
    datas['heterogeneous']['em_readout'], 
    collapse_groups=[datas['heterogeneous']['timing'], datas['heterogeneous']['ID'], datas['heterogeneous']['stimulus']],
    collapse_func=utils.circmean,
    collapse_kwargs={'axis':-1},
    return_list=True,
)
traj_ssb = utils.wrap(np.array(traj_ssb) - stim_list[:,None], period=180.)

# (iii) run linear regression on collapsed data against behavior ssb curve
ssb_weights_traj = {'early': [], 'late': []}

for v_id in id_list:
    ssb_weights_traj['early'].append(
        LinearRegression(fit_intercept=False).fit(
            ssb_funs[v_id][:,None], traj_ssb[0,v_id],
        ).coef_[:,0]
    )
    ssb_weights_traj['late'].append(
        LinearRegression(fit_intercept=False).fit(
            ssb_funs[v_id][:,None], traj_ssb[1,v_id],
        ).coef_[:,0]
    )
utils.save(ssb_weights_traj, f"{utils.ORIGIN}/data/outputs/rnn/results_stimulus_specific_bias_weight_traj.pickle") 
print('results_stimulus_specific_bias_weight_traj.pickle saved')


"""
3. Near-reference variability in RNNs
    output : results_near_reference_variability.pickle
"""
nrv = {'refwise'  : {k: {} for k in models}, 
       'combined' : {k: {} for k in models}}

for v_model in models:
    err = utils.wrap(datas[v_model]['estim']-datas[v_model]['stimulus'], period=180.)
    ids = datas[v_model]['ID']
    rfs = datas[v_model]['relref']
    
    nrv['refwise'][v_model] = {}
    for v_r in ref_list:
        nrv['refwise'][v_model][v_r] = []
        for v_id in id_list:
            nrv['refwise'][v_model][v_r].append( iqr( err[(ids == v_id) & (rfs == v_r)] ) )

    nrv['combined'][v_model] = {'near': [], 'far': []}
    for v_id in id_list:
        nrv['combined'][v_model]['near'].append( iqr( err[(ids == v_id) & (np.abs(rfs)<thres)] ) )
        nrv['combined'][v_model]['far'].append(  iqr( err[(ids == v_id) & (np.abs(rfs)>thres)] ) )

utils.save(nrv, f"{utils.ORIGIN}/data/outputs/rnn/results_near_reference_variability.pickle") 
print('results_near_reference_variability.pickle saved')


"""
4. Bootstrapping stimulus-conditioned trajectory
    bootstrapping to compute SEM of stimulus-conditioned trajectories
        for each timing / stimulus condition, resample data with replacement
    output : bootstrap_stimulus_conditioned.pickle
"""
N_BOOT = 10000
n_trial = len(datas['heterogeneous']['timing'])

stim_traj_boot = {'early': [], 'late': []}
for i_boot in range(N_BOOT):
    idxb = utils.resample_indices(n_trial, groups=[datas['heterogeneous']['timing'], datas['heterogeneous']['stimulus']], replace=True)
    traj_boot = utils.collapse(
        datas['heterogeneous']['em_readout'][:,idxb], 
        collapse_groups=[datas['heterogeneous']['timing'][idxb], datas['heterogeneous']['stimulus'][idxb]],
        collapse_func=utils.circmean,
        collapse_kwargs={'axis':-1},
        return_list=True,
    )
    stim_traj_boot['early'].append(traj_boot[0])
    stim_traj_boot['late'].append(traj_boot[1])
    
    if i_boot % 500 == 0:
        print(f'iteration {i_boot} done')

# for compact storage, save means and SEMs only
stim_traj_boot['early'] = utils.wrap(np.array(stim_traj_boot['early']) - utils.exp_stim_list()[:, None], period=180. )
stim_traj_boot['late']  = utils.wrap(np.array(stim_traj_boot['late'])  - utils.exp_stim_list()[:, None], period=180. )

earlym, earlys = utils.meanstats( stim_traj_boot['early'], axis=0, sd=True )
latem,  lates  = utils.meanstats( stim_traj_boot['late'],  axis=0, sd=True )
earlym = utils.wrap(earlym + utils.exp_stim_list()[:, None], period=180.)
latem  = utils.wrap(latem  + utils.exp_stim_list()[:, None], period=180.)

stim_traj_boot = {
    'early' : {'m' : earlym, 's' : earlys},
    'late'  : {'m' : latem,  's' : lates},
}
utils.save(stim_traj_boot, f"{utils.ORIGIN}/data/outputs/rnn/bootstrap_stimulus_conditioned.pickle")
print('bootstrap_stimulus_conditioned.pickle saved')


"""
5. Bootstrapping decision-conditioned trajectory 
    bootstrapping to compute SEM of near-reference decision-conditioned trajectories
    output : bootstrap_decision_conditioned.pickle
"""
N_BOOT = 10000
n_trial = len(datas['heterogeneous']['timing'])

idx_near = np.isin(datas['heterogeneous']['relref'], ref_near)
data_near = {
    'cond'   : datas['heterogeneous']['timing'][idx_near],
    'choice' : datas['heterogeneous']['choice'][idx_near],
}
n_trial_near = len(data_near['cond'])
stim_near = datas['heterogeneous']['stimulus'][idx_near]
em_readout_near = datas['heterogeneous']['em_readout'][:,idx_near]
er_readout_near = utils.wrap( em_readout_near - stim_near, period=180. )

decision_traj_boot = {'early': [], 'late': []}

for i_boot in range(N_BOOT):
    idxb = utils.resample_indices(n_trial_near, groups=[data_near['cond'], data_near['choice']], replace=True)
    traj_boot = utils.collapse( 
        er_readout_near[:,idxb], 
        collapse_groups=[data_near['cond'][idxb], data_near['choice'][idxb]],
        collapse_func=utils.circmean,
        collapse_kwargs={'axis':-1},
        return_list=True,
    )
    decision_traj_boot['early'].append(traj_boot[0]) # [ccw/cw, n_time]
    decision_traj_boot['late'].append(traj_boot[1])  # [ccw/cw, n_time]

    if i_boot % 500 == 0:
        print(f'iteration {i_boot} done')

# for compact storage, save means and SEMs only
earlym, earlys = utils.meanstats( decision_traj_boot['early'], axis=0, sd=True )
latem,  lates  = utils.meanstats( decision_traj_boot['late'],  axis=0, sd=True )

decision_traj_boot = {
    'early' : {'m' : earlym, 's' : earlys},
    'late'  : {'m' : latem,  's' : lates},
}
utils.save(decision_traj_boot, f'{utils.ORIGIN}/data/outputs/rnn/bootstrap_decision_conditioned.pickle')
print('bootstrap_decision_conditioned.pickle saved')