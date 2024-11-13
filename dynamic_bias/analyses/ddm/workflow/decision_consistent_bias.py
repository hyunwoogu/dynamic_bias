"""decision-consistent bias in drift-diffusion models
"""
import sys
import numpy as np
from scipy.interpolate import interp1d

from dynamic_bias import utils
from dynamic_bias.analyses.ddm import Euler

utils.download_dataset("data/outputs/behavior")
utils.download_dataset("models/ddm/reduced")
utils.download_dataset("models/ddm/full")

behavior = utils.load_behavior()
fixed_points = utils.load(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias_fixed_points.pickle")

# load models
from dynamic_bias.models.ddm import model
sys.modules['model'] = model
model_path = f'{utils.ORIGIN}/models/ddm/'
models_full = {}
models_rdcd = {}
id_list = np.unique(behavior.ID)
for v_id in id_list:
    fn_id = f'{v_id:04d}'
    fn = f'fitted_model_sub-{fn_id}.pkl'
    models_full[fn_id] = utils.load(model_path+'full/'+fn)    # drift-diffusion (full) model 
    models_rdcd[fn_id] = utils.load(model_path+'reduced/'+fn) # diffusion-only (reduced) model

# [1] construct lists
idx_stim = np.zeros(len(behavior.stim), dtype=int) 
for i_id, v_id in enumerate( id_list ):
    idv_idx = (behavior.ID==v_id)
    idx_stim[idv_idx] = np.select(
        [np.isin(behavior[idv_idx].stim, fixed_points['diverging'][v_id]), 
         np.isin(behavior[idv_idx].stim, fixed_points['converging'][v_id])],
        [1, 2], default=0
    )

# [2] construct data
datas = []
for v_id in id_list:
    idx_id   = (behavior.ID==v_id)
    idx_fxpt = (idx_stim==1) | (idx_stim==2) # diverging or converging stmiuli
    data = {
        't_dm'   : 6. * behavior.Timing[idx_id & idx_fxpt].to_numpy(),
        'stim'   : utils.ori2dir( behavior.stim[idx_id & idx_fxpt].to_numpy() ),
        'relref' : utils.ori2dir( behavior.ref[idx_id & idx_fxpt].to_numpy() ),
        'fxpt'   : idx_stim[idx_id & idx_fxpt],
    }
    datas.append(data)
datas_concat = {k: np.concatenate([v[k] for v in datas]) for k in datas[0].keys()}
n_trials = len(datas_concat['fxpt'])

# [3] run euler simulation
euler = Euler()
t_dm_e = utils.exp_onset_list()['early']
t_dm_l = utils.exp_onset_list()['late']



"""
1. Estimate decision-consistent bias in drift-diffusion models using Euler-Maruyama
    under normative unique stimuli for experiment
    output : results_decision_consistent_bias.pickle
"""
N_MC = 10000
np.random.seed(2023)
euler.params['n_trial'] = N_MC

name_par = {}
name_par['full']    = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_a', 'lam', 's']
name_par['reduced'] = ['w_E', 'w_P', 'w_D', 'w_r', 'w_a', 'lam', 's']

mc_dcb = {'full': {}, 'reduced': {}}
for i_models, (k_models, models) in enumerate({'full': models_full, 'reduced': models_rdcd}.items()):
    mc_dcb[k_models]['d_bpre']  = []
    mc_dcb[k_models]['d_bpost'] = []
    
    # compute each participant's decision-consistent bias under normative unique stimuli
    for i_m, (k_m, model) in enumerate(models.items()):

        # extract parameters and exclude reference attraction
        fitted_params = dict(zip(name_par[k_models], model.fitted_params.copy()))
        fitted_params['w_r'] = 0  # nuisance parameter
        
        # make K through interpolation
        kappa = np.concatenate([model.kappa,[model.kappa[0]]])
        kappa = interp1d(np.concatenate([model.m,[np.pi]]), kappa)

        # efficient coding
        p,F   = model.efficient_encoding(fitted_params['s'])
        m_    = np.concatenate([model.m, [np.pi]])
        F_fun = interp1d(m_, np.concatenate([F,[2*np.pi]]), kind='linear')
        F_inv = interp1d(np.concatenate([F,[2*np.pi]]), m_, kind='linear')

        # update parameters for euler simulation
        euler.params.update(fitted_params)
        euler.params['F'] = F_fun
        euler.params['F_inv'] = F_inv
        
        if k_models == 'full':
            euler.params['K'] = kappa
        elif k_models == 'reduced':
            euler.params['K'] = lambda x: 0
        
        # run euler simulation
        res = euler.run(return_data=True, returns=['decision', 'memory', 'production'])
        n_trial = len(res['data']['t_dm'])
        t_dm_idx = euler.get_time_index( res['data']['t_dm'] ) - 1 
        res['memory'] = res['memory'][ np.arange(n_trial), :, t_dm_idx ] # extract memory only at decision time

        # extract results
        idx_e = res['data']['t_dm']==t_dm_e
        idx_l = res['data']['t_dm']==t_dm_l
        idx_near = np.abs(res['data']['relref']) < euler.params['thres']

        # compute errors
        error_dm = utils.dir2ori( res['memory']  - res['data']['stim'][:,None] )
        error = utils.dir2ori( res['production'] - res['data']['stim'][:,None] )

        idx   = (idx_e & idx_near)
        cw_e  = utils.circmean( error[idx][ res['decision'][idx]== 1 ] )
        ccw_e = utils.circmean( error[idx][ res['decision'][idx]==-1 ] )
        b_pre_e = utils.circmean( error_dm[idx] * res['decision'][idx] )

        idx   = (idx_l & idx_near)
        cw_l  = utils.circmean( error[idx][ res['decision'][idx]== 1 ] )
        ccw_l = utils.circmean( error[idx][ res['decision'][idx]==-1 ] )
        b_pre_l = utils.circmean( error_dm[idx] * res['decision'][idx] )

        # compute biases
        b_e      = ( np.array(cw_e) - np.array(ccw_e) )/2.
        b_l      = ( np.array(cw_l) - np.array(ccw_l) )/2.
        b_post_e = b_e - b_pre_e
        b_post_l = b_l - b_pre_l
        mc_dcb[k_models]['d_bpre'].append( b_pre_l - b_pre_e )
        mc_dcb[k_models]['d_bpost'].append( b_post_l - b_post_e )

        print(f'{k_models} model {k_m} done')

utils.save(mc_dcb, f'{utils.ORIGIN}/data/outputs/ddm/results_decision_consistent_bias.pickle')
print('results_decision_consistent_bias.pickle saved')



"""
2. Boostrapping & permutation of converging / diverging biases using Euler-Maruyama
    output : results_decision_consistent_bias_fixed_points.pickle
"""
N_MC = 10000
np.random.seed(2023)
euler.params['n_trial'] = N_MC

## initialize
res_fxpt = {'boot' : {}, 'perm' : {}}
for k in ['full', 'reduced']:
    res_fxpt['boot'][k] = {'d_bpre': {}, 'd_bpost': {}}
    res_fxpt['perm'][k] = {'d_bpre': {}, 'd_bpost': {}}

name_par = {}
name_par['full']    = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_a', 'lam', 's']
name_par['reduced'] = ['w_E', 'w_P', 'w_D', 'w_r', 'w_a', 'lam', 's']

for i_models, (k_models, models) in enumerate({'full': models_full, 'reduced': models_rdcd}.items()):

    idx_fxpt_perms = [ utils.resample_indices( n_trials, replace=False ) for _ in range(N_MC) ]
    
    # euler simulation of experimental trials
    ress  = []
    for i_m, (_, model) in enumerate(models.items()):
        # extract parameters and exclude reference attraction
        fitted_params = dict(zip(name_par[k_models], model.fitted_params.copy()))
        fitted_params['w_r'] = 0  # nuisance parameter
        
        # make K through interpolation
        kappa = np.concatenate([model.kappa,[model.kappa[0]]])
        kappa = interp1d(np.concatenate([model.m,[np.pi]]), kappa)

        # efficient coding
        p,F   = model.efficient_encoding(fitted_params['s'])
        m_    = np.concatenate([model.m, [np.pi]])
        F_fun = interp1d(m_, np.concatenate([F,[2*np.pi]]), kind='linear')
        F_inv = interp1d(np.concatenate([F,[2*np.pi]]), m_, kind='linear')

        # update parameters for euler simulation
        euler.params.update(fitted_params)
        euler.params['F'] = F_fun
        euler.params['F_inv'] = F_inv
        
        if k_models == 'full':
            euler.params['K'] = kappa
        elif k_models == 'reduced':
            euler.params['K'] = lambda x: 0

        res = euler.run( data=datas[i_m], returns=['decision', 'memory', 'production'], verbose=True )
        n_trial = len( datas[i_m]['t_dm'] )
        t_dm_idx = euler.get_time_index( datas[i_m]['t_dm'] ) - 1 
        res['memory'] = res['memory'][ np.arange(n_trial), :, t_dm_idx ] # extract memory only at decision time
        ress.append( res )

    ress = {k: np.concatenate([v[k] for v in ress]) for k in ress[0].keys()}

    # save results 
    idx_e = datas_concat['t_dm']==t_dm_e
    idx_l = datas_concat['t_dm']==t_dm_l
    idx_near = np.abs(datas_concat['relref']) < euler.params['thres']
    error_dm = utils.dir2ori( ress['memory']  - datas_concat['stim'][:,None] )
    error = utils.dir2ori( ress['production'] - datas_concat['stim'][:,None] )

    for i_fxpt, n_fxpt in zip( [1,2], ['diverging', 'converging'] ):

        # [1] boostrapping (using the sampling from the drift-diffusion model)
        idx_fxpt = datas_concat['fxpt'] == i_fxpt
        idx   = idx_e & idx_near & idx_fxpt
        cw_e  = [ utils.circmean( error[idx,i] [ ress['decision'][idx,i]== 1 ] ) for i in range(N_MC) ]
        ccw_e = [ utils.circmean( error[idx,i] [ ress['decision'][idx,i]==-1 ] ) for i in range(N_MC)]
        b_pre_e = utils.circmean( error_dm[idx] * ress['decision'][idx], axis=0 )

        idx   = idx_l & idx_near & idx_fxpt
        cw_l  = [ utils.circmean( error[idx,i] [ ress['decision'][idx,i]== 1 ] ) for i in range(N_MC) ]
        ccw_l = [ utils.circmean( error[idx,i] [ ress['decision'][idx,i]==-1 ] ) for i in range(N_MC)]
        b_pre_l = utils.circmean( error_dm[idx] * ress['decision'][idx], axis=0 )

        b_e = np.array([(cw - ccw)/2. for cw, ccw in zip(cw_e, ccw_e)])
        b_l = np.array([(cw - ccw)/2. for cw, ccw in zip(cw_l, ccw_l)])
        b_post_e = b_e - b_pre_e
        b_post_l = b_l - b_pre_l

        res_fxpt['boot'][k_models]['d_bpre'][n_fxpt]  = b_pre_l  - b_pre_e
        res_fxpt['boot'][k_models]['d_bpost'][n_fxpt] = b_post_l - b_post_e

        # [2] permutation test
        idxs  = [ idx_e & idx_near & (datas_concat['fxpt'][i] == i_fxpt) for i in idx_fxpt_perms ]
        cw_e  = [ utils.circmean( error[idxs[i],i] [ ress['decision'][idxs[i],i]== 1 ] ) for i in range(N_MC) ]
        ccw_e = [ utils.circmean( error[idxs[i],i] [ ress['decision'][idxs[i],i]==-1 ] ) for i in range(N_MC) ]
        b_pre_e = np.array( [utils.circmean( error_dm[idxs[i],i]*ress['decision'][idxs[i],i] ) for i in range(N_MC)] )

        idxs  = [ idx_l & idx_near & (datas_concat['fxpt'][i] == i_fxpt) for i in idx_fxpt_perms ]
        cw_l  = [ utils.circmean( error[idxs[i],i] [ ress['decision'][idxs[i],i]== 1 ] ) for i in range(N_MC) ]
        ccw_l = [ utils.circmean( error[idxs[i],i] [ ress['decision'][idxs[i],i]==-1 ] ) for i in range(N_MC) ]
        b_pre_l = np.array( [utils.circmean( error_dm[idxs[i],i]*ress['decision'][idxs[i],i] ) for i in range(N_MC)] )

        b_e = np.array([(cw - ccw)/2. for cw, ccw in zip(cw_e, ccw_e)])
        b_l = np.array([(cw - ccw)/2. for cw, ccw in zip(cw_l, ccw_l)])
        b_post_e = b_e - b_pre_e
        b_post_l = b_l - b_pre_l

        res_fxpt['perm'][k_models]['d_bpre'][n_fxpt]  = b_pre_l  - b_pre_e
        res_fxpt['perm'][k_models]['d_bpost'][n_fxpt] = b_post_l - b_post_e

        print(f'{k_models} {n_fxpt} done')  
    
utils.save(res_fxpt, f'{utils.ORIGIN}/data/outputs/ddm/results_decision_consistent_bias_fixed_points.pickle')
print('results_decision_consistent_bias_fixed_points.pickle saved')