"""fMRI-DDM correspondence
"""
import sys
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from dynamic_bias import utils
from dynamic_bias.models.ddm import model
from dynamic_bias.analyses.ddm import Euler
from dynamic_bias.analyses.fmri import HemodynamicModel
from dynamic_bias.analyses.behavior import StimulusSpecificBias

# load dataset
utils.download_dataset("data/outputs/fmri")
behavior = utils.load_behavior()
bold_channel = utils.load_bold().transpose((0,2,1))
visual_params = utils.load(f"{utils.ORIGIN}/data/outputs/fmri/results_visual_drive.pickle")
ssb_fits = utils.load(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle")

# decode
label = utils.exp_stim_list(step=1.5)
bold_decode = utils.pop_vector_decoder( bold_channel, labels=label )

sys.modules['model'] = model
utils.download_dataset("models/ddm")
model_path = f'{utils.ORIGIN}/models/ddm/'

models_full = {}
models_rdcd = {}
id_list = np.unique(behavior.ID)
for v_id in id_list:
    fn_id = f'{v_id:04d}'
    fn = f'fitted_model_sub-{fn_id}.pkl'
    models_full[fn_id] = utils.load(model_path+'full/'+fn) # drift-diffusion (full) model 
    models_rdcd[fn_id] = utils.load(model_path+'reduced/'+fn) # diffusion-only (reduced) model

cutoff = 0.01 # cutoff for the proportion of DMs under a given trial
euler = Euler()
data = {
    'id'     : behavior.ID.to_numpy(),
    't_dm'   : 6. * behavior.Timing.to_numpy(),
    'stim'   : utils.ori2dir(behavior.stim.to_numpy()),
    'relref' : utils.ori2dir(behavior.ref.to_numpy()),
    'choice' : 2.-behavior.choice.to_numpy(),
}
idx_near = np.abs(data['relref']) < euler.params['thres']
data_near = {k: v[idx_near] for k, v in data.items()}
bold_near = bold_decode[:,idx_near]


"""
1. Drift-diffusion predictions based on hemodynamics
    output : results_stimulus_conditioned_pred.pickle
"""
ssb = StimulusSpecificBias()
stim_list = utils.exp_stim_list()
ssb_funs = {}
for v_id in id_list:
    ssb.weights = ssb_fits[v_id]
    ssb_funs[v_id] = ssb( stim_list )

hdt  = np.arange(4,28,step=2) # timing of interest
hdtv = np.arange(5,27,step=2)
hdm  = HemodynamicModel(onset_hemodynamic=hdtv[0], offset_hemodynamic=hdt[-1], visual_params=visual_params)

# compute biases by directly accessing the errors
readout_time = np.arange(28,step=0.5)
data_ddm = utils.exp_structure()
data_ddm = {
    'deg': {
        'stim': data_ddm['stimulus'],
        'ref' : utils.wrap(data_ddm['stimulus']+data_ddm['reference'], period=180.)
    },
    'relref'   :  data_ddm['reference'],
    'evidence' : -data_ddm['reference'],
    'delay'    :  data_ddm['timing']
}

labels = utils.dir2ori( np.linspace(-np.pi, np.pi, 96, endpoint = False) )
data_pred = {'stim'  : data_ddm['deg']['stim'],
             'ref'   : data_ddm['deg']['ref'],
             't_dms' : 6. * data_ddm['delay']}

trajs = {'ssb': {}, 'ssb_weights': {}}
for i_models, (nmodels, models) in enumerate(zip(['full', 'reduced'], [models_full, models_rdcd])):
    traj_pred = []
    for i_model, (n_model, model) in enumerate(models.items()):
        # load model
        fitted_params = model.fitted_params.copy()
        data_ddm = model.convert_unit(data_ddm)
        model.gen_mask(data_ddm)

        # compute densities
        _, Pm = model.forward(fitted_params, data_ddm, readout_time=readout_time)
        Pm = np.sum(Pm, axis=0)
        Pm = utils.pop_vector_decoder( Pm.transpose((0,2,1)), labels )

        # predictions
        traj_pred.append(hdm.predict(data_pred,
                                     t_underlying=readout_time,
                                     traj_underlying=utils.ori2dir( Pm.T )))

    traj_pred = utils.collapse(np.array(traj_pred).transpose((0,2,1)),
                               collapse_groups=[data_ddm['delay'], data_ddm['deg']['stim']],
                               collapse_func=utils.circmean,
                               collapse_kwargs={'axis':-1}, return_list=True)
    
    trajs['ssb'][nmodels] = {
        'early' : np.array( traj_pred )[0],
        'late'  : np.array( traj_pred )[1],
    }


for i_models, nmodels in enumerate( ['full', 'reduced'] ):    
    traj_ssb_e = utils.wrap(trajs['ssb'][nmodels]['early'] - stim_list[:,None,None], period=180.)
    traj_ssb_l = utils.wrap(trajs['ssb'][nmodels]['late']  - stim_list[:,None,None], period=180.)

    trajs['ssb_weights'][nmodels] = {'early': [], 'late': []}
    for i_id, v_id in enumerate(id_list):
        trajs['ssb_weights'][nmodels]['early'].append(
            LinearRegression(fit_intercept=False).fit(
                ssb_funs[v_id][:,None], traj_ssb_e[:,i_id],
            ).coef_[:,0]
        )
        trajs['ssb_weights'][nmodels]['late'].append(
            LinearRegression(fit_intercept=False).fit(
                ssb_funs[v_id][:,None], traj_ssb_l[:,i_id],
            ).coef_[:,0]
        )

utils.save(trajs, f"{utils.ORIGIN}/data/outputs/ddm/results_stimulus_conditioned_pred.pickle")
print('results_stimulus_conditioned_pred.pickle saved')



"""
2. Correspondence scores based on drift-diffusion simulation (Euler)
    output : results_correspondence_score.pickle
"""
hdt  = np.arange(4,28,step=2) # timing of interest
hdm  = HemodynamicModel(onset_hemodynamic=hdt[0], visual_params=visual_params)

N_MC  = 10000
t_dm_e = utils.exp_onset_list()['early']
t_dm_l = utils.exp_onset_list()['late']

np.random.seed(2023)
euler.params['T'] = 28
euler.params['n_trial'] = N_MC

name_par = {}
name_par['full']    = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_a', 'lam', 's']
name_par['reduced'] = ['w_E', 'w_P', 'w_D', 'w_r', 'w_a', 'lam', 's']
name_par['full_encoding'] = ['w_K', 'w_E', 'w_P', 'w_D', 'w_r', 'w_a', 'lam', 's']

scores = {'full': {}, 'reduced': {}, 'full_encoding': {}}
for i_models, (k_models, models) in enumerate({'full': models_full, 
                                               'reduced': models_rdcd, 
                                               'full_encoding': models_full}.items()):

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
        
        if k_models.startswith('full'):
            euler.params['K'] = kappa
        elif k_models.startswith('reduced'):
            euler.params['K'] = lambda x: 0
        
        # run euler simulation
        idx_id   = data_near['id']==int(k_m)
        data_sub = {k: v[idx_id] for k,v in data_near.items()}
        res_sub  = euler.run( data=data_sub, return_time=True )
        bold_sub = bold_near[:,idx_id].T

        # scores
        scores[k_models][int(k_m)] = utils.nan(sum(idx_id))
        for i_trial, (v_c, v_t, v_s, v_r, v_d, v_m, v_b) in enumerate(zip( 
            *([data_sub[k] for k in ['choice', 't_dm', 'stim', 'relref']] +
              [res_sub[k]  for k in ['decision', 'memory']] + [bold_sub]) 
        )): 
            idx_match = (v_d==(v_c*2-1))
            p_match = np.mean(idx_match)
            if p_match > cutoff:
                n_match = np.sum(idx_match)
                data_pred = {'stim'  : np.repeat(utils.dir2ori(v_s),n_match),
                            'ref'    : np.repeat(utils.dir2ori(v_s+v_r),n_match), # absolute reference
                            'choice' : np.repeat(v_c,n_match),
                            't_dms'  : np.repeat(v_t,n_match)}
                
                if k_models.endswith('encoding'):
                    v_m[idx_match] = np.outer(v_m[idx_match][:,0], np.ones_like(res_sub['time']))
            
                traj_pred = hdm.predict(data_pred,
                                        t_underlying=res_sub['time'],
                                        traj_underlying=v_m[idx_match])
                
                scores[k_models][int(k_m)][i_trial] = hdm.score(v_b, traj_pred, method='cosine', axis=None)

        print(f'{k_models} model {k_m} done')

utils.save(scores, f'{utils.ORIGIN}/data/outputs/fmri/results_correspondence_score.pickle')
print('results_correspondence_score.pickle saved')

