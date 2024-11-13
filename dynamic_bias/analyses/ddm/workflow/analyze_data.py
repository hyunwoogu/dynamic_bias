"""analyze the biases from drift-diffusion models
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

from dynamic_bias import utils
from dynamic_bias.analyses.behavior import StimulusSpecificBias

utils.download_dataset("data/outputs/behavior")
utils.download_dataset("models/ddm/reduced")
utils.download_dataset("models/ddm/full")
behavior = utils.load_behavior()
ssb_fits = utils.load(f'{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle')

# load models
from dynamic_bias.models.ddm import model
sys.modules['model'] = model

model_path = f'{utils.ORIGIN}/models/ddm/'
models_full = {}
models_rdcd = {}
models_null = {}

id_list = np.unique(behavior.ID)
for v_id in id_list:
    fn_id = f'{v_id:04d}'
    fn = f'fitted_model_sub-{fn_id}.pkl'
    models_full[fn_id] = utils.load(model_path+'full/'+fn)    # drift-diffusion (full) model 
    models_rdcd[fn_id] = utils.load(model_path+'reduced/'+fn) # diffusion-only (reduced) model
    models_null[fn_id] = utils.load(model_path+'null/'+fn)    # encoding-only (null) model


"""
1. Model predictions of growth of stimulus-specific and decision-consistent biases
    under normative unique stimuli for experiment
    outputs :
        results_stimulus_specific_bias.pickle
        results_stimulus_specific_bias_weight.pickle
        results_decision_conditioned.pickle
"""
# compute biases by directly accessing the errors
ssb = StimulusSpecificBias()
readout_time   = utils.exp_onset_list(return_dict=False) # decision times for early and late conditions
dm_ssb         = utils.nan([3,50,2,24])  # [ddm/dom, n_IDs, n_delays, n_stims]
dm_ssb_weights = utils.nan([3,50,2])     # [ddm/dom, n_IDs, n_delays]
em_dcb         = utils.nan([3,50,2,5,2]) # [ddm/dom, n_IDs, n_delays, n_evidence, cw/ccw]
em_dcb_near    = utils.nan([3,50,2,2])   # [ddm/dom, n_IDs, n_delays, cw/ccw]

# 
collapse_args = {'collapse_func':utils.pop_vector_decoder, 'return_list':True,
                 'collapse_kwargs':{'label_unit':'radian', 'label_data':'direction'}}

data = utils.exp_structure()
data = {
    'deg': {
        'stim': data['stimulus'],
        'ref' : utils.wrap(data['stimulus']+data['reference'], period=180.)
    },
    'relref'   :  data['reference'],
    'evidence' : -data['reference'],
    'delay'    :  data['timing']
}

#
for i_models, models in enumerate([models_full, models_rdcd, models_null]):
    for i_model, (n_model, model) in enumerate(models.items()):
        # load model
        fitted_params = model.fitted_params.copy()
        data = model.convert_unit(data)
        model.gen_mask(data)

        # compute densities
        P, Pm = model.forward(fitted_params, data, readout_time=readout_time)
        P  = np.stack(P, axis=0)  # [cw/ccw, n_space, n_trials]
        Pm = np.stack(Pm, axis=0) # [cw/ccw, n_timing, n_space, n_trials]
        Pm = np.concatenate([Pm[:,0][...,data['delay']==1],
                             Pm[:,1][...,data['delay']==2]], axis=-1) # [cw/ccw, n_space, n_trials]
        Pssb = np.sum(Pm, axis=0) # [n_space, n_trials]

        # labels for mean computation (stimulus-shifted for error computation)
        labels = np.linspace(-np.pi, np.pi, 96, endpoint = False)
        labels = {stim : np.roll(labels, shift) for stim, shift in zip( utils.exp_stim_list(), range(0,96,4) )}
        labels = np.stack([labels[stim] for stim in data['deg']['stim']], axis=-1) # [n_space, n_trials]

        # [1] DM-time stimulus-specific bias (SSB)
        Pssb = utils.collapse( (Pssb, labels), collapse_groups=[data['delay'], data['deg']['stim']], **collapse_args )
        dm_ssb[i_models,i_model] = Pssb  # radian unit
        
        # [2] DM-time SSB weights using intercept-free regression
        ssb_pred = ssb( utils.exp_stim_list(), ssb_fits[int(n_model)] ) * np.pi/90  # radian unit
        dm_ssb_weights[i_models,i_model,0] = LinearRegression().fit(ssb_pred.reshape(-1,1), Pssb[0]).coef_[0]
        dm_ssb_weights[i_models,i_model,1] = LinearRegression().fit(ssb_pred.reshape(-1,1), Pssb[1]).coef_[0]

        # [3] EM-time decision-consistent bias (DCB)
        P1 = utils.collapse( (P, labels), collapse_groups=[data['delay'], data['evidence']], **collapse_args )
        em_dcb[i_models,i_model] = P1  # radian unit

        # [4] EM-time DCB at near references
        idx_near = np.abs(data['evidence']) < 8
        P2 = utils.collapse(
            (P[...,idx_near], labels[...,idx_near]), collapse_groups=data['delay'][idx_near], **collapse_args
        )
        em_dcb_near[i_models,i_model] = P2  # radian unit

# save results
res_ssb = {}
res_ssb_weights = {}
res_dcb = { 'refwise': {}, 'combined': {} }

stim_eval = utils.exp_stim_list(step=0.75)

for i_models, v_models in enumerate(['full', 'reduced', 'null']):
    res_ssb[v_models] = {'early' : [], 'late' : []}
    res_ssb_weights[v_models] = {}

    # stimulus-specific bias
    for i_id in range(50):
        ssb.fit( utils.dir2ori(dm_ssb[i_models,i_id,0]) )
        res_ssb[v_models]['early'].append( ssb(stim_eval) )

        ssb.fit( utils.dir2ori(dm_ssb[i_models,i_id,1]) )
        res_ssb[v_models]['late'].append( ssb(stim_eval) )

    # stimulus-specific bias weights
    res_ssb_weights[v_models]['early'] = dm_ssb_weights[i_models,:,0]
    res_ssb_weights[v_models]['late']  = dm_ssb_weights[i_models,:,1]

    # decision-consistent bias
    res_dcb['refwise'][v_models] = {} # decision-consistent bias
    res_dcb['combined'][v_models] = {} # decision-consistent bias overall at near references
    
    for i_delay, v_delay in enumerate(['early', 'late']):
        res_dcb['refwise'][v_models][v_delay] = {
            'cw'  : utils.dir2ori(em_dcb[i_models,:,i_delay,:,0]), 
            'ccw' : utils.dir2ori(em_dcb[i_models,:,i_delay,:,1]),
        }
        res_dcb['combined'][v_models][v_delay] = {
            'cw'  : utils.dir2ori(em_dcb_near[i_models,:,i_delay,0]), 
            'ccw' : utils.dir2ori(em_dcb_near[i_models,:,i_delay,1]),
        }
utils.save(res_dcb, f'{utils.ORIGIN}/data/outputs/ddm/results_decision_conditioned.pickle')
utils.save(res_ssb, f'{utils.ORIGIN}/data/outputs/ddm/results_stimulus_conditioned.pickle')
utils.save(res_ssb_weights, f'{utils.ORIGIN}/data/outputs/ddm/results_stimulus_specific_bias_weight.pickle')
print('model predictions saved')


"""
2. Shapes of stimulus-specific decision-consistent biases
    under normative unique stimuli for experiment
    outputs : results_stimulus_specific_decision_conditioned.pickle
"""
data = utils.exp_structure(near_only=True)
data = {
    'deg': {
        'stim': data['stimulus'],
        'ref' : utils.wrap(data['stimulus']+data['reference'], period=180.)
    },
    'relref'   :  data['reference'],
    'evidence' : -data['reference'],
    'delay'    :  data['timing']
}
labels = np.linspace(-np.pi, np.pi, 96, endpoint = False)
dlabel = labels[1] - labels[0]
shifts = {stim : shift for stim, shift in zip( utils.exp_stim_list(), range(0,96,4) )}

ssdcb_m = {'full': {}, 'reduced': {}}
ssdcb_s = {'full': {}, 'reduced': {}}
for i_models, (modeln, models) in enumerate(zip(['full', 'reduced'], [models_full, models_rdcd])):

    Ps = [] 
    for i_model, (n_model, model) in enumerate(models.items()):
        # load model
        fitted_params = model.fitted_params.copy()
        data = model.convert_unit(data)
        model.gen_mask(data)

        # compute densities
        P = model.forward(fitted_params, data)
        Ps.append(P)

    # compute conditional density of errors
    Ps = np.mean(Ps,axis=0)
    Ps_err = [np.roll(Ps[...,i], -shifts[stim], axis=-1) for i, stim in enumerate(data['deg']['stim'])]
    Ps_err = np.stack(Ps_err, axis=-1)
    Ps_err = utils.collapse(Ps_err,
                            collapse_groups=[data['delay'], data['deg']['stim']], 
                            collapse_kwargs={'axis':-1},
                            return_list=True)
    Ps_err = np.array(Ps_err) / np.sum(Ps_err, axis=-1, keepdims=True) / utils.dir2ori(dlabel) 

    # compute means and standard errors
    ns = 75  # (=50x1.5) trials under normative unique stimuli and equal-prob dm
    ms = utils.pop_vector_decoder( Ps_err, utils.dir2ori(labels) )
    ds = ( utils.dir2ori(labels)[None,None,None,:] - ms[...,None] )**2
    vs = np.sum( Ps_err[0,:,:]*ds*utils.dir2ori(dlabel), axis=-1 )
    ss = np.sqrt(vs) / np.sqrt(ns)

    ssdcb_m[modeln]['early'] = {'cw': ms[0,:,0], 'ccw': ms[0,:,1]}
    ssdcb_m[modeln]['late']  = {'cw': ms[1,:,0], 'ccw': ms[1,:,1]}
    ssdcb_s[modeln]['early'] = {'cw': ss[0,:,0], 'ccw': ss[0,:,1]}
    ssdcb_s[modeln]['late']  = {'cw': ss[1,:,0], 'ccw': ss[1,:,1]}

utils.save({'m': ssdcb_m, 's': ssdcb_s}, f'{utils.ORIGIN}/data/outputs/ddm/results_stimulus_specific_decision_conditioned.pickle')
print('results_stimulus_specific_decision_conditioned.pickle saved')


"""
3. Near-reference variability in DDMs
    output : results_near_reference_variability.pickle
"""
data = utils.exp_structure()
data = {
    'deg': {
        'stim': data['stimulus'],
        'ref' : utils.wrap(data['stimulus']+data['reference'], period=180.)
    },
    'relref'   :  data['reference'],
    'evidence' : -data['reference'],
    'delay'    :  data['timing']
}

Ps = [] 
Ps_no_wa = []
models = models_full
for i_model, (n_model, model) in enumerate(models.items()):
    # load model
    fitted_params = model.fitted_params.copy()
    data = model.convert_unit(data)
    model.gen_mask(data)

    # compute densities
    fitted_params[-4] = 0 # nuisance parameter
    P = model.forward(fitted_params, data)
    Ps.append(P)

    fitted_params[-3] = 0 # choice-induced bias 
    P = model.forward(fitted_params, data)
    Ps_no_wa.append(P)

Ps = np.array(Ps)
Ps_no_wa = np.array(Ps_no_wa)

# 
nrv = {'refwise' : {}, 'combined' : {}}
for pmod, nmod in zip( [Ps, Ps_no_wa], ['full', 'full_no_wa'] ):
    Ps_err = [np.roll(pmod[...,i], -shifts[stim], axis=-1) for i, stim in enumerate(data['deg']['stim'])]
    Ps_err = np.stack(Ps_err, axis=-1).sum(axis=1)

    # refwise
    Ps_err_refwise = utils.collapse(Ps_err,
                                    collapse_groups=[data['relref']],
                                    collapse_kwargs={'axis':-1}, return_list=True)
    Ps_err_refwise = np.array(Ps_err_refwise) / np.sum(Ps_err_refwise, axis=-1, keepdims=True) / utils.dir2ori(dlabel) 
    iqrs = np.apply_along_axis(utils.quantile, axis=-1, arr=Ps_err_refwise, support=utils.dir2ori(labels), quantile=[0.25, 0.75])
    iqrs = iqrs[...,1] - iqrs[...,0]
    nrv['refwise'][nmod] = dict(zip(utils.exp_ref_list(), iqrs))

    # combined
    idx_near = np.abs(data['evidence']) < 8
    Ps_err_combined = utils.collapse(Ps_err,
                                     collapse_groups=[idx_near],
                                     collapse_kwargs={'axis':-1}, return_list=True)
    Ps_err_combined = np.array(Ps_err_combined) / np.sum(Ps_err_combined, axis=-1, keepdims=True) / utils.dir2ori(dlabel)
    iqrs = np.apply_along_axis(utils.quantile, axis=-1, arr=Ps_err_combined, support=utils.dir2ori(labels), quantile=[0.25, 0.75])
    iqrs = iqrs[...,1] - iqrs[...,0]
    nrv['combined'][nmod] = dict(zip(['far', 'near'], iqrs))

utils.save(nrv, f'{utils.ORIGIN}/data/outputs/ddm/results_near_reference_variability.pickle')
print('near_reference_variability.pickle saved')