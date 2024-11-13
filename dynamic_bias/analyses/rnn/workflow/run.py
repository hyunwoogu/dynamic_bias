"""
run RNNs in generalization episodes save outputs
    time-consuming : can be parallelized
"""
import argparse
import numpy as np
import torch

from dynamic_bias import utils
import dynamic_bias.models.rnn.train.pytorch as rnnt
import dynamic_bias.analyses.rnn as rnna
from dynamic_bias.models.rnn import \
    Stimulus, par, update_parameters, hp, update_hp

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='heterogeneous')
args = parser.parse_args()
MODEL_TYPE = args.model_type

SEED      = 27
N_MODEL   = 50
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = f"{utils.ORIGIN}/models/rnn/{MODEL_TYPE}" + "/network{:02d}"

if MODEL_TYPE.endswith('emonly'):
    hp['lam_decision'] = 0. # no decision cost

if MODEL_TYPE.endswith('d2e_ablation'):
    hp['DtoE_off'] = True # no DM to EM connection

if 'heterogeneous' in MODEL_TYPE:
    par['gamma']    = 10. # gamma_D
    par['noise_sd'] = 0.  # noise level, 0 when gamma > 0
    noise_vec  = np.abs(np.sin(np.linspace(0,2*np.pi,24,endpoint=False)))
    noise_vec *= par['gamma']
    par['noise_center'] = noise_vec

par['batch_size'] = 1
par['reference'] = np.array([-3,-2,-1,0,1,2,3]) # per 7.5 degree
par = update_parameters(par)

hp  = update_hp(hp)
hp  = rnnt.tensorize_hp(hp)
hp  = {k:v.to(DEVICE) for k,v in hp.items()}

n_ori, n_ref, n_batch, n_time = \
    par['n_ori'], len(par['reference']), par['batch_size'], par['n_timesteps']
labels = np.linspace(0,180,num=par['n_ori'],endpoint=False)

#
datas = []
np.random.seed(SEED)
for i_model in range(N_MODEL): 
    dm_rg, em_rg = {}, {}
    model = torch.load(model_dir.format(i_model)).to(DEVICE)
    
    # run RNNs
    ids = []
    timings = []
    trial_info = []
    for timing in ['early', 'late']:
        if timing == 'early':
            # task structure for early DM condition
            par['design'].update({'iti'   : (0, 0.1),
                                'stim'    : (0.1, 0.7),
                                'decision': (2.5, 3.1),
                                'delay'   : ((0.7,2.5),(3.1,7.3)),
                                'estim'   : (7.3,9.1)})
        
        if timing == 'late':
            # task structure for late DM condition
            par['design'].update({'iti'     : (0, 0.1),
                                  'stim'    : (0.1, 0.7),                      
                                  'decision': (4.9, 5.5),
                                  'delay'   : ((0.7,4.9),(5.5,7.3)),
                                  'estim'   : (7.3,9.1)})

        par = update_parameters(par)
        dm_rg[timing] = par['design_rg']['decision']
        em_rg[timing] = par['design_rg']['estim']

        # 
        for i_stim in range(par['n_ori']):
            par['stim_dist'] = np.eye(par['n_ori'])[i_stim]

            for i_ref, v_ref in enumerate(par['reference']):
                par['ref_dist'] = np.eye(len(par['reference']))[ i_ref ]
                par             = update_parameters(par)
                stimulus        = Stimulus(par)
                trial_info.append( rnnt.tensorize_trial(stimulus.generate_trial(), DEVICE) )
                ids.append(i_model)
                timings.append(timing)

    #         
    timings = np.array(timings)
    stimulus_ori  = torch.concat([t['stimulus_ori']  for t in trial_info]).numpy() * 180./par['n_ori']
    reference_ori = torch.concat([t['reference_ori'] for t in trial_info]).numpy() * 180./par['n_ori']
    trial_info = {
        k: torch.concat([x[k] for x in trial_info], dim=1) for k in trial_info[0].keys() if not k.endswith('ori') 
    }

    #
    pred_output_dm, pred_output_em, _, _  = model.rnn_model(trial_info['u_rho'], trial_info['u_the'], hp)
    em_readout = utils.pop_vector_decoder( rnna.softmax_pred_output(pred_output_em.detach()), labels )

    output_dm = pred_output_dm.detach().numpy()
    output_dm = np.concatenate([output_dm[dm_rg['early']][:,timings=='early'],
                                output_dm[dm_rg['late']] [:,timings=='late']], axis=1)
    output_em = pred_output_em.detach().numpy()
    output_em = np.concatenate([output_em[em_rg['early']][:,timings=='early'],
                                output_em[em_rg['late']] [:,timings=='late']], axis=1)
    
    dm_behav  = rnna.behavior_summary_dm(output_dm)
    em_behav  = rnna.behavior_summary_em(output_em, par=par)

    datas.append({
        'ID'        : ids,
        'timing'    : timings,
        'stimulus'  : stimulus_ori,
        'relref'    : reference_ori,
        'choice'    : dm_behav,
        'estim'     : em_behav,
        'em_readout': em_readout,
    })

    print(f"Model {i_model} done")

# aggregate
datas = {k: np.concatenate([v[k] for v in datas], axis=-1) for k in datas[0].keys()}
utils.save(datas, f"{utils.ORIGIN}/data/outputs/rnn/results_{MODEL_TYPE}.pickle")
print(f"results_{MODEL_TYPE}.pickle saved")