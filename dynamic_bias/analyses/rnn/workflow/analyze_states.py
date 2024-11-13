"""analyze states of trained RNNs
"""
import numpy as np
from sklearn.decomposition import PCA

import torch
from dynamic_bias import utils
import dynamic_bias.analyses.rnn as rnna
from dynamic_bias.models.rnn import par, hp, update_parameters

"""
1. Load trained Js
    outputs : trained_J.pickle
"""
N_MODELS = 50
Js = {}
for model_type in ['heterogeneous', 'homogeneous']:
    model_dir = f"{utils.ORIGIN}/models/rnn/{model_type}" + "/network{:02d}"
    Js[model_type] = {
        'J11': utils.nan((N_MODELS,48,48)),
        'J12': utils.nan((N_MODELS,48,48)),
        'J21': utils.nan((N_MODELS,48,48)),
        'J22': utils.nan((N_MODELS,48,48)),
    }
    for i_model in range(N_MODELS): 
        model = torch.load(model_dir.format(i_model))
        Js[model_type]['J11'][i_model] = model.var_dict['w_rnn11'].detach().numpy() # DM to DM
        Js[model_type]['J12'][i_model] = model.var_dict['w_rnn12'].detach().numpy() # DM to EM
        Js[model_type]['J21'][i_model] = model.var_dict['w_rnn21'].detach().numpy() # EM to DM
        Js[model_type]['J22'][i_model] = model.var_dict['w_rnn22'].detach().numpy() # EM to EM

utils.save(Js, f"{utils.ORIGIN}/data/outputs/rnn/trained_J.pickle")
print(f"trained_J.pickle saved")


"""
2. Estimate connectivity params
    output: params_connectivity.pickle
"""
Js = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/trained_J.pickle")
J_hom = {k:np.mean(v,axis=0) for k,v in Js['homogeneous'].items()}
J_het = {k:np.mean(v,axis=0) for k,v in Js['heterogeneous'].items()}

print('\nHomogeneous RNN : connectivity parameters (direction representation, degrees)')
J_recon_hom, J_param_hom = rnna.lowrank_J(J_hom)
print('{:<7s} {:<12s} {:<10s}'.format(
    'Jhat', r'phi', 'w1')
     )
for k,v in J_param_hom.items():
    print('{:<7s} {:<12f} {:<10f}'.format(
        k, -v[0]*180/np.pi, v[1]) 
    )

print('\nHeterogeneous RNN : connectivity parameters (direction representation, degrees)')
J_recon_het, J_param_het = rnna.lowrank_J(J_het, hom=False)
print('{:<7s} {:<12s} {:<10s} {:<10s}'.format(
    'Jhat', r'phi', 'w1', 'w2')
     )
for k,v in J_param_het.items():
    print('{:<7s} {:<12f} {:<10f} {:<10f}'.format(
        k, -v[0]*180/np.pi, v[1][0], v[1][1]) 
    )

utils.save({
    'J_recon_hom' : J_recon_hom,
    'J_param_hom' : J_param_hom,
    'J_recon_het' : J_recon_het,
    'J_param_het' : J_param_het,
}, f"{utils.ORIGIN}/data/outputs/rnn/params_connectivity.pickle")
print('params_connectivity.pickle saved')


"""
3. RUN homogeneous and heterogeneous RNNs and project their activities onto the PCA space
    outputs: 
        results_pca.pickle
        demo_rnn_activity.pickle
        results_state_space.pickle
"""
Js = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/trained_J.pickle")
J_hom = {k:np.mean(v,axis=0) for k,v in Js['homogeneous'].items()}
J_het = {k:np.mean(v,axis=0) for k,v in Js['heterogeneous'].items()}

# simpler structure for state-space analysis
hp['noise_rnn_sd']  = 0
par['gamma_ref']    = 3.
par['batch_size']   = 1
par['noise_sd']     = 0

DUR = 0.6  # DM duration
par['design'].update({'iti'     : (0,   0.1),
                      'stim'    : (0.1, 0.7),
                      'decision': (0.8, 0.8+DUR),
                      'delay'   : ((0.7, 0.8),(0.8+DUR, 0.8+DUR+0.1)),
                      'estim'   : (0.8+DUR+0.1,0.8+DUR+1.)})
par['reference'] = np.array([-1,0,1])
par        = update_parameters(par)
r1, r2 = rnna.run_rnn(par, hp, J_hom)

# fit PCA to the stacked r_theta (for relative reference=0)
rndseed    = 2023
r_theta    = (r2[:,:,1,:,:24] + r2[:,:,1,:,24:])/2.
pca_shared = PCA(n_components=2, random_state=rndseed)
pca_shared.fit(r_theta.reshape((-1,24)))

utils.save(pca_shared, f"{utils.ORIGIN}/data/outputs/rnn/results_pca.pickle")
print('results_pca.pickle saved')


# [1] reference input directions 
phi, flip  = utils.fit_rotation( pca_shared.components_.T )
ref_inputs = utils.rotate( pca_shared.transform( np.eye(24) ), -phi, flip=flip, late_flip=False )

# [2] homogeneous RNNs
r1, r2 = rnna.run_rnn(par, hp, J_hom)
hom_r1_cw = np.stack([
    np.stack([
        utils.rotate( pca_shared.transform(r1[:,s,r,0,:24]), -phi, flip=flip, late_flip=False ) for s in range(24)
    ],axis=1) for r in range(3)
], axis=2)

hom_r1_ccw = np.stack([
    np.stack([
        utils.rotate( pca_shared.transform(r1[:,s,r,0,24:]), -phi, flip=flip, late_flip=False ) for s in range(24)
    ],axis=1) for r in range(3)
], axis=2)

hom_r2 = np.stack([
    np.stack([
        utils.rotate( pca_shared.transform((r2[:,s,r,0,:24]+r2[:,s,r,0,24:])/2.), -phi, flip=flip, late_flip=False ) for s in range(24)
    ],axis=1) for r in range(3)
], axis=2)

utils.save({
    'hom_r2'         : (r2[:,12,:,0,:24]+r2[:,12,:,0,24:])/2.,
    'hom_r1_proj_cw' : rnna.project_to_θ(r1[:,12,:,0,:], J_hom['J12'])[...,:24],
    'hom_r1_proj_ccw': rnna.project_to_θ(r1[:,12,:,0,:], J_hom['J12'])[...,24:],
}, f"{utils.ORIGIN}/data/outputs/rnn/demo_rnn_activity.pickle")
print('demo_rnn_activity.pickle saved')

# [3] heterogeneous RNNs
r1, r2 = rnna.run_rnn(par, hp, J_het)
het_r1_cw = np.stack([
    np.stack([
        utils.rotate( pca_shared.transform(r1[:,s,r,0,:24]), -phi, flip=flip, late_flip=False ) for s in range(24)
    ],axis=1) for r in range(3)
], axis=2)

het_r1_ccw = np.stack([
    np.stack([
        utils.rotate( pca_shared.transform(r1[:,s,r,0,24:]), -phi, flip=flip, late_flip=False ) for s in range(24)
    ],axis=1) for r in range(3)
], axis=2)

het_r2 = np.stack([
    np.stack([
        utils.rotate( pca_shared.transform((r2[:,s,r,0,:24]+r2[:,s,r,0,24:])/2.), -phi, flip=flip, late_flip=False ) for s in range(24)
    ],axis=1) for r in range(3)
], axis=2)

utils.save({
    'ref_input'  : ref_inputs,
    'hom_r1_cw'  : hom_r1_cw,
    'hom_r1_ccw' : hom_r1_ccw,
    'hom_r2'     : hom_r2,
    'het_r1_cw'  : het_r1_cw,
    'het_r1_ccw' : het_r1_ccw,
    'het_r2'     : het_r2,
}, f"{utils.ORIGIN}/data/outputs/rnn/results_state_space.pickle")
print('results_state_space.pickle saved')


"""
4. Reference-strength-driven changes
    output: results_state_space_strengths.pickle
"""
pca_shared = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/results_pca.pickle")
phi, flip  = utils.fit_rotation( pca_shared.components_.T )

Js = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/trained_J.pickle")
J_hom = {k:np.mean(v,axis=0) for k,v in Js['homogeneous'].items()}
J_het = {k:np.mean(v,axis=0) for k,v in Js['heterogeneous'].items()}

r2_homs = {}
r2_hets = {}

## simpler structure for state-space analysis
hp['noise_rnn_sd']  = 0
par['batch_size']   = 1
par['noise_sd']     = 0

DUR = 0.6  # DM duration
par['design'].update({'iti'     : (0,   0.1),
                      'stim'    : (0.1, 0.7),
                      'decision': (0.8, 0.8+DUR),
                      'delay'   : ((0.7, 0.8),(0.8+DUR, 0.8+DUR+0.1)),
                      'estim'   : (0.8+DUR+0.1,0.8+DUR+1.)})
par['reference'] = np.array([-1,1])

for sth, srnth in enumerate( np.linspace(0,3,num=31) ):
    par['gamma_ref'] = srnth
    par        = update_parameters(par)
    
    # homogeneous 
    _, _r2 = rnna.run_rnn(par, hp, J_hom)
    r2_homs[srnth] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform((_r2[:,s,r,0,:24]+_r2[:,s,r,0,24:])/2.), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)
    
    # heterogeneous
    _, _r2 = rnna.run_rnn(par, hp, J_het)
    r2_hets[srnth] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform((_r2[:,s,r,0,:24]+_r2[:,s,r,0,24:])/2.), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)

utils.save({
    'r2s_hom' : r2_homs,
    'r2s_het' : r2_hets,
}, f"{utils.ORIGIN}/data/outputs/rnn/results_state_space_strengths.pickle")
print('results_state_space_strengths.pickle saved')



"""
5. Reference-duration-driven changes
    output: results_state_space_durations.pickle
"""
pca_shared = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/results_pca.pickle")
phi, flip  = utils.fit_rotation( pca_shared.components_.T )

Js = utils.load(f"{utils.ORIGIN}/data/outputs/rnn/trained_J.pickle")
J_hom = {k:np.mean(v,axis=0) for k,v in Js['homogeneous'].items()}
J_het = {k:np.mean(v,axis=0) for k,v in Js['heterogeneous'].items()}

r1_homs, r2_homs = {'cw': {}, 'ccw': {}}, {}
r1_hets, r2_hets = {'cw': {}, 'ccw': {}}, {}

## simpler structure for state-space analysis
hp['noise_rnn_sd'] = 0
par['batch_size']  = 1
par['noise_sd']    = 0
par['reference']   = np.array([-1,1])
par['gamma_ref']   = 3.

for dur, durn in zip([0.6, 0.9, 1.2], ['s', 'm', 'l']):
    par['design'].update({'iti'     : (0,   0.1),
                          'stim'    : (0.1, 0.7),
                          'decision': (0.8, 0.8+dur),
                          'delay'   : ((0.7, 0.8),(0.8+dur, 0.8+dur+0.1)),
                          'estim'   : (0.8+dur+0.1,0.8+dur+1.)})

    par = update_parameters(par)

    # homogeneous 
    _r1, _r2 = rnna.run_rnn(par, hp, J_hom)

    r1_homs['cw'][durn] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform(_r1[:,s,r,0,:24]), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)

    r1_homs['ccw'][durn] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform(_r1[:,s,r,0,24:]), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)

    r2_homs[durn] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform((_r2[:,s,r,0,:24]+_r2[:,s,r,0,24:])/2.), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)

    
    # heterogeneous
    _r1, _r2 = rnna.run_rnn(par, hp, J_het)
    
    r1_hets['cw'][durn] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform(_r1[:,s,r,0,:24]), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)

    r1_hets['ccw'][durn] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform(_r1[:,s,r,0,24:]), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)

    r2_hets[durn] = np.stack([
        np.stack([
            utils.rotate( pca_shared.transform((_r2[:,s,r,0,:24]+_r2[:,s,r,0,24:])/2.), -phi, flip=flip, late_flip=False ) for s in range(24)
        ],axis=1) for r in range(2)
    ], axis=2)

    print(f'{durn} done')

utils.save({
    'r1s_hom' : r1_homs,
    'r1s_het' : r1_hets,
    'r2s_hom' : r2_homs,
    'r2s_het' : r2_hets,
}, f"{utils.ORIGIN}/data/outputs/rnn/results_state_space_durations.pickle")
print('results_state_space_durations.pickle saved')