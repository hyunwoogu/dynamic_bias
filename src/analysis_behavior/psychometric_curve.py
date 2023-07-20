"""generates bootstrap_psychometric_curve.pickle
"""
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.append('../..')
from src import utils

utils.download_dataset("data/processed/behavior")
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]

def nll(par, evi, dm):
    """negative log likelihoods for psychometric curves (no constraints)"""
    m, s, lam = par
    prob = utils.psi(evi, m, s, lam)
    prob_cw  = prob[dm==1]
    prob_ccw = prob[dm==0]
    prob_cw[prob_cw<=utils.EPS] = utils.EPS            
    prob_ccw[prob_ccw>=1.-utils.EPS] = 1.-utils.EPS
    ll   = np.sum(np.log(prob_cw)) + np.sum(np.log(1.-prob_ccw))
    return -ll

# ground truth
ground_truth = np.nan*np.zeros((2,3))
for i_t, v_t in enumerate([1,2]):
    idx = behavior.Timing.to_numpy()==v_t
    evi = -behavior.ref.to_numpy()
    dm  = 2.-behavior.choice.to_numpy()
    res = minimize(nll, [0., 1., 0.1], 
                   args=(evi[idx], dm[idx]), bounds=[[-20.,20.], [utils.EPS, 20.], [0, 0.15]])
    ground_truth[i_t] = res['x']

# parameters
n_boostrap = 10000
timings    = [1,2]
ref_relative = [21,4,0,-4,-21] # relative orientations of reference
choice_boot  = np.nan*np.empty([len(timings),len(ref_relative),n_boostrap])

for i_t, v_t in enumerate(timings):
    for i_r, v_r in enumerate(ref_relative):
        idx = (behavior.ref==v_r) & (behavior.Timing==v_t)
        evi = -behavior.ref.to_numpy()[idx]
        dm  = 2.-behavior.choice.to_numpy()[idx]
        for i_boot in range(n_boostrap):
            idx_boot = np.random.choice(sum(idx), sum(idx), replace=True)
            choice_boot[i_t,i_r,i_boot] = np.mean(dm[idx_boot])

# 
bootstrap_psi = np.nan*np.empty([len(timings),3,n_boostrap])
for i_t, v_t in enumerate(timings):
    idx = behavior.Timing==v_t
    evi = -behavior.ref.to_numpy()
    dm  = 2.-behavior.choice.to_numpy()

    for i_boot in range(n_boostrap):    
        idx_boot = np.random.choice(sum(idx), sum(idx), replace=True)
        res      = minimize(nll, [0., 1., 0.1], 
                            args=(evi[idx][idx_boot], dm[idx][idx_boot]),
                            bounds=[[-20.,20.], [utils.EPS, 20.], [0, 0.15]])
        
        bootstrap_psi[i_t,:,i_boot] = res['x']
        if (i_boot+1) % 500 == 0: print(i_boot)

# 
bootstrap_psi_H0 = np.nan*np.empty([len(timings),3,n_boostrap])
bootstrap_psi_H1 = np.nan*np.empty([len(timings),3,n_boostrap])
x0   = [0., 1., 0.1]
bnds = [[-20.,20.], [utils.EPS, 20.], [0, 0.15]]
tim  = behavior.Timing.to_numpy()
evi  = -behavior.ref.to_numpy()
dm   = 2.-behavior.choice.to_numpy()

# bootstrap distribution under H0
for i_boot in range(n_boostrap):
    idx_boot = np.random.choice(len(behavior), len(behavior), replace=True)
    tim_boot = tim[idx_boot]
    np.random.shuffle(tim_boot)
    evi_boot = evi[idx_boot]
    dm_boot  = dm[idx_boot]
    
    success_e = False 
    success_l = False 
    while not success_e:
        res_e = minimize(nll, x0, args=(evi_boot[tim_boot==1], dm_boot[tim_boot==1]), bounds=bnds)
        success_e = res_e['success']
    while not success_l:
        res_l = minimize(nll, x0, args=(evi_boot[tim_boot==2], dm_boot[tim_boot==2]), bounds=bnds)
        success_l = res_l['success']
        
    bootstrap_psi_H0[0,:,i_boot] = res_e['x']
    bootstrap_psi_H0[1,:,i_boot] = res_l['x']


# bootstrap distribution under H1
for i_boot in range(n_boostrap):
    idx_boot = np.random.choice(len(behavior), len(behavior), replace=True)
    tim_boot = tim[idx_boot]
    evi_boot = evi[idx_boot]
    dm_boot  = dm[idx_boot]
    
    success_e = False 
    success_l = False 
    while not success_e:
        res_e = minimize(nll, x0, args=(evi_boot[tim_boot==1], dm_boot[tim_boot==1]), bounds=bnds)
        success_e = res_e['success']
    while not success_l:
        res_l = minimize(nll, x0, args=(evi_boot[tim_boot==2], dm_boot[tim_boot==2]), bounds=bnds)
        success_l = res_l['success']
        
    bootstrap_psi_H1[0,:,i_boot] = res_e['x']
    bootstrap_psi_H1[1,:,i_boot] = res_l['x']

# 
res_boot = {
    'ground_truth': ground_truth,
    'bootstrap_choice': choice_boot,     # bootstrap samples for choices
    'bootstrap0': bootstrap_psi,         # bootstrap samples for the 'shades' of psychometric curves
    'bootstrap1': bootstrap_psi_H1,      # bootstrap samples for drawing 
    'permutation_null': bootstrap_psi_H0 # random permutation samples for null distribution
}

# save
utils.mkdir(f"{utils.ORIGIN}/data/outputs/behavior")
with open(f"{utils.ORIGIN}/data/outputs/behavior/bootstrap_psychometric_curve.pickle", 'wb') as f:
    pickle.dump(res_boot, f)