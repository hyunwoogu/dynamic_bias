"""generates bootstrap_stimulus_specific_bias.pickle
"""
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.append('../..')
from src import utils

# download data
utils.download_dataset("data/processed/behavior")

# load behavior
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]

# fitting PSE & slope for each (stimulus, timing) combination
def nll_EL(par, evi, dm, tim):
    m1, m2, s1, s2_add, lam = par
    s2        = np.sqrt(s1**2 + s2_add**2)
    probE     = utils.psi(evi[tim==1], m1, s1, lam)
    probL     = utils.psi(evi[tim==2], m2, s2, lam)
    sumE_ll   = np.sum(np.log(probE[dm[tim==1]==1])) + np.sum(np.log(1.-probE[dm[tim==1]==0]))
    sumL_ll   = np.sum(np.log(probL[dm[tim==2]==1])) + np.sum(np.log(1.-probL[dm[tim==2]==0]))
    return -sumE_ll - sumL_ll

# fitting PSE from population data with bootstrapping
n_boot = 5
theta  = behavior.stim.to_numpy()
evi    = -behavior.ref.to_numpy()
dm     = 2.-behavior.choice.to_numpy()
timing = behavior.Timing.to_numpy()

stims  = np.linspace(0,180,num=24,endpoint=False)
ent_boot_dm = np.zeros((2,n_boot))
res_boot_dm = np.zeros((24,2,n_boot))
for i_boot in range(n_boot):
    success = False
    while not success:
        for i_theta, v_theta in enumerate(stims):
            idx      = (theta == v_theta)
            idx_boot = np.random.choice(sum(idx), sum(idx))
            res = minimize(nll_EL, [0,0,1.,1.,0], args = (evi[idx][idx_boot], dm[idx][idx_boot], timing[idx][idx_boot]), 
                        bounds = [[-20., 20.], [-20., 20.], [utils.EPS, 20.], [utils.EPS, 20.], [utils.EPS, 0.3]])
            res_boot_dm[i_theta, :, i_boot] = res['x'][:2]
            success = res['success']
            if not success:
                print(i_boot, v_theta, 'not successful, retrying...')

    if i_boot % 10 == 0:
        print('PSE bootstrapping completed, i=', i_boot)

# save
utils.mkdir(f"{utils.ORIGIN}/data/outputs/behavior")
with open(f"{utils.ORIGIN}/data/outputs/behavior/bootstrap_stimulus_specific_bias.pickle", 'wb') as f:
    pickle.dump(res_boot_dm, f)