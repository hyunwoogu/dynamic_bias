"""generates results_stimulus_specific_bias_weight.pickle
"""
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.append('../..')
from src import utils

# load dataset
utils.download_dataset("data/processed/behavior")

with open(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle", 'rb') as f:
    ssb = pickle.load(f)
behavior  = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]

def nll_EL(par, evi, dm, tim, stim, vec):
    """ Negative log-likelihood for psychometric curve under the slope inequality contraints
    """
    w1, w2, s1, s2_add, lam = par
    w  = [w1, w2]
    s  = [s1, np.sqrt(s1**2 + s2_add**2)]
    
    sum_nll = 0
    for i_t, v_t in enumerate([1,2]):
        for i_s, v_s in enumerate(stims):
            idx  = (tim==v_t) & (stim==v_s) 
            prob = utils.psi(evi[idx], -w[i_t]*vec[i_s], s[i_t], lam)
            prob_cw  = prob[dm[idx]==1]
            prob_ccw = prob[dm[idx]==0]
            prob_cw[prob_cw<=utils.EPS] = utils.EPS            
            prob_ccw[prob_ccw>=1.-utils.EPS] = 1.-utils.EPS
            ll   = np.sum(np.log(prob_cw)) + np.sum(np.log(1.-prob_ccw))
            sum_nll  += -ll
    
    return sum_nll

# fitting PSE & slope for each (stimulus, timing) combination
id_list = np.unique(behavior.ID)
stims   = np.arange(180,step=7.5)
res_dec = np.zeros([len(id_list),2])

for i_id, v_id in enumerate(id_list):
    sub_df = behavior[behavior.ID == v_id]
    _evi   = -sub_df.ref.to_numpy()
    _dm    = 2.-sub_df.choice.to_numpy()
    _stim  = sub_df.stim.to_numpy()
    _time  = sub_df.Timing.to_numpy()
    _ssb   = utils.stimulus_specific_bias(stims, ssb['weights'][v_id], **ssb['info'])
    
    res = minimize(nll_EL, [1.,1.,1.,1.,0.1], 
                   args = (_evi, _dm, _time, _stim, _ssb), 
                   bounds = [[utils.EPS, 2.], [utils.EPS, 2.], [utils.EPS, 20.], [0, 20.], [0, 0.15]])        
    res_dec[i_id,:] = res['x'][:2]
    print(v_id)

# save
utils.mkdir(f"{utils.ORIGIN}/data/outputs/behavior")
with open(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias_weight.pickle", 'wb') as f:
    pickle.dump(res_dec, f)