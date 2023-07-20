"""generates results_stimulus_specific_bias.pickle
"""
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
sys.path.append('../..')
from src import utils

# download data
utils.download_dataset("data/processed/behavior")

# load outputs
with open(f"{utils.ORIGIN}/data/outputs/behavior/bootstrap_psychometric_curve.pickle", 'rb') as f:
    psych = pickle.load(f)

# load behavior
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]
sub_list  = np.sort(behavior.ID.unique())
stim_list = np.sort(behavior.stim.unique())

# design matrix
n_stim    = len(stim_list)
n_basis   = 12
p_basis   = n_basis/2.
c_basis   = np.linspace(0, 2*np.pi, n_basis, endpoint = False)
x         = np.linspace(0, 2*np.pi, n_stim,  endpoint = False).reshape((-1,1))
X         = utils.derivative_von_mises(x, c_basis, p_basis)

# fit weights for each participant
results = {}
results['info'] = {'n_basis': n_basis, 'p_basis': p_basis}
results['weights'] = {}

for i_id, v_id in enumerate(sub_list):
    sub_behav = behavior[behavior.ID == v_id]
    y         = [utils.circmean(sub_behav.error[sub_behav.stim==s]) for s in stim_list]
    reg       = LinearRegression().fit(X,y)
    results['weights'][v_id] = [reg.intercept_, *reg.coef_]

# save
utils.mkdir(f"{utils.ORIGIN}/data/outputs/behavior")
with open(f'{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle', 'wb') as f:
    pickle.dump(results,f)