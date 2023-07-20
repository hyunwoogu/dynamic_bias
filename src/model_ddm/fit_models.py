"""fit DDMs
"""
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append('../..')
from src import utils
from src.model_ddm.model import DynamicBiasModel
utils.download_dataset("data/processed/behavior")
utils.download_dataset("data/outputs/behavior")

# 
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]

# load outputs
with open(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle", 'rb') as f:
    results = pickle.load(f)

# fitting models
IDs = behavior.ID
IDs_list = np.unique(IDs)
for sub in IDs_list:
    for model_type in ['full', 'reduced']:
        utils.mkdir(f"{utils.ORIGIN}/models/ddm/{model_type}")
        idx  = behavior.ID == sub
        data =  {
            'deg': {
                'stim'   : behavior.stim.to_numpy()[idx],
                'ref'    : (behavior.stim+behavior.ref).to_numpy()[idx],
                'estim'  : behavior.esti.to_numpy()[idx],
            },
            'relref' : behavior.ref.to_numpy()[idx],
            'delay'  : behavior.Timing.to_numpy()[idx],
            'dm'     : (2 - behavior.choice.to_numpy())[idx],
        }
        
        print(f'sub-{sub}##################################')
        ssb = lambda m: utils.stimulus_specific_bias(m*90/np.pi, results['weights'][sub], **results['info'])
        model = DynamicBiasModel(stimulus_specific_bias=ssb, weights=results['weights'][sub], model=model_type)
        model.fit(data)
        with open(f'{utils.ORIGIN}/models/ddm/{model_type}/fitted_model_sub-{sub:04}'+'.pkl', 'wb') as f:
            pickle.dump(model, f)