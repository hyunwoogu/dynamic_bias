"""fit DDMs
"""
import numpy as np
from dynamic_bias import utils
from dynamic_bias.models.ddm.model import DynamicBiasModel
from dynamic_bias.analyses.behavior import StimulusSpecificBias

utils.download_dataset("data/outputs/behavior")

# 
behavior = utils.load_behavior()
ssb_fits = utils.load(f"{utils.ORIGIN}/data/outputs/behavior/results_stimulus_specific_bias.pickle")
ssb = StimulusSpecificBias()

# fitting models
id_list = np.unique(behavior.ID)
for v_id in id_list:
    for model_type in ['full', 'reduced', 'null']:
        utils.mkdir(f"{utils.ORIGIN}/models/ddm/{model_type}")
        idx  = behavior.ID == v_id
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
        
        print(f'sub-{v_id}##################################')
        ssb_fun = lambda m: ssb(utils.dir2ori(m), ssb_fits[v_id])
        model = DynamicBiasModel(stimulus_specific_bias=ssb_fun, weights=ssb_fits[v_id], model=model_type)
        model.fit(data)
        utils.save(model, f'{utils.ORIGIN}/models/ddm/{model_type}/fitted_model_sub-{v_id:04}.pkl')