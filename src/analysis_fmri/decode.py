"""decode fMRI data
"""
# import os
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append('../..')
from src import utils
from linear_decoding_model import LinearDecodingModel

utils.download_dataset("data/processed/behavior")
utils.download_dataset("data/processed/fmri")
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")

for i_sub, v_sub in enumerate(np.unique(behavior.ID)):    
    utils.mkdir(f"{utils.ORIGIN}/data/processed/fmri/decoding")

    # Load the visual cortex voxels: shape: [n_voxels, n_trials, n_timepoint (TR 3-14)]
    with open(f"{utils.ORIGIN}/data/processed/fmri/visual_voxels/visual_voxels_sub-{v_sub:04}.npy", 'rb') as f: 
        X = np.load(f)
    
    # 
    behav = behavior[behavior.ID==v_sub]
    idx   = (~np.isnan(behav.choice)) & (~np.isnan(behav.error))
    
    # 
    stim  = behav.stim.to_numpy()    # stimuli
    stim  = (stim / 7.5).astype(int) # stimuli(integerized)
    idx_e = (behav.Timing==1)        # idx for "Early" DM trials
    idx_l = (behav.Timing==2)        # idx for "Late" DM trials
    T     = X.shape[-1]              # n_timestep
    
    # Decoding loop for each timepoint
    print(f'Decoding stimuli from the visual voxels of sub-{v_sub:04}...')
    chan_recon = np.nan*np.zeros((T, sum(idx), 120)) # [n_timepoints, n_trials, n_psi]
    
    for t_train in range(T):
        X_train = X[:,:,t_train].T      # [n_trials, n_voxels]
        model_e = LinearDecodingModel() # decoding model for "Early" DM trials
        model_l = LinearDecodingModel() # decoding model for "Late" DM trials
        
        # cross-validation
        cv = np.arange(len(stim)).reshape((-1,12))
        model_e.fit(X_train, stim, cv=cv, constraint=idx_e)
        model_l.fit(X_train, stim, cv=cv, constraint=idx_l)

        chan_recon_t = np.nan*np.zeros((len(stim),120))
        chan_recon_t[idx&idx_e] = model_e.predict(X_train)[idx&idx_e]
        chan_recon_t[idx&idx_l] = model_l.predict(X_train)[idx&idx_l]
        chan_recon[t_train] = chan_recon_t[idx]
        
        print(f'TR={t_train+3} completed')
    
    with open(f"{utils.ORIGIN}/data/processed/fmri/decoding/decoding_sub-{v_sub:04}.pickle", 'wb') as f:
        pickle.dump({'behav': behav[idx],
                     'chan' : chan_recon}, f)