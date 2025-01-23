"""decode BOLD data"""
import pickle
import argparse
import numpy as np
import pandas as pd
from dynamic_bias import utils
from dynamic_bias.analyses.fmri import LinearDecodingModel

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='evc')
parser.add_argument('--input_dir', type=str, default=f'{utils.ORIGIN}/data/processed/fmri/bold')
parser.add_argument('--output_dir', type=str, default=f'{utils.ORIGIN}/data/processed/fmri/decoding')
args = parser.parse_args()

utils.download_dataset("data/processed/behavior")
utils.download_dataset("data/processed/fmri")
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
prefix = args.prefix
input_dir = args.input_dir
output_dir = args.output_dir

for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    # Load voxels
    X = utils.load(f"{input_dir}/{prefix}_sub-{v_sub:04}.pkl")['bold']
    
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
    print(f'Decoding stimuli from the voxels of sub-{v_sub:04}...')
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
        
        print(f'T={t_train} completed')
    
    # due to backward incompatibility of pandas, we serialize data without behavior dataframe (pandas)
    # instead, corresponding behaviors can be recovered using utils.load_behavior()
    utils.save(chan_recon, f"{output_dir}/{prefix}_sub-{v_sub:04}.pickle")