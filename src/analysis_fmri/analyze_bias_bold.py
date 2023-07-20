"""analyze the decision and orientation biases
"""
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append('../..')
from src import utils
utils.download_dataset("data/processed/behavior")
utils.download_dataset("data/processed/fmri")

path_output = f"{utils.ORIGIN}/data/processed/fmri/decoding/"
behavior = pd.read_csv(f"{utils.ORIGIN}/data/processed/behavior/behavior.csv")
behavior = behavior[(~np.isnan(behavior['choice'])) & (~np.isnan(behavior['error']))]

# load dataset
pop_behavior = []
pop_channel = []  # population data of reconstructed channel responses
for i_sub, v_sub in enumerate(np.unique(behavior.ID)):
    with open(path_output+f'decoding_sub-{v_sub:04}.pickle', 'rb') as f: 
        chan_recon = pickle.load(f)
    pop_behavior.append(chan_recon['behav'])
    pop_channel.append(chan_recon['chan'])
pop_behavior = pd.concat(pop_behavior)
pop_channel  = np.concatenate(pop_channel, axis=1)

# bootstrapping orientation bias 
n_boot = 10000
dyn_stim_boot = np.zeros((24,12,2,n_boot)) # stim, tr, timing
labels = np.linspace(0,np.pi,120,endpoint=False)
for i_timing, v_timing in enumerate([1,2]):
    for i_s, v_s in enumerate(np.arange(180,step=7.5)):
        idx = (pop_behavior.Timing == v_timing) & (pop_behavior.stim == v_s) 
        for t in range(12):
            for i_boot in range(n_boot):
                idx_boot = np.random.choice(sum(idx), sum(idx), replace=True)
                _m = np.mean(pop_channel[t,idx][idx_boot],axis=0)
                dyn_stim_boot[i_s,t,i_timing,i_boot] = utils.pop_vector_decoder(_m, labels, unit='radian')
            print(i_timing,i_s,t, 'bootstrapped')

with open(f"{utils.ORIGIN}/data/outputs/fmri/bootstrap_bold_stimulus_conditioned.pickle", 'wb') as f:
    pickle.dump(dyn_stim_boot,f)


# bootstrapping decision bias
stim_shift = (pop_behavior.stim / 1.5).astype(int).to_numpy()
pop_channel_shift = np.nan*pop_channel
for t in range(12):
    for i_vec, vec in enumerate(pop_channel[t,:,:]):
        pop_channel_shift[t, i_vec, :] = np.roll(vec, -stim_shift[i_vec])

dyn_cond_boot = np.zeros((2,12,2,n_boot)) # timing, dm, trs 
labels = np.linspace(0,np.pi,120,endpoint=False)
for i_timing, v_timing in enumerate([1,2]):
    for i_c, v_c in enumerate([1,2]):
        for t in range(12):
            idx = (pop_behavior.Timing==v_timing) & (pop_behavior.choice==v_c) & np.isin(pop_behavior.ref, [-4,0,4])
            for i_boot in range(n_boot):
                idx_boot = np.random.choice(sum(idx), sum(idx), replace=True)
                dyn_cond_boot[i_timing,t,i_c,i_boot] = utils.pop_vector_decoder(np.mean(pop_channel_shift[t,idx][idx_boot],axis=0), labels, unit='radian')
            print(i_timing,i_c,t, 'bootstrapped')

utils.mkdir(f"{utils.ORIGIN}/data/outputs/fmri")
with open(f"{utils.ORIGIN}/data/outputs/fmri/bootstrap_bold_decision_conditioned.pickle", 'wb') as f:
    pickle.dump(dyn_stim_boot,f)