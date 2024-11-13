"""process external datasets
    Zamboni, Ledgeway, McGraw & Schluppeck (2016)
    Fritsche & de Lange (2019)
    Rademaker, Chunharas & Serences (2019)
"""
import os
import tarfile
import numpy as np
import pandas as pd
from scipy.io import loadmat
from dynamic_bias import utils

IN_DIR      = f'{utils.ORIGIN}/data/external'
INTERIM_DIR = f'{utils.ORIGIN}/data/interim'
OUT_DIR     = f'{utils.ORIGIN}/data/processed/behavior'

"""
1. Zamboni, Ledgeway, McGraw, Schluppeck (2016), Proceedings of the Royal Society B: Biological Sciences
    "Do perceptual biases emerge early or late in visual processing? Decision-biases in motion perception"
    Reference-absent condition (N=5)
    URL: https://github.com/schluppeck/zamboni-2016
    Files: subject*-reference-absent.csv

Experiment structure
    Fixation (0.5s) → Reference (0.5s) → Stimulus (1s) → Choice (<2s) → Estimation (<10s)

Dataset variables
    (Delta) Stimulus-Reference (→ evidence)
    (Estimate) (Relative) reproduced stimulus orientation (→ relestim )
"""
data = []
for sid in ['A', 'C', 'D', 'E', 'F']:
    sess = pd.read_csv( f'{IN_DIR}/zamboni2016/subject{sid}-reference-absent.csv', skiprows=1)
    n_trial = len(sess)
    data.append({
        'ID'       : np.repeat(sid, n_trial),
        'evidence' : sess['Delta'].values,
        'relestim' : sess['Estimate'].values,
    })

colnames = ['ID', 'evidence', 'relestim']
data = pd.DataFrame({k: np.concatenate([d[k] for d in data]) for k in colnames})

data.to_csv(f'{OUT_DIR}/zamboni2016.csv', index=False)
print('zamboni2016 data saved')


"""
2. Fritsche & de Lange (2019), Cognition
    "Reference Repulsion Is Not a Perceptual Illusion"
    Experiment 1 (N=24)
    URL : https://data.ru.nl/collections/di/dccn/DSC_3018029.03_140
	Files : Exp1.tar.gz

Experiment structure
    Discrimination boundary (0.75s) → Fixation (0.25s) → Orientation stimulus (1s) → Noise mask (0.5s) → 
    Boundary judgment (self-paced) → Adjustment response (self-paced)

Dataset variables (0-based)
    based on Exp1/Data/data_readme.txt
    (column = 0) Subject number (→ ID)
    (column = 5) Stimulus orientation (→ stim)
    (column = 6) (Absolute) boundary orientation (→ ref )
    (column = 7) Boundary judgment (-1 = ccw; 1 = cw) (→ dm)
    (column = 9) Reproduced stimulus orientation (→ estim)
"""
# unzip and save to interim/ directory
os.makedirs(f"{INTERIM_DIR}/fritsche2019", exist_ok=True)
with tarfile.open(f"{IN_DIR}/fritsche2019/Exp1.tar.gz", "r:gz") as tar:
    tar.extractall(f"{INTERIM_DIR}/fritsche2019")

colnames = ['ID', 'stim', 'ref', 'dm', 'estim']
dtypes = [int, float, float, int, float]
data = []
for sid in range(1,25):
    for sess in [1,2]:
        for blk in range(1,9):
            block = loadmat( f'{INTERIM_DIR}/fritsche2019/Exp1/Data/S{sid}/data_S{sid}_Session_{sess}_Block_{blk}.mat' )
            data.append( {k: [d[0][0][i] for d in block['data']] for i,k in zip([0,5,6,7,9], colnames)} )

data = pd.DataFrame({k: np.concatenate([d[k] for d in data]).astype(d) for k,d in zip(colnames, dtypes)})
data.to_csv(f'{OUT_DIR}/fritsche2019.csv', index=False)
print('fritsche2019 data saved')


"""
3. Rademaker, Chunharas & Serences (2019), Nature Neuroscience
    "Coexisting representations of sensory and mnemonic information in human visual cortex" 
    Behavior experiment (N=17, Supplementary Figure9)
    URL : https://osf.io/dkx6y/wiki/home/
    Files : Data / Behavioral experiment / DataDistRand_*.mat

Experiment structure
    Target (0.2s) → Delay 1 (1.4s) → Distractor (0.2s) → Delay 2 (1.4s) → Estimation (self-paced) → ITI (0.8s-1s)
    4 participants out of 21 were excluded due to dropout per "WM_DistRand_Analysis.m"

Dataset variables (0-based)
    based on WM_DistRand_Analysis.m
    (TheData.TrialStuff.orient_target) Stimulus orientation (→ stim)
    (TheData.TrialStuff.orient_distr) (Absolute) distractor orientation (→ ref )
    (TheData.data[:,0]) Reproduced stimulus orientation (→ estim)
"""
colnames = ['ID', 'stim', 'ref', 'estim']
dtypes = [int, float, float, float]
data = []
for sid in (list(range(1,10)) + list(range(12,20))):
    sess = loadmat( f'{IN_DIR}/rademaker2019/DataDistRand_{sid:02}.mat' )['TheData'][0]
    n_blk = len(sess)
    for i_blk in range(n_blk):
        blk_info  = sess[i_blk][2][0]
        n_trial   = len( blk_info )
        ori_stim  = [ blk_info[i][0] for i in range(n_trial) ]
        ori_dstr  = [ blk_info[i][1] for i in range(n_trial) ]

        data.append({
            'ID'   : np.repeat(sid, n_trial),
            'stim' : np.squeeze( ori_stim ),
            'ref'  : np.array( [d[0,0] if d.size > 0 else np.nan for d in ori_dstr] ),
            'estim': sess[i_blk][3][:,0],
        })

data = pd.DataFrame({k: np.concatenate([d[k] for d in data]).astype(d) for k,d in zip(colnames, dtypes)})
data.to_csv(f'{OUT_DIR}/rademaker2019.csv', index=False)
print('rademaker2019 data saved')
