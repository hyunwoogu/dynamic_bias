"""combine behavior info from openneuro data into csv format
"""
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dynamic_bias import utils

# make argparser
parser = argparse.ArgumentParser()
parser.add_argument('--openneuro_dir', type=str)
parser.add_argument('--out_dir', type=str, default=f'{utils.ORIGIN}/data/processed/behavior')
args = parser.parse_args()

openneuro_dir = Path(args.openneuro_dir)
out_dir       = Path(args.out_dir)
utils.mkdir(out_dir)

# 
run_key  = lambda f: int( str(f).split('run-')[1].split('_')[0] ) # key for natural-sorting run number
openneuro_subs = sorted( sub.name for sub in openneuro_dir.glob('sub-*') )
openneuro_list = {}
for i_sub, v_sub in enumerate(openneuro_subs):
    openneuro_list[v_sub] = {}
    ses_lst = sorted( ses.name for ses in (openneuro_dir / v_sub).glob('ses-*') if ses.name != 'ses-1' )

    for v_ses in ses_lst:
        file_dir = openneuro_dir / v_sub / v_ses / 'func'
        file_lst = sorted( file_dir.glob('sub-*DET*events.tsv'), key=run_key )
        openneuro_list[v_sub][v_ses] = [str(fn) for fn in file_lst]

# 
def convert_data(_filename):
    _events  = pd.read_csv(_filename, sep='\t')    
    _id      = int(re.search('sub-(.*)_ses',    _filename.rsplit('/',1)[-1]).group(1))
    _ses     = int(re.search('ses-(.*)_task',   _filename.rsplit('/',1)[-1]).group(1))
    _run     = int(re.search('run-(.*)_events', _filename.rsplit('/',1)[-1]).group(1))
    _stim    = _events.value_stimulus[np.arange(84,step=7)]
    _timing  = (_events.trial_type[np.arange(84,step=7)] == 'late') * 1 + 1
    _ref     = _events.value_reference[np.arange(2,84,step=7)] 
    _choice  = _events.response_decision[np.arange(2,84,step=7)] * (-0.5) + 1.5
    _correct = ((_ref > 0) == (_choice-1)) * 1
    _correct[_ref==0] = np.nan
    _rt      = _events.response_time[np.arange(2,84,step=7)] 
    _esti    = _events.response_estimation[np.arange(5,84,step=7)] 
    _error   = (_esti.to_numpy() - _stim.to_numpy() - 90.) % 180. - 90.
    return pd.DataFrame({
        'ID'       : _id,                  # participant ID 
        'ses'      : _ses,                 # session number
        'run'      : _run,                 # run number
        'trial'    : np.arange(1,13),      # trial number
        'stim'     : _stim.to_numpy(),     # stimulus orientation (in degrees)
        'Timing'   : _timing.to_numpy(),   # discrimination task timing (early=1, late=2)
        'ref'      : _ref.to_numpy(),      # relative reference orientation (in degrees)
        'choice'   : _choice.to_numpy(),   # discrimination task choice (CW=1, CCW=2)
        'DM_correct': _correct.to_numpy(), # discrimination task correctness (NaN for reference=0)
        'DM_RT'    : _rt.to_numpy(),       # discrimination task response time (in seconds)
        'esti'     : _esti.to_numpy(),     # estimation task response (in degrees)
        'error'    : _error,               # estimation task error (in degrees)
    })

# save
behav = {}
for sub in openneuro_subs:
    behav_sub = []
    for v in openneuro_list[sub].values():
        for vv in v:
            print(vv)
            behav_sub.append( convert_data(vv) )
    if behav_sub:  
        behav[sub] = pd.concat(behav_sub)    
behav_all = pd.concat(behav)
behav_all.to_csv(out_dir / 'behavior.csv', index=False)