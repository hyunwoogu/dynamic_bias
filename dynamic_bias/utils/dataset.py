"""
Utility functions for dataset.
"""
import os
import pickle
import operator
import subprocess
import numpy as np
import pandas as pd
from functools import reduce
from pathlib import Path

PATH = {}
PATH['data'] = {}
# PATH['data']['external'] # Not provided. See analyses/behavior/workflow/preprocess_external.py
PATH['data']['processed'] = {}
PATH['data']['processed']['behavior'] = "https://www.dropbox.com/scl/fi/j6l05vrhgcg0cijd7r5fv/behavior.zip"
PATH['data']['processed']['fmri']     = "https://www.dropbox.com/scl/fi/tmbxsthcwunykrerwfcc6/fmri.zip"
PATH['data']['outputs'] = {}
PATH['data']['outputs']['behavior']   = "https://www.dropbox.com/scl/fi/x69ocf8mph7hwx38m6boe/behavior.zip"
PATH['data']['outputs']['fmri']       = "https://www.dropbox.com/scl/fi/7o9mad0c29coed9ch0zxx/fmri.zip"
PATH['data']['outputs']['ddm']        = "https://www.dropbox.com/scl/fi/i4i1ilguykfs927ua4j4i/ddm.zip"
PATH['data']['outputs']['rnn']        = "https://www.dropbox.com/scl/fi/si7313vucjrgl3rqqgmfb/rnn.zip"
PATH['models'] = {}
PATH['models']['ddm'] = {}
PATH['models']['ddm']['full']         = "https://www.dropbox.com/scl/fi/f2kw1rdgmv4ijy7exkmud/full.zip"
PATH['models']['ddm']['reduced']      = "https://www.dropbox.com/scl/fi/fhjzjpo0fjplk2i209ymg/reduced.zip"
PATH['models']['ddm']['null']         = "https://www.dropbox.com/scl/fi/3as06y0ztpew3qvbwdzvx/null.zip"
PATH['models']['rnn'] = {}
PATH['models']['rnn']['homogeneous']          = "https://www.dropbox.com/scl/fi/5h7zs4bbwy3y2h5vzo1hw/homogeneous.zip"
PATH['models']['rnn']['heterogeneous']        = "https://www.dropbox.com/scl/fi/9kjsol207yv8a964ozs75/heterogeneous.zip"
PATH['models']['rnn']['heterogeneous_emonly'] = "https://www.dropbox.com/scl/fi/pqwgttyq6cggw1y11i1v6/heterogeneous_emonly.zip"
PATH['models']['rnn']['heterogeneous_d2e_ablation'] = "https://www.dropbox.com/scl/fi/7tn82mum6w23trqdi5k0q/heterogeneous_d2e_ablation.zip"
ORIGIN = str(Path(os.path.abspath(__file__)).parent.parent.parent.absolute())

def getFromDict(dataDict, mapList):
    """get value from nested dictionary
    https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    """
    return reduce(operator.getitem, mapList, dataDict)

def mkdir(path):
    """make directory"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save(file, path):
    """save file"""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be str or Path, not {type(path)}")
    
    mkdir(Path(path).parent)
    with open(path, 'wb') as f:
        pickle.dump(file, f)

def load(path):
    """load file"""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be str or Path, not {type(path)}")
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_behavior(exclude_nans=True, verbose=True, sub_id=None):
    """load behavior data"""
    download_dataset("data/processed/behavior", verbose=verbose)
    behavior = pd.read_csv(f"{ORIGIN}/data/processed/behavior/behavior.csv")
    if exclude_nans:
        behavior = behavior[behavior['choice'].notna() & behavior['error'].notna()]
    if sub_id is not None:
        behavior = behavior[behavior['ID']==sub_id]
    return behavior

def load_bold(return_behavior=False, verbose=True, prefix='evc'):
    """load bold decoding data in [t, n_channel, n_trials] format
    """
    behavior = load_behavior(verbose=False)
    download_dataset("data/processed/fmri", verbose=verbose)

    chans = []
    for v_sub in np.unique(behavior.ID):
        chan = load(f"{ORIGIN}/data/processed/fmri/decoding/{prefix}_sub-{v_sub:04}.pickle")
        chans.append(chan)
    chans = np.concatenate(chans, axis=1).transpose((0,2,1))

    if return_behavior:
        return chans, behavior
    return chans

def set_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def recursive_list(d):
    nested_list = []
    for value in d.values():
        if isinstance(value, dict):
            nested_list.append(recursive_list(value))
        else:
            nested_list.append(value)
    return nested_list

def download_dataset(path, cmd='curl', verbose=True):
    """download dataset from the path"""
    path_strip  = path.strip('/')
    path_split  = path_strip.split('/')
    path_parent = '/'.join(path_split[:-1])
    data_url    = getFromDict(PATH, path_split)
    
    if verbose: print(f"downloading {path_strip}...")
    if os.path.exists(f"{ORIGIN}/{path_strip}"):
        if verbose: print(f"{path_strip} already exists. Skipping download...")
    else:
        mkdir(f"{ORIGIN}/{path_parent}")
        if cmd == 'wget':
            command  = f"wget {data_url} -P {ORIGIN}/{path_parent} -c -nc;"
            command += f"unzip -o {ORIGIN}/{path_strip}.zip -d {ORIGIN}/{path_parent};"
        elif cmd == 'curl':
            command  = f"curl -L -o {ORIGIN}/{path_strip}.zip {data_url};"
            command += f"unzip -o {ORIGIN}/{path_strip}.zip -d {ORIGIN}/{path_parent};"
        command += f"rm {ORIGIN}/{path_strip}.zip"
        if verbose: print(command)
        subprocess.run(command, capture_output=True, shell=True)
        if verbose: print(f"downloaded {path_strip}.")

def index_dict(data, index, nested=False):
    """index a dictionary of arrays using a shared index"""
    indexed_data = {}
    for key, value in data.items():
        if nested and isinstance(value, dict):
            # recursively handle nested dictionaries
            indexed_data[key] = index_dict(value, index, nested=True)
        else:
            indexed_data[key] = value[index]
    return indexed_data