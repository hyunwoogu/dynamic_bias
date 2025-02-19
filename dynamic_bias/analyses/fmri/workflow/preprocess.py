"""load fmriprep data, extract ROI voxels, and save the results

# glitches
sub-0015 : ses-2 run-1
    the participant reported discomfort due to the presence of mosquitoes in the scanner, which led to early termination of the scan.

sub-0018 : ses-2 run-1
    due to technical issues, less frames were acquired, so the run was not included in the analysis.

sub-0018 : ses-2 run-6
    due to technical issues, experiment code generated shorter ITIs, so the run was not included in the analysis.

sub-0027 : ses-4 run-5
    the participant was obviously drowsy during the scan, which led to early termination of the scan.

# example usage
    python preprocess.py --exclude sub-0018 sub-0058 sub-0107
    python preprocess.py --participants sub-0018 --exclude_ses 2 --exclude_run 1
    python preprocess.py --participants sub-0058 --path_snr data/processed/fmri/snr/evc_sub-0058.pkl
    python preprocess.py --participants sub-0107 --TR_hirf 1.5 --avg_freq_hirf 0
"""
import os
import re
import argparse
import nibabel as nib
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import zscore
from dynamic_bias import utils


# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--fmriprep_dir',  type=str, default=f'{utils.ORIGIN}/data/raw/ds005381/derivatives/fmriprep')
parser.add_argument('--mask_dir',      type=str, default=f'{utils.ORIGIN}/data/processed/fmri/mask')
parser.add_argument('--output_dir',    type=str, default=f'{utils.ORIGIN}/data/processed/fmri/bold')
parser.add_argument('--space',         type=str, default='T1w')              # 'T1w' or 'MNI'
parser.add_argument('--participants',  type=str, nargs='+', default=['all']) # 'all' or list of participant ids
parser.add_argument('--exclude',       type=str, nargs='+', default=[])      # exclude participants
parser.add_argument('--mask_specific', type=int, default=1)                  # whether to use participant-specific mask
parser.add_argument('--mask_prefix',   type=str, default='evc')              # '{mask_prefix}_sub-{xxxx}.nii.gz'
parser.add_argument('--output_prefix', type=str, default='evc')              # '{output_prefix}_sub-{xxxx}.nii.gz'
parser.add_argument('--n_cpus',        type=int, default=1)                  # number of cpus for parallel processing
parser.add_argument('--exclude_ses',   type=int, nargs='+', default=[])      # exclude sessions
parser.add_argument('--exclude_run',   type=int, nargs='+', default=[])      # exclude runs
parser.add_argument('--run_hirf',      type=int, default=2)                  # run number for HIRF runs
parser.add_argument('--TR_hirf',       type=float, default=2.0)              # TR in seconds for HIRF runs
parser.add_argument('--TR_task',       type=float, default=2.0)              # TR in seconds for task runs
parser.add_argument('--TR_start',      type=int, default=2)                  # start index for the number of timepoints (in TRs)
parser.add_argument('--compute_snr',   type=int, default=1)                  # compute SNR
parser.add_argument('--path_snr',      type=str, default=None)               # path to SNR data
parser.add_argument('--min_snr',       type=float, default=2.0)              # minimum SNR for valid data. Negative values denote not using SNR.
parser.add_argument('--avg_freq_hirf', type=float, default=1/8)              # average frequency for SNR calculation
parser.add_argument('--max_freq_hirf', type=float, default=0.006)            # upper bound for DCT-based frequency filtering
parser.add_argument('--max_freq_task', type=float, default=0.008)            # upper bound for DCT-based frequency filtering
parser.add_argument('--mask_labels',   type=int, nargs='+', default=list(range(1,13))) # mask labels
args = parser.parse_args()

config = {
    'fmriprep_dir'  : Path(args.fmriprep_dir),
    'mask_dir'      : Path(args.mask_dir),
    'output_dir'    : Path(args.output_dir),
    'space'         : args.space,
    'participants'  : args.participants,
    'exclude'       : args.exclude,
    'mask_specific' : args.mask_specific,
    'mask_prefix'   : args.mask_prefix,
    'output_prefix' : args.output_prefix,
    'n_cpus'        : args.n_cpus,
    'exclude_ses'   : args.exclude_ses,
    'exclude_run'   : args.exclude_run,
    'run_hirf'      : args.run_hirf,
    'TR_hirf'       : args.TR_hirf,
    'TR_task'       : args.TR_task,
    'TR_start'      : args.TR_start,
    'mask_labels'   : args.mask_labels,
    'compute_snr'   : args.compute_snr,
    'path_snr'      : args.path_snr,
    'min_snr'       : args.min_snr,
    'avg_freq_hirf' : args.avg_freq_hirf,
    'max_freq_hirf' : args.max_freq_hirf,
    'max_freq_task' : args.max_freq_task,
    'max_FD'        : None,
    'zscore_bold'   : True,
    'confounds'     : ['white_matter', 'csf', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
    'n_TRs'         : 14,
    'n_tasks'       : 12,
}

def compute_percent(bold):
    """convert BOLD data to percent signal change.
        current implementation uses .copy(), which may incur addtional memory usage due to data duplication.
    """
    percent_bold = (bold / np.nanmean(bold.copy(), axis = 0, keepdims = True) * 100) - 100
    return percent_bold

def compute_dct_basis_set(length, freq_cutoff, TR):
    """create a DCT basis set for high-pass filtering."""
    ks = ( np.arange(length*TR*freq_cutoff*2) / 2)[1:]
    t  = np.arange(0, length)/length * (2*np.pi)
    ks = ks.reshape(-1, 1)
    return np.cos(ks * t).T

def compute_nuisance_regressors(confounds, conf_labels, freq_cutoff, TR, include_constant=True, dtype=np.float32):
    """Generate nuisance regressors including confounds and high-pass filtering."""
    conf   = confounds[conf_labels].fillna(0).to_numpy()
    conf_z = zscore(conf, axis=0)
    dct    = compute_dct_basis_set(len(confounds), freq_cutoff, TR)
    regressors = np.hstack([conf_z, dct])
    if include_constant:
        regressors = np.hstack([regressors, np.ones((len(confounds), 1))])
    return regressors.astype(dtype)

def compute_snr(data, TR, signal_freq=1/24, avg_freq=config['avg_freq_hirf']):
    """calculate SNR for the given data."""
    freq = np.fft.fftfreq(data.shape[-1], TR)
    signal_idx = np.argmin(np.abs(freq - signal_freq))
    xf = np.fft.fft(data, axis=-1)
    signal_amplitude = np.abs(xf[:, signal_idx])
    avg_amplitude = np.mean(np.abs(xf[:, freq > avg_freq]), axis=-1)
    snr = np.divide(signal_amplitude, avg_amplitude, out=np.zeros_like(avg_amplitude), where=avg_amplitude > 0)
    return snr

def compute_residual(X, y):
    idx   = ~( np.isnan(X).any(axis=-1) | np.isnan(y) )
    w     = np.linalg.lstsq(X[idx], y[idx], rcond = None)[0]
    resid = y - X@w
    return resid

def compute_residuals(X, Y, n_cpus = 1):
    resids = Parallel(n_jobs=n_cpus)(
        delayed(compute_residual)(X, Y[:,i]) for i in range(Y.shape[1])
    )
    resids = np.stack(resids, axis=1).astype(np.float32)
    return resids

def info(x, which):
    """extract session, run, or task information from the filename"""
    if isinstance(x, Path):
        x = x.name
    if which == 'ses':
        match = re.search(r'ses-(\d+)', x)
        return int(match.group(1)) if match else None
    elif which == 'run':
        match = re.search(r'run-(\d+)', x)
        return int(match.group(1)) if match else None
    elif which == 'task':
        match = re.search(r'task-(\S+)_run', x)
        return match.group(1) if match else None
    
def match( query_fn, fns, which=['ses', 'run', 'task'] ):
    """match the session, run, and task information of a query against a list of filenames"""
    if isinstance(query_fn, Path):
        query_fn = query_fn.name
    x_info = {k: info(query_fn, k) for k in which}
    return next( (fn for fn in fns if all(x_info[k] == info(fn, k) for k in which)), None )


#
os.makedirs(config['output_dir'], exist_ok=True)

if 'all' in config['participants']:
    print("loading all participants...")
    behavior = utils.load_behavior()
    config['participants'] = [ f'sub-{sid:04d}' for sid in np.unique(behavior['ID']) ]

if config['exclude']:
    config['participants'] = sorted( list(set(config['participants']) - set(config['exclude'])) )

for subject in config['participants']:
    sub_id  = int(subject[-4:])

    # exception handling
    sub_dir = next( config['fmriprep_dir'].glob(f"sub-{sub_id:04d}"), None )
    if sub_dir is None:
        print(f"Files missing for subject {subject}. Skipping...")
        continue

    # 
    if config['mask_specific']:
        mask_fn = Path(config['mask_dir']) / f"{config['mask_prefix']}_sub-{sub_id:04d}.nii.gz"
    else:
        mask_fn = Path(config['mask_dir']) / f"{config['mask_prefix']}.nii.gz"
    bold_fns = sorted( sub_dir.glob( f"*/*/*/func/*{config['space']}*bold.nii.gz" ) )
    conf_fns = sorted( sub_dir.glob( '*/*/*/func/*confounds*.tsv' ) )

    # load and process the mask
    mask   = nib.load( mask_fn ).get_fdata()
    mask   = np.isin( mask, config['mask_labels'] )
    coords = np.argwhere( mask )

    if config['compute_snr']:
        if config['path_snr'] is not None:
            snr_data   = utils.load( config['path_snr'] )
            snr_coords = snr_data['coords']
            snr_index  = np.array( [np.where(np.all(snr_coords == c, axis=-1))[0][0] for c in coords] )
            snr        = snr_data['snr'][snr_index]
        else:
            # [1-1] load retinotopy (HIRF) BOLD data
            hirf_fn   = [f for f in bold_fns if info(f.name, 'task') == 'Retino']
            hirf_fn   = [f for f in hirf_fn if info(f.name, 'run') == config['run_hirf']][-1]
            hirf_bold = nib.load( hirf_fn ).get_fdata().astype(np.float32) [ mask ]
            hirf_conf = pd.read_csv( match( hirf_fn, conf_fns ), delimiter='\t' )

            # [1-2] process retinotopy (HIRF) BOLD data
            hirf_reg  = compute_nuisance_regressors(hirf_conf, config['confounds'], config['max_freq_hirf'], config['TR_hirf'], dtype=np.float64)
            hirf_bold = compute_residuals(hirf_reg, hirf_bold.T, n_cpus = config['n_cpus'])        
            snr       = compute_snr( hirf_bold.T, config['TR_hirf'] )
    
    # [2-1] load task BOLD data
    task_fns = [f for f in bold_fns if info(f.name, 'task') == 'DET']
    task_fns.sort( key=lambda x: (info(x.name, 'ses'), info(x.name, 'run')) )

    # [2-2] process task BOLD data
    task_bolds = []
    for task_fn in task_fns:
        ses = info(task_fn.name, 'ses')
        run = info(task_fn.name, 'run')
        if (ses,run) in zip(config['exclude_ses'], config['exclude_run']):
            print( f"skipping {task_fn.name} due to exclusion" )
            continue
        else:
            print( f'processing {subject} ses-{ses} run-{run}...' )

        task_bold = nib.load( task_fn ).get_fdata().astype(np.float32) [ mask ]         # [n_voxels, n_timepoints]
        task_bold = compute_percent( task_bold.T )                                      # [n_timepoints, n_voxels]
        task_conf = pd.read_csv( match( task_fn, conf_fns ), delimiter='\t' )

        if len(task_conf) != int( config['n_TRs']*config['n_tasks'] ):
            print( f"skipping {task_fn.name} due to mismatch in the number of TRs" )
            continue

        task_reg  = compute_nuisance_regressors( task_conf, config['confounds'], config['max_freq_task'], config['TR_task'] )
        task_bold = compute_residuals(task_reg, task_bold, n_cpus = config['n_cpus']).T # [n_voxels, n_timepoints]

        # valid data selection
        idx_valid = np.ones(len(task_bold), dtype=bool)
        if config['max_FD'] is not None:
            idx_valid = task_conf['framewise_displacement'] > config['max_FD']
        if config['compute_snr'] & (config['min_snr'] >= 0):
            idx_valid = snr >= config['min_snr']
        task_bold = task_bold[idx_valid]

        if config['zscore_bold']:
            task_bold = zscore(task_bold, axis=-1, nan_policy='omit')

        task_bolds.append( task_bold )

    # concatenate and save the results
    out = {'bold' : np.concatenate(task_bolds, axis=-1).reshape(sum(idx_valid), -1, config['n_TRs'])[...,config['TR_start']:] }
    out['coords'] = coords[idx_valid]

    if config['compute_snr']:
        out['snr'] = snr

    out_fn = Path( config['output_dir'] ) / f"{config['output_prefix']}_sub-{sub_id:04d}.pkl" 
    utils.save( out, out_fn )
    print( f"saved {out_fn}" )