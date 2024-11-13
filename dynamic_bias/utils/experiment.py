"""experiment structure"""
import numpy as np
from .stats import wrap

def exp_stim_list(radian=False,
                  step=7.5):
    """generate a normative list of unique stimuli for experiment
        stimulus : 0, 7.5, 15, ..., 172.5
        step : step size of stimulus (in degree)
    """
    stim_list = np.arange(0, 180, step=step)
    if radian:
        stim_list = np.deg2rad(2.*stim_list)
    return stim_list


def exp_ref_list(near_only=False,
                 radian=False):
    """generate a normative list of unique (relative) references for experiment
        reference : -4, 0, +4 relative to stimulus
    """
    if near_only:
        ref_list = np.array([-4, 0, 4])
    else:
        ref_list = np.array([-21, -4, 0, 4, 21])
    if radian:
        ref_list = np.deg2rad(2.*ref_list)
    return ref_list


def exp_onset_list(epoch='choice', return_dict=True):
    """generate a normative list of durations for experiment
        choice: 6s after stimulus onset (early) and 12s after stimulus onset (late)
        estimation: 18s
    """
    if epoch == 'choice':
        if return_dict:
            onsets = {'early': 6, 'late': 12}
        else:
            onsets = np.array([6, 12])
    elif epoch == 'estimation':
        onsets = np.array([18])
    return onsets
    

def exp_structure(stimulus='default',
                  reference='default',
                  timing='default',
                  choice=None,
                  dm=None,
                  near_only=False,
                  relative_reference=True,
                  radian=False,
                  ):
    """generate a experiment structure

    normative experiment structure hierarchy
        timing : 1 for early, 2 for late
        reference : -21, -4, 0, +4, +21 relative to stimulus
        stimulus : 0, 7.5, 15, ..., 172.5
        others...
        
    returns
    -------
        exp : dict
            experiment structure
    """

    if stimulus == 'default':
        stimulus = exp_stim_list(radian=radian)
    if reference == 'default':
        reference = exp_ref_list(near_only=near_only, radian=radian)
    if timing == 'default':
        timing = np.array([1, 2])

    names = []
    lists = []
    if timing is not None:
        names.append('timing')
        lists.append(timing)
    if reference is not None:
        names.append('reference')
        lists.append(reference)
    if stimulus is not None:
        names.append('stimulus')
        lists.append(stimulus)
    if choice is not None:  
        names.append('choice')
        lists.append(choice)
    if dm is not None:
        names.append('dm')
        lists.append(dm)
    
    lists = np.meshgrid(*lists, indexing='ij')

    # experiment structure
    exp = {k: v.flatten() for k, v in zip(names, lists)}
    if not relative_reference:
        if radian:
            exp['reference'] = wrap(exp['stimulus'] + exp['reference'])
        else:
            exp['reference'] = wrap(exp['stimulus'] + exp['reference'], period=180.)

    return exp