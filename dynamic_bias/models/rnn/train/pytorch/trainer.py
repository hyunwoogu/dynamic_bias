import numpy as np
import torch
from .model import Model

def initialize_rnn():
    model = Model()
    return model

def initialize_model_performance():
    model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'loss_dm': [], 'loss_em': []}
    return model_performance

def append_model_performance(model_performance, trial_info, Y, Loss, par):
    perf_dm, perf_em = _get_eval(trial_info, Y, par)
    model_performance['loss'].append(Loss['loss'].item())
    model_performance['loss_dm'].append(Loss['loss_dm'].item())
    model_performance['loss_em'].append(Loss['loss_em'].item())
    model_performance['perf_dm'].append(perf_dm)
    model_performance['perf_em'].append(perf_em)
    return model_performance

def print_results(model_performance, iteration, n_print=10):
    if iteration % n_print == 0:
        print_res = 'Iter. {:4d}'.format(iteration)
        print_res += ' | Discrimination Performance {:0.4f}'.format(model_performance['perf_dm'][iteration]) + \
                    ' | Estimation Performance {:0.4f}'.format(model_performance['perf_em'][iteration]) + \
                    ' | Loss {:0.4f}'.format(model_performance['loss'][iteration])
        print(print_res)

def tensorize_hp(hp):
    for k, v in hp.items():
        hp[k] = torch.tensor(v)
    return hp

def tensorize_trial(trial_info, device):
    return {k: torch.tensor(v, device=device) for k, v in trial_info.items()}

def tensorize_model_performance(model_performance):
    tensor_mp = {'perf_dm': torch.tensor(model_performance['perf_dm'], requires_grad=False),
                 'perf_em': torch.tensor(model_performance['perf_em'], requires_grad=False),
                 'loss':    torch.tensor(model_performance['loss'],    requires_grad=False),
                 'loss_dm': torch.tensor(model_performance['loss_dm'], requires_grad=False),
                 'loss_em': torch.tensor(model_performance['loss_em'], requires_grad=False)}
    return tensor_mp

def _get_eval(trial_info, output, par):
    argoutput = torch.argmax(output['dm'], dim=2).numpy()
    perf_dm   = np.mean(np.array([argoutput[t, :] == ((trial_info['reference_ori'].numpy() > 0)) for t in par['design_rg']['decision']]))

    cenoutput = torch.nn.functional.softmax(output['em'], dim=2).detach().numpy()
    post_prob = cenoutput
    post_prob = post_prob / (np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)  # Dirichlet normalization
    post_support = np.linspace(0, np.pi, par['n_ori'], endpoint=False) + np.pi / par['n_ori'] / 2
    pseudo_mean = np.arctan2(post_prob @ np.sin(2 * post_support),
                             post_prob @ np.cos(2 * post_support)) / 2
    estim_sinr = (np.sin(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_cosr = (np.cos(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_mean = np.arctan2(estim_sinr, estim_cosr) / 2
    perf_em = np.mean(np.cos(2. * (trial_info['stimulus_ori'].numpy() * np.pi / par['n_ori'] - estim_mean)))

    return perf_dm, perf_em