"""Euler-Maruyama simulation method for drift-diffusion model
"""
from .... import utils
import numpy as np
from scipy.stats import vonmises

# parameters
params = {
    # hyperparameters
    'T'      : 18,            # total time of interest  
    'ds'     : np.pi/48.,     # stimulus space discretization
    'stim'   : np.linspace(0,2*np.pi, num=24, endpoint=False), # stimulus space
    'relref' : np.array([-21,-4,0,4,21]) * np.pi/90.,          # reference space
    't_dm'   : np.array([6,12]), # decision-making time
    'thres'  : np.pi/10,      # threshold that divides near & far references
    'sig'    : 0.2,           # smoothness of decision-conditioning mask
    'freq'   : 10,            # number of time discretization for 1sec (Hz)
    'n_trial': 100,           # number of trials per combination of conditions

    # parameters (default placeholder) (degree-space)
    'w_K'    : 0.5,
    'w_E'    : 1.5,
    'w_P'    : 0.0,
    'w_D'    : 0.0,
    'w_r'    : 0.5,
    'w_a'    : 0.5,
    'lam'    : 0.1,
    's'      : 0.5,

    # encoding and drift functions (vectorized)
    'K'      : lambda x: 0,  # drift function
    'F'      : lambda x: x,  # encoding function
    'F_inv'  : lambda x: x,
}


class Euler:
    def __init__(self, params=params):
        self.params = params

    def run(self, data=None, verbose=False, 
            returns=['encoding', 'memory', 'production', 'decision'],
            return_data=False, return_time=False):
        """Simulate drift-diffusion dynamics using Euler-Maruyama method
            dmT: timing of decision-making

        inputs
        ------
            data [dict] : dictionary of stimulus, reference, and decision-making time
                t_dm   [n_data] : decision-making time
                stim   [n_data] : stimulus (direction-radian)
                relref [n_data] : relative reference (direction-radian)
            verbose [bool] : verbosity
            return_data [bool] : return data dictionary

        returns
        -------
            res_e  [n_data, n_trial] : encoding results (direction-radian)
            res_m  [n_data, n_trial, freq*T] : memory results (direction-radian)
            res_p  [n_data, n_trial] : production results (direction-radian)
            res_dm [n_data, n_trial] : decision-making results (captial C ⋹ {1,-1})
        """

        # relevant functions
        K     = self.params['K'] # drift function
        F     = self.params['F'] # stimulus-to-measurement function
        F_inv = self.params['F_inv'] # measurement-to-stimulus function

        # relevant parameters (converted into direction-radians)
        lam = self.params['lam']
        w_K = utils.ori2dir(self.params['w_K'])
        w_E = utils.ori2dir(self.params['w_E'])
        w_P = utils.ori2dir(self.params['w_P'])
        w_D = utils.ori2dir(self.params['w_D'])
        w_r = utils.ori2dir(self.params['w_r'])
        w_a = utils.ori2dir(self.params['w_a'])

        # time and space discretization
        nt  = int(self.params['freq']*self.params['T']) # number of time points
        ts  = np.linspace(0, self.params['T'], nt) # time space
        dt  = ts[1] - ts[0]

        # results placeholder
        if data is None:
            comb = np.array( 
                np.meshgrid(
                    self.params['t_dm'], self.params['stim'], self.params['relref'], indexing='ij'
                )
            ).reshape(3,-1)
            data = {k: comb[i] for i,k in enumerate(['t_dm', 'stim', 'relref'])}

        # results placeholder
        n_data = len(data['stim'])
        stim_wrap = utils.wrap(data['stim'])
        res_e  = utils.nan((n_data, self.params['n_trial']))      # encoding
        res_m  = utils.nan((n_data, self.params['n_trial'], nt))  # memory
        res_dm = utils.nan((n_data, self.params['n_trial']))      # decision

        # encoding simulation
        if verbose: print('Simulating encoding...')
        p_e = np.sqrt(1. / np.clip(w_E, utils.EPS, None)) # encoding precision
        for i_theta, v_theta in enumerate(stim_wrap):
            m0 = vonmises.rvs(loc=F(v_theta), kappa=p_e, size=self.params['n_trial'])
            res_e[i_theta] = utils.wrap(F_inv(m0 % (2*np.pi)))

        # memory and decision simulation
        if verbose: print('Simulating memory and decision-making...')
        for i_data, (v_t, v_s, v_r) in enumerate( zip(data['t_dm'], data['stim'], data['relref']) ):
            dm_idx = self.get_time_index(v_t) # closest time index to DM timing

            for i_t, _ in enumerate(ts): 
                if i_t == 0: 
                    # encoding (memory inherits encoding)
                    res_m[i_data, :, i_t] = res_e[i_data, :]
                else:
                    # memory dynamics
                    _th  = utils.wrap(res_m[i_data, :, i_t-1])
                    _dth = w_K*K(_th)*dt + w_D*np.sqrt(dt)*np.random.normal(size=self.params['n_trial'])
                    res_m[i_data, :, i_t] = _th + _dth

                if i_t == dm_idx:
                    # decision-making
                    m_r = v_r + np.random.normal(scale=self.params['sig'], size=self.params['n_trial'])
                    dist = utils.wrap(res_m[i_data, :, i_t] - v_s - m_r)
                    res_dm[i_data, :] = np.where(dist > 0, 1., -1.)

                    if abs(v_r) > self.params['thres']:
                        res_m[i_data,:,i_t] += w_r*(utils.dir2ori(v_r)) # only reference attraction in far
                    else:
                        res_m[i_data,:,i_t] += w_a*res_dm[i_data,:] + w_r*(utils.dir2ori(v_r))

            if verbose: print(f'{i_data+1}/{n_data} completed')
        res_m = utils.wrap(res_m)

        # production simulation
        if verbose: print('Simulating production and lapse...')
        res_p = res_m[:, :, -1]
        res_p = res_p + w_P*np.random.normal(size=res_p.shape)
        res_p = utils.wrap(res_p)

        # lapse
        if lam > 0:
            lap_idx = np.random.choice(2, size=res_dm.shape, p=[1.-lam*2.,lam*2.])
            res_dm[lap_idx==1] = np.random.choice([-1.,1.], size=np.sum(lap_idx))

        res = {}
        for k,v in zip(['encoding', 'memory', 'production', 'decision'], 
                       [res_e, res_m, res_p, res_dm]):
            if k in returns:
                res[k] = v
        del res_e, res_m, res_p, res_dm

        if return_data:
            res.update({'data': data})
        
        if return_time:
            res.update({'time': ts})
        
        return res
    
    def get_time_index(self, t):
        """get the closest time index to the given time point
        """
        nt  = int(self.params['freq']*self.params['T']) 
        ts  = np.linspace(0, self.params['T'], nt)
        index = np.argmin(np.abs(t - ts[...,None]), axis=0)

        if np.isscalar(t):
            return index.item()
        
        return index