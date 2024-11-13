import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from .... import utils

class PsychometricFunction:
    """psychometric function class"""
    def __init__(self, 
                 link='gaussian', 
                 pse=None,
                 sequential_inequality=None, 
                 return_success=False,
                 **kwargs
                 ):
        """initialize psychometric function class

        inputs
        ----------
            link : {'gaussian', 'gaussian_pse'}, str
                link function of psychometric function
                'gaussian'    : Gaussian psychometric function with lapse = guess
                'gaussian_pse': Gaussian psychometric function with PSE modulated by stimulus
            pse  : (function) -> float, optional
                point-of-subjective-equality function. Only used when link='gaussian_pse'
            sequential_inequality : list, optional
                list of parameter names for sequential inequality constraints
                e.g. ['m', 's'] for m1<=m2<=... and s1<=s2<=...
            return_success : bool, optional
                whether to return success of fitting
        """
        self.link    = link
        self.pse     = pse
        self.si      = sequential_inequality
        self.success = return_success

        if self.link == 'gaussian':
            self.param_names = ['m', 's', 'lam']
        elif self.link == 'gaussian_pse':
            self.param_names = ['w', 's', 'lam']


    def __call__(self, x, **kwargs):
        return self.predict(x, **kwargs)
        

    def predict(self, x, **kwargs):
        """forward pass of psychometric functions"""
        if self.link == 'gaussian':
            """Gaussian psychometric function with lapse = guess"""
            m, s, lam = kwargs['m'], kwargs['s'], kwargs['lam']
            return lam + norm.cdf(x, loc=m, scale=s) * (1.-lam*2)
        
        if self.link == 'gaussian_pse':
            """Gaussian psychometric function with PSE modulated by stimulus"""
            assert self.pse is not None, 'PSE function should be specified'
            w, s, lam = kwargs['w'], kwargs['s'], kwargs['lam']
            pse = self.pse(kwargs['stim'])
            return lam + norm.cdf(x, loc=w*pse, scale=s) * (1.-lam*2)
        

    def sequential_inequality(self, params, direction='fwd'):
        """transform parameters for sequential inequality constraints (increasing)
            only valid if every parameter is nonnegative
            'fwd': from (m1,m2,...) to (m1,m2-m1,...)
            'bwd': from (m1,m2-m1,...) to (m1,m2,...)
        """
        _par = params.copy()
        if direction == 'fwd':
            for p in self.si:
                _par[:, self.param_names.index(p)] = np.cumsum(_par[:, self.param_names.index(p)])
        elif direction == 'bwd':
            for p in self.si:
                _par[:, self.param_names.index(p)] = np.diff(_par[:, self.param_names.index(p)], prepend=0)
        return _par
    

    @property
    def slope(self):
        """Gaussian psychometric function slopes = (1-2λ) /( σ*sqrt(2π) )"""
        assert self.link in ['gaussian', 'gaussian_pse'], 'Only valid for Gaussian psychometric functions.'
        _slope = 1. / (np.sqrt(2*np.pi)*self.fitted_params[...,self.param_names.index('s')])
        _slope = _slope * (1.-2.*self.fitted_params[...,self.param_names.index('lam')])
        return _slope


    def nll(self, params, data, **kwargs):
        """negative log likelihoods (loss function) for psychometric functions

        inputs
        ----------
            params   : (n_params_total,) array-like
                parameters of psychometric functions in total. e.g. [m1, s1, lam1, m2, s2, lam2, ...]
                n_params_total = n_params*n_cond

            data     : {'evidence', 'choice', ('cond'), ('stim'),}, dict
                evidence : (n_trial,) evidence (in degrees)
                choice   : (n_trial,) choice (1 for cw and 0 for ccw)
                cond     : (n_trial,) condition for fitting psychometric functions (ex. Timing).
                                      # unique values = n_cond. None if only one condition. 
                stim     : (n_trial,) stimulus orientations (in degrees)
                
            seq_ineq : array-like
                
        returns
        -------
            nll : float
                negative log likelihood
        """
        # unpack data
        evidence = data['evidence']
        choice   = data['choice']
        cond     = data.get('cond', None)
        stim     = data.get('stim', None)

        # arrange inputs
        if cond is None:
            cond = np.ones_like(evidence)
        u_cond = np.unique(cond)
        n_cond = len(u_cond)

        assert len(params) == len(self.param_names)*n_cond, '# parameters should equal n_params*n_cond'

        # parameter matrix
        param_mat = params.reshape((n_cond,-1))
        if self.si is not None:
            param_mat = self.sequential_inequality(param_mat, direction='fwd')

        # compute negative log likelihood
        _nll = 0
        for v_cond, v_params in zip(u_cond, param_mat):
            idx  = cond==v_cond
            _evi = evidence[idx]
            _cho = choice[idx]

            if self.link == 'gaussian':
                prob = self.predict(_evi, **{**kwargs, **dict(zip(self.param_names, v_params))})
            elif self.link == 'gaussian_pse':
                prob = self.predict(_evi, stim=stim[idx], **{**kwargs, **dict(zip(self.param_names, v_params))})

            prob1 = prob[_cho==1]
            prob0 = prob[_cho==0]
            prob1[prob1<=utils.EPS]    = utils.EPS
            prob0[prob0>=1.-utils.EPS] = 1.-utils.EPS
            
            _nll += -np.sum(np.log(prob1)) - np.sum(np.log(1.-prob0))
        
        return _nll


    def fit(self, data, 
            x0=None, lb=None, ub=None, 
            inherit_params=None, 
            constrain_params=None,
            constraints=[],
            **kwargs):
        """fit psychometric functions

        inputs
        ----------
            data : {'evidence', 'choice', ('cond'), ('stim')}
                evidence : (n_trial,) evidence (in degrees)
                choice   : (n_trial,) choice (1 for cw and 0 for ccw)
                cond     : (n_trial,) condition for fitting psychometric functions (ex. Timing)
                stim     : (n_trial,) stimulus orientations (in degrees)
            lb : (n_params,) array-like, optional
                lower bounds for parameters
            ub : (n_params,) array-like, optional
                upper bounds for parameters
            inherit_params : list of str, optional
                list of parameter names to inherit from `self.fitted_params`
            kwargs : dict
                arguments for scipy.optimize.minimize
        """
        evidence = data['evidence']
        cond     = data.get('cond', None)
        if cond is None:
            cond = np.ones_like(evidence)
        u_cond = np.unique(cond)
        n_cond = len(u_cond)

        # set initial values
        if x0 is None:
            x0 = np.zeros(len(self.param_names)*n_cond)
            x0[1::3] = 5. 
            x0[2::3] = 0.05
            if self.link == 'gaussian':
                x0[0::3] = 0.
            elif self.link == 'gaussian_pse':
                x0[0::3] = 1.

            if self.si is not None:
                if ('m' in self.si) & (n_cond > 1):
                    x0[4::3] = 1.

        # set bounds
        if lb is None:
            lb = np.zeros(len(self.param_names)*n_cond)
            lb[1::3] = utils.EPS
            if self.link == 'gaussian':
                lb[0::3] = -20.

        if ub is None:
            ub = np.zeros(len(self.param_names)*n_cond)
            ub[1::3] = 20.
            ub[2::3] = 0.3
            if self.link == 'gaussian':
                ub[0::3] = 20.
            elif self.link == 'gaussian_pse':
                ub[0::3] = 2.

        # inherit parameters
        if inherit_params is not None:
            for i_param, n_param in enumerate(self.param_names):
                if n_param in inherit_params:
                    for i_cond in range(n_cond):
                        idx_param = i_param + i_cond * len(self.param_names)
                        if self.fitted_params is not None:
                            x0[idx_param] = self.fitted_params[i_cond, i_param] if n_cond > 1 else self.fitted_params[i_param]
                        lb[idx_param] = x0[idx_param]
                        ub[idx_param] = x0[idx_param]

        bounds = Bounds(lb, ub)

        # constraints
        _constraints = constraints.copy()
        if constrain_params is not None:
            assert n_cond > 1, 'Constraining parameters is only valid for multiple conditions.'
            for c in constrain_params:
                for i_cond in range(n_cond-1):
                    idx_curr = self.param_names.index(c) + len(self.param_names)*i_cond
                    idx_next = self.param_names.index(c) + len(self.param_names)*(i_cond+1)
                    _constraints.append({'type': 'eq', 'fun': lambda p: p[idx_curr]-p[idx_next]})

        # fit
        res_fit = minimize(self.nll, x0, 
                           args = (data,),
                           bounds = bounds, 
                           constraints = _constraints,
                           **kwargs)
        
        # arrange outputs
        if n_cond == 1:
            params = res_fit['x']
        else:
            params = res_fit['x'].reshape((n_cond,-1))

        if self.si is not None:
            params = self.sequential_inequality(params)

        self.fitted_params = params

        if self.success:
            return res_fit['success']