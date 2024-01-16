import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds

EPS = 1e-10

class PsychometricFunction:
    """psychometric function class"""
    def __init__(self, 
                 link='gaussian', 
                 pse=None,
                 sequential_inequality=None, 
                 return_success=False,
                 ):
        """initialize psychometric function class

        Inputs
        ----------
            link : {'gaussian', 'gaussian_pse'}, str
                link function of psychometric function
                'gaussian'    : Gaussian psychometric function with lapse = guess
                'gaussian_pse': Gaussian psychometric function with PSE modulated by stimulus
            pse  : (n_stim,) array-like, optional
                Point-of-Subjective-Equality function. Only used when link='gaussian_pse'
            sequential_inequality : list, optional
                list of parameter names for sequential inequality constraints
                e.g. ['m', 's'] for m1<=m2<=... and s1<=s2<=...
            return_success : bool, optional
                whether to return success of fitting
        """
        self.link    = link
        self.pse     = pse
        self.se      = sequential_inequality
        self.success = return_success

        if self.link == 'gaussian':
            self.param_names = ['m', 's', 'lam']
        elif self.link == 'gaussian_pse':
            self.param_names = ['w', 's', 'lam']

    def __call__(self, x, **kwargs):
        return self.psi(x, **kwargs)
        
    def psi(self, x, **kwargs):
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
            for p in self.se:
                _par[:, self.param_names.index(p)] = np.cumsum(_par[:, self.param_names.index(p)])
        elif direction == 'bwd':
            for p in self.se:
                _par[:, self.param_names.index(p)] = np.diff(_par[:, self.param_names.index(p)], prepend=0)
        return _par

    def nll(self, params, data, **kwargs):
        """negative log likelihoods (loss function) for psychometric functions

        Inputs
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
                
        Returns
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
        if self.se is not None:
            param_mat = self.sequential_inequality(param_mat, direction='fwd')

        # compute negative log likelihood
        _nll = 0
        for v_cond, v_params in zip(u_cond, param_mat):
            idx  = cond==v_cond
            _evi = evidence[idx]
            _cho = choice[idx]

            prob = self.psi(_evi, stim=stim, **{**kwargs, **dict(zip(self.param_names, v_params))})
            prob1 = prob[_cho==1]
            prob0 = prob[_cho==0]
            prob1[prob1<=EPS]    = EPS
            prob0[prob0>=1.-EPS] = 1.-EPS
            
            _nll += -np.sum(np.log(prob1)) - np.sum(np.log(1.-prob0))
        
        return _nll

    def fit(self, data, 
            x0=None, lb=None, ub=None, **kwargs):
        """fit psychometric functions

        Inputs
        ----------
            lb : (n_params,) array-like, optional
                lower bounds for parameters
            ub : (n_params,) array-like, optional
                upper bounds for parameters
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

            if self.se is not None:
                if ('m' in self.se) & (n_cond > 1):
                    x0[4::3] = 1.

        # set bounds
        if lb is None:
            lb = np.zeros(len(self.param_names)*n_cond)
            lb[1::3] = EPS
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

        bounds = Bounds(lb, ub)

        # fit
        res_fit = minimize(self.nll, x0, 
                           args = (data,),
                           bounds = bounds, 
                           **kwargs)
        
        # arrange outputs
        if n_cond == 1:
            params = res_fit['x']
        else:
            params = res_fit['x'].reshape((n_cond,-1))

        if self.se is not None:
            params = self.sequential_inequality(params)

        self.fitted_params = params

        if self.success:
            return res_fit['success']