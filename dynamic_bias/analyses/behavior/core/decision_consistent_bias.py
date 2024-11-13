import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .... import utils
from .psychometric_function import PsychometricFunction

class DecisionConsistentBias:
    """Decision-consistent bias class"""
    def __init__(self, 
                 **kwargs):
        self.psi = PsychometricFunction(**kwargs)

    def fit(self,
            data,
            **kwargs):
        """
        fit parameters for computing bias components (b_pre, b_post)
        
        inputs
        ------
            data : {'error', 'evidence', 'choice', ('near'), ('cond'), ('group')}
                error    : (n_trial,) error (in degrees)
                evidence : (n_trial,) evidence (in degrees)
                choice   : (n_trial,) choice (1 for cw and 0 for ccw)
                near     : (n_trial,) near reference labels (1 for near and 0 for far)
                group    : (n_trial,) group labels, if not provided, fit assuming single group
                cond     : (n_trial,) condition labels (ex. Timing)
        """
        # process data
        data = self.process(data, ['evidence', 'choice', 'cond', 'group'], **kwargs)
        u_group = np.unique(data['group'])

        # fit parameters
        params = []
        for v_group in u_group:
            idx_group = (data['group']==v_group)
            self.psi.fit(data={
                'evidence' : data['evidence'][idx_group],
                'choice'   : data['choice'][idx_group],
                'cond'     : data['cond'][idx_group]
            }, **kwargs)
            params.append(self.psi.fitted_params) # [n_cond,n_params]

        self.fitted_params = params # [n_groups][n_cond,n_params]


    def decompose(self,
                  data,
                  nuisance=['evidence'],
                  monte_carlo=False,
                  include_far=False,
                  **kwargs):
        """decompose the near-reference decision-consistent bias 
            into average b_pre and b_post across groups using linear regression
            currently, this only supports each group having the same conditions

        inputs
        ------
            monte_carlo (bool) : whether to use monte-carlo simulation for b_pre estimation    
            nuisance (list of keys in data) : nuisance regressors for b_post estimation.
        """
        # process data
        datap, n_trial = self.process(data, 
            ['error', 'evidence', 'choice', 'near', 'group', 'cond', 'b'], return_n_trial=True, **kwargs)

        if include_far:
            idx_base = np.ones(n_trial, dtype=bool)
        else:
            idx_base = datap['near'].astype(bool)

        u_group, i_group = np.unique(datap['group'], return_inverse=True)
        n_group = len(u_group)
        u_cond, i_cond = np.unique(datap['cond'], return_inverse=True)
        n_cond = len(u_cond)

        # 1. predict b_pre for each trial
        b_preds = self.predict(data, return_b_post=True, monte_carlo=monte_carlo, **kwargs)
        b_pres  = b_preds['b_pres']  # (n_trial,)
        b_posts = b_preds['b_posts'] # (n_trial,)

        b_pre_means = utils.nan([n_group, n_cond])
        for i_group, v_group in enumerate(u_group):
            idx_group = (datap['group']==v_group)
            for i_cond, v_cond in enumerate(u_cond):
                idx_cond = (datap['cond']==v_cond)
                idx = idx_base & idx_group & idx_cond
                b_pre_means[i_group, i_cond] = np.mean(b_pres[idx])

        # 2. generate a design matrix with the indicators of group and condition
        X = self.design_matrix(data, nuisance=nuisance)
        reg = LinearRegression().fit(X[idx_base], b_posts[idx_base])

        # 3. generate the table with the estimated b_post
        ii_group, ii_cond = np.meshgrid(np.arange(n_group), np.arange(n_cond), indexing='ij')
        X = self.design_matrix( dict(group=ii_group.flatten(), cond=ii_cond.flatten()) )
        b_post_means = reg.intercept_ +  X @ reg.coef_[:X.shape[-1]]
        b_post_means = b_post_means.reshape(n_group, n_cond)

        return {
            'b_pre'  : b_pre_means,
            'b_post' : b_post_means,
        }

    def predict(self, 
                data,
                params=None,
                nuisance=None,
                return_b_post=False,
                monte_carlo=False,
                **kwargs):
        """trial-wise prediction of the decision-consistent bias b into b_pre and b_post

        inputs
        ------
            nuisance (list of keys in data) : nuisance regressors for b_post estimation (without intercept)
            return_b_post (bool) : whether to return b_post (residuals). b = b_pre + b_post
            monte_carlo (bool) : whether to use monte-carlo simulation for b_pre estimation
        """
        # process data
        datap, n_trial = self.process(data, 
            ['error', 'evidence', 'choice', 'group', 'cond', 'b'], return_n_trial=True, **kwargs)

        u_group, i_group = np.unique(datap['group'], return_inverse=True)
        u_cond, i_cond = np.unique(datap['cond'], return_inverse=True)

        if params is None:
            params = self.fitted_params
        
        # b_pres according to the groups and conditions
        b_pres = np.zeros(n_trial)
        for i_group, v_group in enumerate(u_group):
            idx_group = (datap['group']==v_group)
            for i_cond, v_cond in enumerate(u_cond):
                idx_cond = (datap['cond']==v_cond)
                idx = idx_group & idx_cond
                b_pres[idx] = self.predict_b_pre(
                    datap['evidence'][idx], 
                    params=params[i_group][i_cond], 
                    monte_carlo=monte_carlo
                )

        res = {'b_pres' : b_pres}

        if return_b_post and nuisance is not None:
            reg_nuisance = np.stack([data[k] for k in nuisance], axis=-1)
            reg = LinearRegression( fit_intercept=False ).fit( reg_nuisance, data['b'] - b_pres )
            res['b_posts'] = datap['b'] - b_pres - reg.predict(reg_nuisance)
            res['b_posts'] = utils.wrap(res['b_posts'], period=180.)

        elif return_b_post:
            res['b_posts'] = datap['b'] - b_pres
            res['b_posts'] = utils.wrap(res['b_posts'], period=180.)

        return res


    def predict_b_pre(self, 
                      evidence, 
                      params=None,
                      monte_carlo=False,
                      **kwargs):
        """estimation of pre-decision bias b_pre (in degrees)

        inputs
        ------
            evidence : (n_trial,) evidence (in degrees)
            params : (m, s, lam) fitted parameters of the psychometric function
            (monte_carlo) (bool) : whether to use monte-carlo simulation for b_pre estimation

        returns
        -------
            bpre : (n_trial,) pre-decision biases
        """
        if params is None:
            params = self.params

        if monte_carlo:
            NotImplementedError( "Monte-Carlo simulation is not implemented yet" )

        m, s, lam = params
        acw  = lam + (1.-2*lam) * norm.cdf( m+evidence, scale=s )
        bpre = (1.-2*lam) / ( 2.*acw*(1.-acw) ) * norm.pdf( (m+evidence)/s ) * s

        return bpre


    def process(self, 
                data, 
                keys,
                thres=8,
                return_n_trial=False,
                **kwargs):
        """
        helper function to handle missing data
            (thres) (float) : threshold for near reference (|evidence| ≤ thres) if "near" is not provided
        """
        processed = dict()
        if 'error' in keys:
            processed['error']    = data['error']
        if 'evidence' in keys:
            processed['evidence'] = data['evidence']
        if 'choice' in keys:
            processed['choice']   = data['choice']
        if 'error' in keys and 'choice' in keys:
            processed['b'] = data['error'] * (data['choice']*2 - 1)

        n_trial = next((len(v) for v in data.values() if v is not None), 0) # len(1st non-None element)
        if 'near' in keys:
            processed['near']  = data.get('near', np.abs(data['evidence']) <= thres)
        if 'group' in keys:
            processed['group'] = data.get('group', np.zeros(n_trial))
        if 'cond' in keys:
            processed['cond']  = data.get('cond', np.ones(n_trial))

        processed = {k : processed[k] for k in keys}

        if return_n_trial:
            return processed, n_trial

        return processed


    def design_matrix(self, data, nuisance=None, **kwargs):
        """
        helper function to create a design matrix for regression.
        regressors : (n_group-1), (n_cond-1), and (n_group-1)x(n_cond-1) interaction terms
        """
        datap, n_trial = self.process(data, ['group', 'cond'], return_n_trial=True, **kwargs)

        if nuisance is not None:
            reg_nuisance = np.stack([data[k] for k in nuisance], axis=-1)
        else:
            reg_nuisance = np.zeros((n_trial, 0))

        u_group, i_group = np.unique(datap['group'], return_inverse=True)
        u_cond, i_cond = np.unique(datap['cond'], return_inverse=True)
        n_group = len(u_group)
        n_cond  = len(u_cond)

        ind_group = np.eye(n_group)[i_group][:, 1:]  # [n_trial, 0] when n_group=1
        ind_cond  = np.eye(n_cond)[i_cond][:, 1:]
        ind_inter = np.einsum('ni,nj->nij', ind_group, ind_cond).reshape(n_trial, -1)

        X = np.concatenate([ind_group, ind_cond, ind_inter, reg_nuisance], axis=-1)
        return X