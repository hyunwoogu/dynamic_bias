""" Drift-diffusion models
"""
import time
import numpy as np
from scipy.linalg import toeplitz, expm
from scipy.stats import vonmises, norm
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d

class DynamicBiasModel:
    """Dynamical models of decision-consistent bias

    inputs
    ----------
        stimulus_specific_bias: estimated stimulus-specific bias
        p_basis : precision of basis functions of stimulus-specific bias  
        delay_condition : Early(1) and Late(2) delay conditions
        rel_ref : relative reference orientations [deg]
        delay : 1st, 2nd delay time for Early/Late conditions [sec]
        n_bins : #bins for the circular space
        n_iter : #iterations for the independent model fitting
        n_print : #steps to print the model fitting info
        near_cutoff : cut-off value for "near" reference [deg]
        sig : smoothness of decision-conditioning masks [rad]
        eps : tolerance for the likelihood value
        model : model types, 'full' or 'reduced'
        data_unit : unit of the data, 'deg' or 'rad'
    """
    def __init__(self, 
                 stimulus_specific_bias,
                 weights,
                 p_basis     = 6.,
                 delay_condition=[1,2], 
                 rel_ref     = [-21,-4,0,4,21],
                 delay       = [[6.,12.],[12.,6.]],
                 n_bins      = 96,
                 n_iter      = 20, 
                 n_print     = 100,
                 near_cutoff = 8,
                 sig         = 0.2,
                 eps         = 1e-10,
                 model       = 'full',
                 data_unit   = 'deg'):

        self.n                  = n_bins
        self.n_iter             = n_iter
        self.n_print            = n_print
        self.delay_condition    = delay_condition
        self.delay              = delay
        self.p_basis            = p_basis
        self.weights            = weights
        self.eps                = eps
        self.sig                = sig
        self.model              = model
        self.kappa              = stimulus_specific_bias
        self.relative_reference = rel_ref
        self.data_unit          = data_unit
        self.near_cutoff        = near_cutoff
        
        # construction of the circular space: [-pi,pi] as range
        self.m  = np.linspace(-np.pi, np.pi, self.n, endpoint = False)
        self.dm = self.m[1] - self.m[0]
        self.extend_m  = np.linspace(-1.5*np.pi, 2.5*np.pi, 2*self.n, endpoint = False)
        
        # finite difference matrices
        self.diff1  = 1./(self.dm**2)
        self.diff2  = 1./self.dm /2.
        h_diffu     = np.concatenate([[-2*self.diff1, self.diff1], np.zeros(self.n-3), [self.diff1]])
        self.Hdiffu = toeplitz(h_diffu)
        h_drift     = np.concatenate([[0, -self.diff2], np.zeros(self.n-3), [self.diff2]])
        self.Hdrift = toeplitz(h_drift, -h_drift)

        # construction of stimulus-specific bias
        self.kappa = self.kappa(self.m)
        self.kappa = self.kappa/np.max(np.abs(self.kappa))

        # decision-conditioning masks
        self.mask_vec = np.ones(self.n)
        self.mask_vec[int(self.n/2):] = 0

        # wrapping-around indices
        self.wrap_idx = np.arange((self.n+1))%self.n
        self.wrap_m   = np.concatenate([self.m,[self.m[0]+2*np.pi]])


    def convert_unit(self, data):
        """convert data to radian"""
        if self.data_unit == 'deg':
            data.update({k: ((v-90.)%180.-90.)*np.pi/90. for (k,v) in data['deg'].items()})
        elif self.data_unit == 'rad':
            data.update({k: ((v-np.pi)%(2*np.pi)-np.pi) for (k,v) in data['rad'].items()})
        return data

    def transform(self, params, direction='fwd'):
        """transform (normalized to [0,1]) parameters to help optimization
            'fwd': from original to [0,1] normalized
            'inv': from [0,1] normalized to original
        """
        lb = self.bounds.lb
        ub = self.bounds.ub
        
        if direction == 'fwd':
            _params = (params-lb)/(ub-lb)

        elif direction == 'inv':
            _params = params*(ub-lb) + lb

        return _params

    def efficient_encoding(self, s):
        """efficient encoding constraint (Wei & Stocker, 2015; Hanh & Wei, 2022)
        """
        c_basis = np.linspace(0, 2*np.pi, 12, endpoint = False)
        v = vonmises.pdf(self.m.reshape(-1,1), loc=c_basis, kappa=self.p_basis)
        inv2  = np.sum(v*np.array(self.weights)[np.newaxis,1:],axis=-1)
        inv2 -= inv2.min()-self.eps
        inv2 /= inv2.max()
        inv2  = (1.-s)*inv2+s
        p  = 1./np.sqrt(inv2)
        F  = np.cumsum(p)
        F /= F[-1]/(2*np.pi)
        F  = np.concatenate([[0],F[:-1]])
        return p, F

    def F_interp(self, F):
        F = interp1d(np.concatenate([self.m,[np.pi]]),
                     np.concatenate([F,[2*np.pi]]), kind='linear')
        return F

    def gen_mask(self, data):
        """generate decision-conditioning masks for the given reference data
        """
        distance = (self.m.reshape(-1,1) - data['ref'] - np.pi) % (2.*np.pi) - np.pi
        min_dist = np.argmin(np.abs(distance), axis = 0)

        mask_cw = np.zeros((self.n, len(data['ref'])))
        for i_r, v_r in enumerate(data['ref']):
            arg_min = min_dist[i_r]
            lh  = self.extend_m[arg_min:(arg_min+int(self.n/2))]
            uh  = self.extend_m[(arg_min+int(self.n/2)):(arg_min+self.n)]
            vec = np.concatenate([norm.cdf(lh, loc=v_r, scale=self.sig), norm.sf(uh, loc=v_r+np.pi, scale=self.sig)])
            mask_cw[:,i_r] = np.roll(vec,arg_min-int(self.n/4))

        self.mask_cw  = mask_cw
        self.mask_ccw = 1. - mask_cw
    
    def forward(self, params, data, readout_time=None):
        """forward model
            for full model, parameters are: [w_K, w_E, w_P, w_D, w_r, w_a, lam, s]
            for reduced model, parameters are: [w_E, w_P, w_D, w_r, w_a, lam, s]
            for null model, parameters are: [w_E, w_P, w_r, w_a, lam, s]

            readout_time [sec]: time points to readout the density propagation. 
                If None, only the final readout is returned.
        """
        # parameters
        if self.model == 'full':    
            w_K, w_E, w_P, w_D, w_r, w_a = [p*np.pi/90. for p in params[:(-2)]]
        elif self.model == 'reduced':
            w_E, w_P, w_D, w_r, w_a = [p*np.pi/90. for p in params[:(-2)]] 
        elif self.model == 'null':
            w_E, w_P, w_r, w_a = [p*np.pi/90. for p in params[:(-2)]]   

        lam, s = params[(-2):]

        # efficient encoding 
        kappa = self.kappa
        p,F = self.efficient_encoding(s)
        p_e = np.sqrt(1./np.clip(w_E,self.eps,None))
        F_interp = self.F_interp(F)
        P0  = np.exp(p_e*(np.cos((F.reshape(-1,1)-F_interp(data['stim'])))-1.))
        P0  = P0*p.reshape(-1,1)
        P0 /= P0.sum(axis=0,keepdims=True)

        # transition matrices
        Lp = self.Hdiffu/2.*np.power(w_P,2) # production transition
        
        if self.model == 'null':
            L = self.Hdiffu/2.*0.
        elif self.model == 'reduced':
            L = self.Hdiffu/2.*np.power(w_D,2)
        elif self.model == 'full':    
            L = self.Hdiffu/2.*np.power(w_D,2) - self.Hdrift@np.diag(kappa*w_K)

        # density propagation
        P   = [np.zeros_like(P0)*np.nan, np.zeros_like(P0)*np.nan] # CW, CCW
        Pdm = [np.zeros_like(P0)*np.nan, np.zeros_like(P0)*np.nan] # CW, CCW
        L1  = [L*self.delay[0][0], L*self.delay[0][1]] # 1st delay transitions for early and late DM conditions
        L2  = [L*self.delay[1][0], L*self.delay[1][1]] # 2nd delay transitions for early and late DM conditions

        # conditional density propagation
        for i_delay, delay in enumerate(self.delay_condition):
            _P = expm(L1[i_delay]) @ P0
            _P_cw   = (1.-lam) * self.mask_cw * _P + lam * self.mask_ccw * _P
            _P_ccw  = lam * self.mask_cw * _P + (1.-lam) * self.mask_ccw * _P
            idx_dly = data['delay']==delay
            for r in self.relative_reference:
                idx = idx_dly & (data['relref']==r)
                Pdm[0][:,idx] = expm(self.Hdrift*(-w_a*(np.abs(r)<self.near_cutoff)-r*w_r)) @ _P_cw[:,idx]
                Pdm[1][:,idx] = expm(self.Hdrift*( w_a*(np.abs(r)<self.near_cutoff)-r*w_r)) @ _P_ccw[:,idx]

            # post-decision dynamics + readout noise
            P[0][:,idx_dly] = expm(Lp) @ expm(L2[i_delay]) @ Pdm[0][:,idx_dly]
            P[1][:,idx_dly] = expm(Lp) @ expm(L2[i_delay]) @ Pdm[1][:,idx_dly]

        # return
        if readout_time is None:
            P[0][P[0]<0] = 0.
            P[1][P[1]<0] = 0.
            sums  = (P[0]+P[1]).sum(axis=0,keepdims=True)
            P[0] /= sums
            P[1] /= sums
            return P
        
        else:            
            # forward propagation to conditioned distribution
            Pm = [np.nan*np.zeros((len(readout_time),)+P0.shape), 
                  np.nan*np.zeros((len(readout_time),)+P0.shape)] # CW, CCW

            for i_delay, delay in enumerate(self.delay_condition):
                idx = (data['delay']==delay) 

                # pre-decision dynamics
                pre_dm_readout_time = readout_time[readout_time<=self.delay[0][i_delay]]

                for i_t, v_t in enumerate(pre_dm_readout_time):
                    m_pri     = expm(L*v_t) @ P0[:,idx]
                    m_lik_cw  = (expm(L*(self.delay[0][i_delay]-v_t))[:,:,np.newaxis] * self.mask_cw[:,np.newaxis,idx]).sum(axis=0)
                    m_lik_ccw = (expm(L*(self.delay[0][i_delay]-v_t))[:,:,np.newaxis] * self.mask_ccw[:,np.newaxis,idx]).sum(axis=0)
                    m_pos_cw  = ((1.-lam)*m_lik_cw  + lam*m_lik_ccw) * m_pri
                    m_pos_ccw = ((1.-lam)*m_lik_ccw + lam*m_lik_cw)  * m_pri
                    Pm[0][i_t,:,idx] = m_pos_cw.T
                    Pm[1][i_t,:,idx] = m_pos_ccw.T

                # post-decision dynamics
                post_dm_readout_time = readout_time[readout_time>self.delay[0][i_delay]]

                for i_t, v_t in enumerate(post_dm_readout_time):
                    Pm[0][len(pre_dm_readout_time)+i_t,:,idx] = (expm(L*(v_t-self.delay[0][i_delay])) @ Pdm[0][:,idx]).T
                    Pm[1][len(pre_dm_readout_time)+i_t,:,idx] = (expm(L*(v_t-self.delay[0][i_delay])) @ Pdm[1][:,idx]).T

            return P, Pm
    
    def nll(self, params, data, info=None, transform=False):
        """negative log-likelihood function for the given data"""
        
        # transform parameters
        if transform:
            params = self.transform(params,'inv')

        # evaluate P
        P = self.forward(params, data)

        # evalutate LL
        prob_cw  = self.multi_interp(data['estim'][data['dm']==1], self.wrap_m, P[0][self.wrap_idx][:,data['dm']==1].T)
        prob_ccw = self.multi_interp(data['estim'][data['dm']==0], self.wrap_m, P[1][self.wrap_idx][:,data['dm']==0].T)

        prob_cw[prob_cw   < self.eps] = self.eps
        prob_ccw[prob_ccw < self.eps] = self.eps
        
        neg_ll   = -np.sum(np.log(prob_cw)) - np.sum(np.log(prob_ccw))
        
        if ~np.isreal(neg_ll) | ~np.isfinite(neg_ll):
            neg_ll = np.inf
            
        # diplay info during optimization
        if info:
            if (info['Nfeval'] % self.n_print == 0):
                print('Iter: {0:05d} / Params:'.format(info['Nfeval']), 
                ['{:.3f}'.format(par) for par in params], ' / nLL:{:.3f}'.format(neg_ll))
            info['Nfeval'] += 1
        return neg_ll

    def fit(self, data, lb = None, ub = None, disp_prog = False):
        """fit the model to the data"""

        data = self.convert_unit(data)
        if lb is None:
            if self.model == 'full':
                lb = np.array([0, 0, 0, 0, 0, -15., 0, 0])
            elif self.model == 'reduced':
                lb = np.array([0, 0, 0, 0, -15., 0, 0])
            elif self.model == 'null':
                lb = np.array([0, 0, 0, -15., 0, 0])
        if ub is None:
            if self.model == 'full':
                ub = np.array([15., 15., 15., 15., 15., 15., 0.15, 1.])
            elif self.model == 'reduced':
                ub = np.array([15., 15., 15., 15., 15., 0.15, 1.])
            elif self.model == 'null':
                ub = np.array([15., 15., 15., 15., 0.15, 1.])
        self.bounds = Bounds(lb, ub)
        bounds_norm = Bounds(np.zeros(len(lb)), np.ones(len(ub)))
        self.gen_mask(data)

        # fit
        self.rng     = np.random.default_rng()
        self.loglik  = np.nan
        x0 = np.zeros((self.n_iter, len(ub)))
        x  = np.zeros((self.n_iter, len(ub)))
        ll = np.zeros(self.n_iter) 
        for i_iter in range(self.n_iter):
            print('Starting Iteration #{0:02d} out of {1:02d} iterations'.format(i_iter+1, self.n_iter))
            tic = time.time()
            gtg = False            
            info = {'Nfeval': 0}
            while not gtg:
                _x0  = np.random.rand(len(lb))
                neg_ll  = self.nll(_x0, data, transform=True)
                if np.isfinite(neg_ll):
                    gtg = True
    
            x0[i_iter,:] = self.transform(_x0,'inv')
            print('Initialized Iteration #{0:02d} out of {1:02d} iterations'.format(i_iter+1, self.n_iter))
            tmp_fit     = minimize(self.nll, _x0, args =(data, info, True), 
                                   bounds=bounds_norm, options = {'disp': disp_prog})
            x[i_iter,:] = self.transform(tmp_fit.x,'inv')
            ll[i_iter]  = -tmp_fit.fun
            
            if (i_iter == 0) | (self.loglik < -tmp_fit.fun):
                self.fitted_params = self.transform(tmp_fit.x,'inv')
                self.loglik        = -tmp_fit.fun
            
            self.x0   = x0
            self.x    = x
            self.ll   = ll
            toc = time.time()
            print('Finished Iteration #{0:02d} out of {1:02d} iterations (Elapsed: {2:5.2f})'.format(i_iter+1, self.n_iter, toc-tic))

    def multi_interp(self, x, xp, fp):
        """interpolation for 2D array
            a vectorized version to `np.array([np.interp(x[i], xp, fp[i]) for i in range(x.size)])`
            https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array
        """
        i = np.arange(x.size)
        j = (np.searchsorted(xp, x) - 1)
        d = (x - xp[j]) / (xp[j + 1] - xp[j])
        return (1 - d) * fp[i, j] + fp[i, j + 1] * d