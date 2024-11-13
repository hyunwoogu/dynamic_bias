"""Working memory dynamics under hemodynamic response characteristics
"""
import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize, Bounds

class HemodynamicModel:
    """Dynamics under hemodynamic response characteristics"""
    def __init__(self, 
                 TR=2.0,
                 hrf='glover',
                 oversampling=4,
                 onset=0,
                 offset=28.,
                 onset_hemodynamic=0,
                 offset_hemodynamic=28.,
                 onset_stimulus=0,
                 onset_memory=None,
                 duration_stimulus=1.5,
                 duration_reference=.735,
                 duration_memory=None,
                 t_dms={1:6, 2:12},
                 visual_params=None,
                 loss='l2',
                 calib_t=None,):
        """Hemodynamic model

        inputs
        ------
            TR : repetition time
            hrf : hemodynamic response function (only canonical HRF - 'glover' is supported)
            oversampling : oversampling factor for the underlying time series
            onset : onset time for the underlying time series
            offset : offset time for the underlying time series
            onset_hemodynamic : onset time for the hemodynamic time series
            offset_hemodynamic : offset time for the hemodynamic time series
            onset_stimulus : onset time for the stimulus
            onset_memory : onset time for the memory (default: onset)
            duration_stimulus : duration of the stimulus (default: experiment)
            duration_reference : duration of the reference (default: participants' median rt)
            duration_memory : duration of the memory (default: experiment)
            t_dms : time points for the DM task (default: experiment)
            visual_params : visual parameters for the model
            loss : loss function for fitting the model
        """
        self.TR = TR 
        self.hrf = self.hrf_function(hrf)
        self.oversampling = oversampling
        self.onset = onset
        self.offset = offset
        self.onset_hemodynamic = onset_hemodynamic
        self.offset_hemodynamic = offset_hemodynamic
        self.onset_stimulus = onset_stimulus
        self.duration_stimulus = duration_stimulus
        self.duration_reference = duration_reference
        self.onset_memory = self.onset if onset_memory is None else onset_memory
        self.duration_memory = self.offset - self.onset if duration_memory is None else duration_memory
        self.t_dms = t_dms
        self.loss_name = loss
        self.calib_t = calib_t
        self.fitted_params = dict()
        if visual_params is not None:
            self.fitted_params['visual'] = visual_params


    def fit(self,
            data,
            traj,
            model='visual',
            t_underlying=None, 
            t_hemodynamic=None,
            onset_reference=None,
            initial_params=None,
            lb=None, ub=None,
            shared_params=True,
            loss_function='l2',
            inherit_params=None,
            return_success=False,
            **kwargs):
        """fits the underlying model's parameters given hemodynamic observations

        inputs
        ------
            data : {'choice', 'cond', 'evidence'} (each [t]) where t is the time points
            traj : [n_trial, t] or [t] : observed trajectory (in degrees)
            shared_params : whether to share all the parameters across conditions (WARNING: not tested)

        returns
        -------
            fitted_params[model] : [n_cond, n_params]
        
        """
        # unpack data
        choice = data.get('choice', None)
        cond   = data.get('cond', None)
        if cond is None:
            cond = np.ones_like(choice)
        u_cond = np.unique(cond)
        n_cond = len(u_cond)

        # 
        model_name = model

        if model == 'visual':
            name_params = ['beta_stimulus', 'beta_reference']

            if initial_params is None:
                initial_params = np.ones(2) if shared_params else np.ones((n_cond,2))

            if lb is None:
                lb = np.zeros_like(initial_params)
            
            if ub is None:
                ub = 30.*np.ones_like(initial_params)

        
        if model == 'piecewise_linear':
            name_params = ['c0', 'c1', 'c2', 'c3']

            if initial_params is None:
                initial_params = np.zeros(4) if shared_params else np.zeros((n_cond,4))

            if lb is None:
                lb = -20. * np.ones_like(initial_params)
            
            if ub is None:
                ub = 20. * np.ones_like(initial_params)

        # inherit parameters
        if inherit_params is not None:
            for i_param, n_param in enumerate(name_params):
                if n_param in inherit_params:
                    if self.fitted_params[model] is not None:
                        initial_params[...,i_param] = self.fitted_params[model][n_param]
                    lb[...,i_param] = initial_params[...,i_param]
                    ub[...,i_param] = initial_params[...,i_param]

        # fit
        bounds = Bounds(lb, ub)
        res_fit = minimize(self.loss,
                           x0=initial_params,
                           args=(data, traj, model, 
                                 t_underlying, t_hemodynamic, onset_reference,
                                 shared_params, loss_function,),
                           bounds=bounds,
                           **kwargs)

        params = res_fit['x'].reshape((n_cond,-1)) # params[model] : [n_cond, n_params] or [n_params]
        
        if shared_params:
            params = params[0]

        params = np.squeeze(params)

        if params.ndim == 0:
            self.fitted_params[model_name] = {name_params: params}
        elif params.ndim == 1:
            self.fitted_params[model_name] = {k:v for k,v in zip(name_params, params)}
        else:
            self.fitted_params[model_name] = {k:v for k,v in zip(name_params, params.T)}

        if return_success:
            return res_fit['success']
        

    def loss(self,
             params,
             data,
             traj,
             model='visual',
             t_underlying=None, 
             t_hemodynamic=None,
             onset_reference=None,
             shared_params=True,
             loss_function='l2',
             axis=None,
             **kwargs):
        """loss of the hemodynamic model for each condition
            compute underlying trajectories based on model using information from data

        inputs
        ------
            params : [n_params*n_cond] or [n_params] : parameters for the model
        """
        # extract (stimulus, reference, condition) from data
        stimulus  = data['stim']
        reference = data['ref']
        choice    = data.get('choice', None)
        cond      = data.get('cond', None)
        t_dms     = data.get('t_dms', None)

        if cond is None:
            cond = np.ones_like(stimulus)
        u_cond = np.unique(cond)
        n_cond = len(u_cond)

        if t_dms is None:
            t_dms = np.array([self.t_dms[c] for c in cond])

        if onset_reference is None:
            onset_reference = t_dms

        if choice is None:
            choice = np.ones_like(stimulus)

        # parameter matrix
        if shared_params:
            param_mat = np.tile(params, (n_cond,1))
        else:
            param_mat = params.reshape((n_cond,-1))

        # compute loss
        loss = 0
        for v_cond, v_params in zip(u_cond, param_mat):
            # construct parametes
            if model == 'visual':
                _params = {
                    'visual': dict(zip(['beta_stimulus', 'beta_reference'], v_params))
                }

            elif model == 'piecewise_linear':
                _params = {
                    'visual': self.fitted_params['visual'],
                    'piecewise_linear': dict(zip(['c0', 'c1', 'c2', 'c3'], v_params))
                }

            # construct data
            idx = cond==v_cond
            _data = {'stim'   : stimulus[idx],
                     'ref'    : reference[idx],
                     'choice' : choice[idx],
                     't_dms'  : onset_reference[idx]}

            # construct hemodynamic trajectory
            traj_pred = self.predict(
                data=_data,
                model=model,
                params=_params,
                t_underlying=t_underlying,
                t_hemodynamic=t_hemodynamic,
                **kwargs
            )
            loss += self.loss_function(traj[idx], traj_pred, axis=axis, loss_name=loss_function)

        return loss


    def predict(self,
                data,
                traj_underlying=None,
                model=None,
                params=None,
                t_underlying=None, 
                t_hemodynamic=None,
                onset_reference=None,
                **kwargs
                ):
        """predict the hemodynamic trajectory given underlying dynamics

        inputs
        ------
            t : time points
            data : {'relref', 'underlying', 'target', 'ref', 'evidence', ('t_dms')}
                relref : relative reference orientation (deg)
                underlying : underlying dynamics
                target : target orientation
                ref : reference orientation
                evidence : evidence
                t_dms : discrimination task time
            traj_underlying [n_trial, t] : underlying dynamics (in radians)
            t_underlying : time points for the underlying dynamics
                if None, equally spaced time points btw. onset and offset by (tr/oversampling)
            t_hemodynamic : time points for the hemodynamic time series
                if None, equally spaced time points btw. onset and offset by (tr)
        outputs
        -------
            traj : predicted trajectory [n_trial, times]

        """
        # convert data
        stimulus  = data['stim'] # stimulus orientation
        reference = data['ref']  # (absolute) reference orientation
        choice    = data.get('choice', None) # choice

        if t_underlying is None:
            t_underlying = np.arange(self.onset, self.offset, self.TR/self.oversampling)

        if t_hemodynamic is None:
            t_hemodynamic = np.arange(self.onset_hemodynamic, self.offset_hemodynamic, self.TR)

        if self.calib_t is not None:
            t_hemodynamic = t_hemodynamic + self.calib_t

        if onset_reference is None:
            onset_reference = data['t_dms']

        # params
        if params is None:
            params = self.fitted_params

        # underlying dynamics
        if traj_underlying is None:
            if model == 'visual':
                # "memory boxcar" over the memory demand
                traj_underlying = self.traj_piecewise_linear(
                    t=t_underlying
                )
            
            elif model == 'piecewise_linear':
                if choice is None: 
                    choice = np.ones_like(stimulus)

                traj_underlying = self.traj_piecewise_linear(
                    t=t_underlying,
                    t_dm=onset_reference[:,np.newaxis],
                    c0 = params['piecewise_linear']['c0'],
                    c1 = params['piecewise_linear']['c1'],
                    c2 = params['piecewise_linear']['c2'],
                    c3 = params['piecewise_linear']['c3'],
                )
                traj_underlying = (2*choice-1)[...,np.newaxis] * traj_underlying
            
        # 
        if len(traj_underlying.shape) == 1:
            traj_underlying = traj_underlying[np.newaxis]
        n_trial, n_t = traj_underlying.shape

        assert n_t == len(t_underlying), \
            "traj_underlying and t_underlying should have the same length"
        
        # visual drives
        traj_visual_drives = self.traj_visual_drive(t_underlying, 
                                                    t_hemodynamic,
                                                    stimulus,
                                                    params['visual']['beta_stimulus'],
                                                    self.onset_stimulus, 
                                                    self.duration_stimulus,
                                                    reference,
                                                    params['visual']['beta_reference'],
                                                    onset_reference,
                                                    self.duration_reference,
                                                    **kwargs)

        # memory drives
        traj_memory_drives = self.traj_memory(t_underlying, 
                                              t_hemodynamic,
                                              traj_underlying,
                                              self.onset_memory,
                                              self.duration_memory,
                                              **kwargs)

        # BOLD predictions
        pred_traj = traj_visual_drives + traj_memory_drives
        pred_traj = np.arctan2(pred_traj.imag, pred_traj.real)
        pred_traj = np.rad2deg(pred_traj/2.) # in degrees

        return pred_traj


    def traj_visual_drive(self,
                          t_underlying, 
                          t_hemodynamic,
                          stimulus,
                          beta_stimulus,
                          onset_stimulus, 
                          duration_stimulus,
                          reference,
                          beta_reference,
                          onset_reference,
                          duration_reference,
                          **kwargs):
        """modeled visual trajectory
            traj(t) = Hemodynamic [ β_θ * exp(2iθ) * boxcar_θ(t) + β_ρ * exp(2iρ) * boxcar_ρ(t) ]

        inputs
        ------
            stimulus [n_trial] : stimulus (in degrees)
            reference [n_trial] : (absolute) reference (in degrees)
        """
        # stimulus drives
        s = 2.*np.deg2rad(stimulus)
        sin_s_traj = self.traj_boxcar(t_underlying, onset_stimulus, duration_stimulus, np.sin(s))
        cos_s_traj = self.traj_boxcar(t_underlying, onset_stimulus, duration_stimulus, np.cos(s))
        sin_s_traj = self.convolve_hrf(sin_s_traj, t_underlying, t_hemodynamic)
        cos_s_traj = self.convolve_hrf(cos_s_traj, t_underlying, t_hemodynamic)

        # reference drives
        r = 2.*np.deg2rad(reference)
        sin_r_traj = self.traj_boxcar(t_underlying, onset_reference, duration_reference, np.sin(r))
        cos_r_traj = self.traj_boxcar(t_underlying, onset_reference, duration_reference, np.cos(r))
        sin_r_traj = self.convolve_hrf(sin_r_traj, t_underlying, t_hemodynamic)
        cos_r_traj = self.convolve_hrf(cos_r_traj, t_underlying, t_hemodynamic)

        # visual trajectory
        traj = beta_stimulus  * (cos_s_traj + 1j*sin_s_traj) + \
               beta_reference * (cos_r_traj + 1j*sin_r_traj)
        traj = np.squeeze(traj)

        return traj


    def traj_memory(self,
                    t_underlying, 
                    t_hemodynamic,
                    memory,
                    onset_memory,
                    duration_memory,
                    **kwargs):
        """modeled memory trajectory
            traj(t) = Hemodynamic [ exp(iu(t)) * boxcar_u(t) ], where u(t) is the underlying dynamics

        inputs
        ------
            memory [n_trial] or [n_trial, t] : memory or memory trajectory (in direction-radian)
        """
        if len(memory.shape) == 1:
            memory = memory[:,np.newaxis]

        boxcar_traj = self.traj_boxcar(t_underlying, onset_memory, duration_memory) # [t]
        sin_m_traj = self.convolve_hrf(boxcar_traj*np.sin(memory), t_underlying, t_hemodynamic)
        cos_m_traj = self.convolve_hrf(boxcar_traj*np.cos(memory), t_underlying, t_hemodynamic)

        # memory trajectory
        traj = cos_m_traj + 1j*sin_m_traj
        traj = np.squeeze(traj)

        return traj


    def traj_boxcar(self, t, onset, duration, gain=1.):
        """modeled boxcar (rectangular) trajectory
            boxcar(t) = gain (for onset<=t<onset+duration) or zero (otherwise)

        inputs
        ------
            t [t] : time points
            onset, duration, gain [(n_trial,) or scalar]
        
        returns
        -------
            traj [n_trial, t] or [t] : boxcar trajectory 
        """
        t, onset, duration, gain = map(np.asarray, (t, onset, duration, gain))

        if t.ndim == 1:
            t = t[:, np.newaxis]

        if onset.ndim == 0:
            onset = onset[np.newaxis]
        
        if duration.ndim == 0:
            duration = duration[np.newaxis]

        if gain.ndim == 0:
            gain = gain[np.newaxis]
        
        traj = gain * ((t >= onset) & (t < onset + duration))
        traj = np.squeeze(traj.T)
            
        return traj


    def traj_piecewise_linear(self, t, t_dm=0, c0=0,c1=0,c2=0,c3=0):
        """Piecewise linear function for modeling underlying             
            t : time points (in seconds) with shape
            t_dm : decision moment times with shape
            c0 + c1*t (for t<t_dm)
            c0 + c1*t_dm + c2 + c3*(t-t_dm) (for t>=t_dm)
                c0 : pre-dm intercept, c1 : pre-dm slope, c2 : jump, c3 : post-dm slope
        """
        traj = c0+c1*t + (c2+(c3-c1)*(t-t_dm))*(t>=t_dm)
        return traj


    def convolve_hrf(self,
                     traj_underlying,
                     t_underlying,
                     t_hemodynamic,
                     **kwargs):
        """convolve a time series with an HRF
            currently, assumes the equally spaced time points for underlying time series

        inputs
        ------
            traj_underlying [n_trial, t] or [t] : underlying time series.
            t_underlying [t] : time points for the underlying time series (in seconds).
            t_hemodynamic : time points for the hemodynamic time series (in seconds).
            oversampling (int) : oversampling factor for the underlying time series
            onset : onset time for the time series.
            offset : offset time for the time series.
        """    
        n_trial, n_t = traj_underlying.shape

        # perform convolution
        dt = np.diff(t_underlying[0:2])
        traj_hrf = self.hrf(t_underlying)
        traj_hemodynamic = np.zeros( (n_trial, len(t_hemodynamic)) )
        for i, traj in enumerate(traj_underlying):
            traj_hd  = np.convolve(traj, traj_hrf, mode='full')[:n_t]
            traj_hd *= dt # response per second
            traj_hemodynamic[i] = np.interp(t_hemodynamic, t_underlying, traj_hd)

        traj_hemodynamic = np.squeeze(traj_hemodynamic)

        return traj_hemodynamic


    def score(self,
              traj,
              traj_pred,
              method='cosine',
              axis=0):
        """ scoring the predicted trajectory

        inputs
        ------
            traj [...,t] or [t] : observed trajectory (in radians)
            traj_pred [...,t] : predicted trajectory (in radians)

        """
        # check if the method is cosine
        if method == 'cosine':
            traj = np.deg2rad(traj*2.)
            traj_pred = np.deg2rad(traj_pred*2.)
            score = np.mean(np.cos(traj - traj_pred), axis=axis)

        elif method == 'r_squared':
            rss = np.sum((traj - traj_pred)**2, axis=axis)
            tss = np.sum((traj - np.mean(traj, axis=axis, keepdims=True))**2, axis=axis)
            score = 1 - rss/tss
            
        return score


    def loss_function(self, 
                      traj,
                      traj_pred,
                      axis=None,
                      loss_name=None,
                      **kwargs):
        """loss function for fitting the model"""
        if loss_name is None:
            loss_name = self.loss_name

        if loss_name == 'l2':            
            return np.sum((traj_pred-traj)**2, axis=axis)
        elif loss_name == 'negative_cosine':
            return -np.mean(np.cos(traj - traj_pred), axis=axis)        
        else:
            raise ValueError("Only L2 and negative cosine losses are supported")


    def hrf_function(self, hrf='glover'):
        if hrf == 'glover':
            return self.glover
        else:
            raise ValueError("Only canonical HRF is supported")


    def glover(self, t, a1=6., a2=16., b1=1., b2=1., c=1/6.):
        """canonical double-gamma hemodynamic response function"""
        h1 = gamma.pdf(t, a=a1, scale=1./b1)
        h2 = gamma.pdf(t, a=a2, scale=1./b2)
        h  = h1 - c*h2
        return h


    def calibrate(self,
                  data,
                  traj_hemodynamic,
                  t_hemodynamic,
                  model='piecewise_linear',
                  method='r_squared',
                  axis=None,
                  lb=None, ub=None,
                  **kwargs):
        """calibrate the model by shifting the time points"""

        def _loss(dt):
            traj_pred = self.predict(data, model=model, 
                                     t_hemodynamic=t_hemodynamic+dt, **kwargs)
            return -self.score(traj_hemodynamic, traj_pred, method=method, axis=axis)
        
        lb = -self.TR if lb is None else lb
        ub = +self.TR if ub is None else ub
        res = minimize(_loss, 0, bounds=[(lb, ub)], **kwargs)
        self.calib_t = res['x'].item()


    def decompose_b(self,
                    t_dms=None,
                    t_em=18.,
                    params=None,
                    return_b=False):
        """decompose decision-consistent bias (b) into b_pre and b_post
             using the fitted parameters of the piecewise linear model    
        """
        if t_dms is None:
            t_dms = list(self.t_dms.values())
        if params is None:
            params = self.fitted_params['piecewise_linear']
        
        # list of the length len(t_dms) in degrees
        b = []
        b_pres  = []
        b_posts = []
        for t_dm in t_dms:
            b_pre = (params['c0']+params['c1']*t_dm) * 90/np.pi
            b_post = (params['c2']+params['c3']*(t_em-t_dm)) * 90/np.pi
            b_pres.append(b_pre)
            b_posts.append(b_post)
            if return_b:
                b.append(b_pre+b_post)
        
        if return_b:
            return {'b_pre': b_pres, 'b_post': b_posts, 'b': b}

        return {'b_pre': b_pres, 'b_post': b_posts}


    def interp(self,
               t,
               t_traj,
               traj,
               axis=-1,
               **kwargs):
        """linear continuum estimation of trajectory

        inputs
        ------
            traj [..., n_t_traj, ...] : trajectory
            axis : int, optional, specifies the dimension of t_traj

        returns
        -------
            traj_interp [..., n_t, ...] : interpolated trajectory
        """
        traj = np.moveaxis(traj, axis, -1)
        n_trials = traj.shape[:-1]  # all dimensions except the last one
        n_t_traj = traj.shape[-1]

        # iterate over n_trials (similar to itertools.product)
        traj_interp = np.zeros(n_trials + (len(t),))
        for idx in np.ndindex(n_trials):
            traj_interp[idx] = np.interp(t, t_traj, traj[idx])
        traj_interp = np.moveaxis(traj_interp, -1, axis)

        return traj_interp