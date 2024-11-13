"""Linear decoding model
"""
import numpy as np
import scipy.linalg

class LinearDecodingModel:
    """Linear decoder based on the inversion of encoding model (Brouwer & Heeger, 2009)
    
    An estimator for the cortical sensory information representation.
    Model: X = WY, where the matrices X: BOLD response, W: channel weights, Y: channel response to stimuli.
    Y, the ground-truth channel response, is generated by uniformly paced raised cosine bases. 
    
    Starts by splitting data into "train" set X_T and "validation" set X_V (leave-one-out cross-validation).
    First, fits W by least-squares method: W' = XC^T(C*C^T)^(-1).
    Second, predicts X for the training set by least-squares method: C' = (W'^T*W')^(-1) W^TX

    Parameters
    ----------
        ori_shift: the degree of shift which the shifting process will take. Minimal unit of analysis. 
        n_ori_chan: the number of channels used for modeling the channel response
        n_stim: the number of unique stimuli
        n_resol: 
        span: the extent of the orientation space
        cv_method: the method how the cross-validation will be performed
    """

    def __init__(self, ori_shift=1.5, n_ori_chan=8, n_stim=24, n_resol=360, span=180, cv_method='loto'):
        
        self.span        = span
        self.ori_shift   = ori_shift
        self.n_ori_chan  = n_ori_chan
        self.n_chan_bin  = int(self.span/self.ori_shift)
        self.n_resol     = n_resol
        self.n_stim      = n_stim
        self.n_model     = int(self.n_chan_bin/self.n_ori_chan)         # number of independent linear models
        self.xx          = np.arange(self.span, step=self.span/self.n_resol) # 
        self.cv_method   = cv_method
        self._make_basis_set()
                
    def fit(self, X, y, constraint=None, cv=None, return_W='none'):
        # X:  [n_trial, n_voxels]   (BOLD activities)
        # y:  [n_trial]           (stimulus)
        # cv: [n_stim/n_cv, n_cv] (testing set matrix, n_cv: testing set size for one-time feed)
        # contraint_flag: boolean index specifying the targeted training trials

        self.n_trial, self.n_unit = X.shape
        self.stim = y
        self._make_stim_mask()

        if constraint is None:
            constraint = np.ones((len(self.stim),), dtype=bool)

        self.constraint = constraint

        if cv is None:
            if self.cv_method == 'loto':
                cv = np.arange(self.n_trial).reshape((-1,1))
                
            elif self.cv_method == 'loro':
                raise NotImplementedError("Implicit leave-one-run-out is not yet implemented. Use explicit method.")
                
            else:
                raise ValueError("Argument cv should be specified.")
                
        self.cv = cv
        
        # fit channel weight
        weight_recon = np.zeros((len(self.cv), int(self.span/self.ori_shift), self.n_unit))

        for i_cv, s_cv in enumerate(self.cv):    
            test_flag  = np.in1d(np.arange(self.n_trial), s_cv)
            train_flag = (~test_flag) & self.constraint
            trn        = X[train_flag,:]
            
            for i_model in range(self.n_model): 
                model_index = self._model_index(i_model)
                W, _, _, _  = scipy.linalg.lstsq(self.chanX[:,model_index][train_flag,:], trn) 
                weight_recon[i_cv,model_index,:] = W
        
        self.W = weight_recon
        
        if return_W == 'all':
            return self.W
        elif return_W == 'average':
            return np.mean(self.W, axis=0)

        
    def predict(self, X):
        # X:  [n_trial, n_voxels] or [n_general, n_trial, n_voxels] (BOLD activities)
        # reconstruct channel response function
        
        if len(X.shape) == 2:
            chan_recon       = np.zeros((1, self.n_trial, self.n_chan_bin)) 
            
        elif len(X.shape) == 3:
            n_general, _, _  = X.shape
            chan_recon       = np.zeros((n_general, self.n_trial, self.n_chan_bin)) 
        
        for i_cv, s_cv in enumerate(self.cv):    
            test_flag  = np.in1d(np.arange(self.n_trial), s_cv)
            
            if len(X.shape) == 2:
                tsts   = X[test_flag,:].reshape((1,sum(test_flag),-1))
            elif len(X.shape) == 3:
                tsts   = X[:,test_flag,:]
            
            for i_model in range(self.n_model): 
                model_index = self._model_index(i_model)
                
                for i_tst, tst in enumerate(tsts):
                    C, _, _, _  = scipy.linalg.lstsq(self.W[i_cv,model_index,:].T, tst.T)                                        
                    chan_recon[i_tst][np.ix_(test_flag,model_index)] = C.T

        
        if len(X.shape) == 2:
            return chan_recon.reshape((self.n_trial,-1))
        
        elif len(X.shape) == 3:
            return chan_recon

    def _model_index(self, model, boolean=True): 
        integer = np.arange(model, self.n_chan_bin, step=self.n_model)
        
        if boolean: 
            return np.in1d(np.arange(self.n_chan_bin),integer)
        else:
            return integer
    
    def _make_basis(self, mu): 
        return (np.cos(np.pi/self.span*(self.xx-mu)))**(self.n_ori_chan-(self.n_ori_chan % 2)) # 
    
    def _make_basis_set(self):
        basis_set   = np.zeros((self.n_resol,int(self.span/self.ori_shift)))
        
        for step in range(int(self.span/self.ori_shift)): 
            basis_set[:,step] = self._make_basis(step*self.ori_shift)
            
        self.basis_set = basis_set
        
    def _make_stim_mask(self):
        stim_mask = np.zeros((self.n_trial, self.n_resol))    
        
        for i_stim, s_stim in enumerate(self.stim): 
            stim_mask[i_stim,int(s_stim*self.n_resol/self.n_stim)] = 1.

        self.chanX = stim_mask @ self.basis_set