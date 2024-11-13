import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from .... import utils

class StimulusSpecificBias:
    """Stimulus-specific bias class"""
    def __init__(self, 
                 n_basis=12, 
                 p_basis=None):
        self.n_basis = n_basis
        self.p_basis = p_basis if p_basis is not None else n_basis / 2.0
        self.c_basis = np.linspace(0, 2 * np.pi, n_basis, endpoint=False)
        self.weights = None
        
    def __call__(self, x, w=None, **kwargs):
        return self.predict(x, w, **kwargs)

    def predict(self, x, w=None):
        """estimate smooth, non-parametric stimulus-specific bias function κ 
            based on the measured at n different stimuli

        inputs
        ----------
            x (size n) : stimulus (in degree, [0, 180])
            w (size m) : weights of the von-Mises basis functions (optional)

        returns
        -------
            kappa_hat : stimulus-specific bias function evaluated at x (in degree)
        """
        if w is None:
            w = self.weights

        X = self.derivative_von_mises(x.reshape((-1,1)), self.c_basis, self.p_basis)
        kappa_hat = w[0] + np.sum(X * w[1:], axis=-1)

        return kappa_hat

    def fit(self, y, x=None, penalty=utils.EPS):
        """fit weights of the stimulus-specific bias function κ

        inputs
        ----------
            y (size n) : measured errors
            x (size n) : stimulus (in degree, [0, 180]). If None, use the default stimulus list.
            penalty : penalty term for regularization in ridge regression (optional)
        """
        if x is None:
            x = utils.exp_stim_list().reshape((-1, 1))
        X = self.derivative_von_mises(x, self.c_basis, self.p_basis)

        if penalty is not None:
            reg = Ridge(alpha=penalty).fit(X, y)
        else:
            reg = LinearRegression().fit(X, y)
        self.weights = [reg.intercept_, *reg.coef_]


    def derivative_von_mises(self, x, mu, kappa):
        """derivative of von-Mises density functions
        
        inputs
        ----------
            x (size n) : stimulus (in degree, [0, 180])
            mu (size m) : mean of the von-Mises basis functions
            kappa (size m) : concentration of the von-Mises basis functions
        """
        x = np.deg2rad(2.*x)
        return (kappa * np.sin(mu - x) * np.exp(kappa * np.cos(mu - x)) / (np.i0(kappa) * 2 * np.pi))
