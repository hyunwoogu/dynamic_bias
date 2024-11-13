import numpy as np
from scipy.optimize import minimize

class NearReferenceStatistics:
    def __init__(self, 
                 thres=8,
                 n_edge=8,
                 bin_range=25):
        """
        inputs
        ------
            thres : threshold for near/far reference
            n_edge : number of edges for binning
            bin_range : valid range for binning
        """
        self.thres = thres
        self.n_edge = n_edge
        self.bin_range = bin_range

    def bin(self, x, edges=None, labels=None, n_edge=None, bin_range=None, invalid_label=np.nan, right=True, round=3):
        """bin x [1D array] values based on custom edges and bin labels."""
        if bin_range is None:
            bin_range = self.bin_range

        if n_edge is None:
            n_edge = self.n_edge

        if edges is None:
            edges = np.linspace(-bin_range, bin_range, num=n_edge)
            # values between (-np.inf, -bin_range] = 0
        
        if labels is None:
            # centroids of the edges
            labels = (edges[:-1] + edges[1:]) / 2
        
        if round is not None:
            labels = np.round(labels, round)
        
        digitized = np.digitize(x, edges, right=right)

        bins = np.full(len(x), invalid_label)
        for i in range(1, len(edges)):
            bins[digitized==i] = labels[i-1]
        
        return bins
    
    def centered_gaussian(self, x, params=None):
        """Centered Gaussian function."""
        if params is None:
            params = self.fitted_params
        baseline, sigma, gain = params
        return baseline + gain * np.exp(-0.5 * (x / sigma) ** 2)

    def fit_gaussian(self, x, y, init_params=None):
        """Fit a centered Gaussian (baseline, sigma, gain) to the data x, y."""
        def loss(params):
            pred = self.centered_gaussian(x, params)
            return np.sum((y - pred) ** 2)
        
        if init_params is None:
            init_params = [0, 1, 1]
        res = minimize(loss, x0=init_params)
        self.fitted_params = res.x