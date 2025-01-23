"""
Utility functions for statistics.
"""
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, chi2
from .dataset import set_dict, recursive_list

EPS = 1e-10
EPS32 = np.finfo(np.float32).eps

def nan(size, **kwargs):
    """make a nan matrix of the given size"""
    return np.full(size, np.nan, **kwargs)

def wrap(x, period=2*np.pi):
    """wrap the circular data to the range of [0, period]"""
    return (x - period/2.) % (period) - period/2.

def asvoid(arr):
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

def inNd(a, b, assume_unique=False):
    a = asvoid(a)
    b = asvoid(b)
    return np.in1d(a, b, assume_unique)

def jitter(x, scale=1e-2):
    x = np.asarray(x)
    return x + np.random.randn(*x.shape)*scale

def ori2dir(ori, unit_ori='degree', unit_dir='radian'):
    """convert orientation to direction"""
    if unit_ori == 'radian':
        ori = np.rad2deg(ori)
    dir = ori * np.pi/90
    if unit_dir == 'degree':
        dir = np.rad2deg(dir)
    return dir

def dir2ori(dir, unit_dir='radian', unit_ori='degree'):
    """convert direction to orientation"""
    if unit_dir == 'degree':
        dir = np.deg2rad(dir)
    ori = dir * 90/np.pi
    if unit_ori == 'radian':
        ori = np.deg2rad(ori)
    return ori

def reflect(x, index=0, axis=-1): 
    """reflecting function
        designed for the wrapping-around behavior of the circular space
        example: reflect([a,b,c,d,e,f], index=2) = [e,d,c,b,a,f]
    """    
    # roll so that the specified index becomes the first element in the axis
    x = np.roll(x, shift=-index, axis=axis)
    
    # flip along the axis
    x = np.flip(x, axis=axis)
    x = np.roll(x, shift=1, axis=axis)
    
    # roll back so that the first element goes back to its original position
    x = np.roll(x, shift=index, axis=axis)
    
    return x

def se(x, axis=-1):
    """standard error, excluding NaN values"""
    x = np.asarray(x)
    n = np.sum(~np.isnan(x), axis=axis)
    return np.nanstd(x, axis=axis) / np.sqrt(n)

def project(p, p1=None, p2=None, slope=None, intercept=0):
    """project point p ([2] or [N,2]) onto
        (i) line defined by p1 and p2 or
        (ii) line defined by slope and intercept
    """
    p = np.array(p)
    if p1 is not None and p2 is not None:
        p1 = np.array(p1)
        p2 = np.array(p2)
        dir = p2-p1
        proj = p1 + dir*np.sum((p-p1)*dir,axis=-1,keepdims=True) / np.sum(dir**2)
    
    elif slope is not None:
        dir  = np.array([1, slope])
        bias = np.array([0, intercept])
        proj = bias + dir*np.sum((p-bias)*dir,axis=-1,keepdims=True) / np.sum(dir**2)

    return proj

def simple_linregress(x, y, 
                      return_p_value=False,
                      return_residual_variance=False):
    """simple linear regression"""
    x = sm.add_constant(x)
    model   = sm.OLS(y, x)
    results = model.fit()
    params  = {k: v for k, v in zip(['intercept', 'slope'], results.params)}
    returns = (params,)
    if return_p_value:
        pvals    = {k: v for k, v in zip(['intercept', 'slope'], results.pvalues)}
        returns += (pvals,)
    if return_residual_variance:
        returns += (results.scale,)
    return returns

def quantile(prob, support, quantile=0.5):
    """compute the specified quantile of a probability density.
        assumes equally spaced and increasing support
    
    inputs
    ------
        prob     : probability densities
        support  : support values
        quantile : desired quantile (e.g., 0.5 for median, 0.25 for first quartile)
    """
    prob = np.asarray(prob) / np.sum(prob)
    prob_cum = np.concatenate( [[0], np.cumsum(prob)] )

    ds = support[1] - support[0]
    support0 = np.concatenate( [ [support[0]-ds/2.], np.asarray(support) + ds/2.] )

    value = np.interp(quantile, prob_cum, support0)
    return value

def resample_indices(n, replace=True, seed=None, groups=None, size=None, **kwargs):
    """resample the indices used for bootstrapping and permutation tests
        groups : list of group labels for stratified resampling
    """
    if seed is not None:
        np.random.seed(seed)

    if groups is not None:
        if not isinstance(groups, list):
            groups = [groups]

        # unique combinations of group labels
        i_groups = []
        for group in groups:
            _, i_group = np.unique(group, return_inverse=True)
            i_groups.append(i_group[:, None])
        i_groups = np.concatenate(i_groups, axis=-1)
        u_groups = np.unique(i_groups, axis=0)

        # stratified resampling
        indices = np.empty(n, dtype=int) if size is None else []
        for u_group in u_groups:
            idx = np.all(i_groups == u_group, axis=1)
            idx_global = np.where(idx)[0]
            if size is None:
                indices[idx] = np.random.choice(idx_global, size=np.sum(idx), replace=replace, **kwargs)
            else:
                indices.append(np.random.choice(idx_global, size=size, replace=replace, **kwargs))

        if size is not None:
            indices = np.concatenate(indices)

    else:
        size = n if size is None else size
        indices = np.random.choice(n, size, replace=replace, **kwargs)

    return indices

def permutation_test(obs, null, alternative='two-sided', axis=0, **kwargs):
    """
    compute p-value for a permutation test.
    if n_dim > 1 and joint is True, compute the p-value based on the joint distribution.

    inputs
    -------
        obs [n_dim] : observed test statistic value.
        null [n_perm,n_dim] : null distribution from permutations
        (alternative) : alternative hypothesis ('two-sided', 'greater', 'less')

    returns
    -------
        p_value : p-values
    """
    if alternative == 'two-sided':
        bool_vec = np.abs(null) >= np.abs(obs)

    elif alternative == 'greater':
        bool_vec = null >= obs

    elif alternative == 'less':
        bool_vec = null <= obs

    p_value = np.mean(bool_vec, axis=axis, **kwargs)
    
    return p_value


def meanstats(data, axis=-1, median=False, sd=False, ci=False, propagate_nan=False, central=None, **kwargs):
    """means and variability measures for the data"""
    std  = np.std  if propagate_nan else np.nanstd
    
    if central is None:
        if median:
            central = np.median if propagate_nan else np.nanmedian
        else:
            central = np.mean if propagate_nan else np.nanmean

    m = central(data, axis=axis, **kwargs)
    s = std(data, axis=axis, **kwargs)

    if ci:
        s *= 1.96 # 95% CI for gaussian data

    if sd:
        return m, s

    n = data.shape[axis] if propagate_nan else np.sum(~np.isnan(data), axis=axis)
    s = s / np.sqrt(n)

    return m, s

def circmean(x, unit='degree', data='orientation', **kwargs):
    """mean orientation / direction for circular data
    
    Inputs
    -------
        unit : 'radian' or 'degree'
            unit of circular data used for input and output
        data : 'orientation' ([0,180]) or 'direction' ([0,360])
            type of circular data used for input and output

    Returns
    -------
        m : circular mean estimate
    """

    # make x in the range of [0, 2pi]
    if unit == 'radian':
        if data == 'orientation':
            x = x*2.
    elif unit == 'degree':
        if data == 'orientation':
            x = x*np.pi/90.
        elif data == 'direction':
            x = x*np.pi/180.
    
    # compute circular mean
    s, c = np.mean(np.sin(x),**kwargs), np.mean(np.cos(x),**kwargs)
    m    = np.arctan2(s, c)

    # convert m to data/unit
    if unit == 'radian':
        if data == 'orientation':
            m /= 2.
    elif unit == 'degree':
        if data == 'orientation':
            m /= np.pi/90.
        elif data == 'direction':
            m /= np.pi/180.

    return m

def circcorr(x1, x2, uniform=False,
             unit='degree', data='orientation', **kwargs):
    """correlation for orientation / direction data
        Jammalamadaka & Sengupta (2001)
        if x1, x2 are uniform distributed, circular means m1, m2 are not well-defined.
        in such case, circular mean m1 chosen as 0 and m2 chosen as circmean(x2-x1).

    inputs
    ------
        uniform : whether both x1 and x2 are approximately uniform-distributed.
    """ 

    if unit == 'radian':
        if data == 'orientation':
            x1 = x1*2.
            x2 = x2*2.
    elif unit == 'degree':
        if data == 'orientation':
            x1 = x1*np.pi/90.
            x2 = x2*np.pi/90.
        elif data == 'direction':
            x1 = x1*np.pi/180.
            x2 = x2*np.pi/180.

    # now x1, x2 are in (radian, direction)
    if uniform:
        m1 = 0
        m2 = circmean(x2-x1, unit='radian', data='direction', **kwargs)

    else:
        m1 = circmean(x1, unit='radian', data='direction', **kwargs)
        m2 = circmean(x2, unit='radian', data='direction', **kwargs)

    numer = np.sum( np.sin(x1 - m1) * np.sin(x2 - m2) )
    denom = np.sqrt( np.sum(np.sin(x1 - m1)**2) * np.sum(np.sin(x2 - m2)**2) )
    return numer / denom if denom != 0 else np.nan


def pearson_CI(x, y, alpha=0.05):
    """Pearson's correlation coefficient confidence interval
      https://stackoverflow.com/questions/33176049/how-do-you-compute-the-confidence-interval-for-pearsons-r-in-python
    """
    assert len(x) == len(y), 'Not available'

    n = len(x)
    r = np.corrcoef(x,y)[0,1]
    
    def r_to_z(r):
        return np.log((1 + r) / (1 - r)) / 2.0

    def z_to_r(z):
        e = np.exp(2 * z)
        return((e - 1) / (e + 1))

    z = r_to_z(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))

def isoprob_ellipse(data, prob, fill=True):
    """iso-Gaussian-probability line

    inputs:
        data [2, n] : each column represents a data point
        prob : Probability level for the isoprobability contour
        fill : returns parameters for an ellipse patch

    returns:
        ellipse_points : 2D array
            contour points of the ellipse
        ellipse_params : dict, optional
            parameters for an ellipse patch if fill is True
    """
    # fit Gaussian
    m_fit = np.mean(data, axis=0)
    c_fit = np.cov(data, rowvar=False)
    evals, evecs = np.linalg.eig(c_fit)
    lams  = np.sqrt(evals)
    
    # ellipse points
    theta = np.linspace(0, 2 * np.pi, 250)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    scale_factor = np.sqrt(chi2.ppf(prob, df=2))
    ellipse_points = scale_factor * evecs @ np.diag(lams) @ unit_circle
    ellipse_points = ellipse_points + m_fit[:, np.newaxis]
    
    if fill:
        ellipse_params = {
            'xy'     : m_fit,
            'width'  : lams[0] * 2*np.sqrt(chi2.ppf(prob, df=2)),
            'height' : lams[1] * 2*np.sqrt(chi2.ppf(prob, df=2)),
            'angle'  : np.rad2deg(np.arctan2(evecs[1,0], evecs[0,0]))
        }
        return ellipse_points, ellipse_params
    
    return ellipse_points


def pop_vector_decoder(x, 
                       labels,
                       label_unit='degree', 
                       label_data='orientation',
                       axis=-1,
                       **kwargs):
    """circular 'labeled-line' decoding from the channel or unit response vectors, given labels.

    inputs
    ----------
        x : array-like, (..., n_channel, ...)
            population 'neural' signals for each of channels or units
        labels : (..., n_channel, ...) 
            circular 'labels' in the specified unit (degree/radian)

    returns
    -------
        decoded : array-like 
            decoded labels

    examples
    --------
    >>> y = pop_vector_decoder(x, labels) # x [t,n,n_chan], labels [n_chan]
    y.shape # [t,n]

    >>> y = pop_vector_decoder(x, labels) # x [t,n,n_chan], labels [1,n_chan]
    y.shape # [t]
    """
    if label_unit == 'radian':
        if label_data == 'orientation':
            labels = labels*2.
    elif label_unit == 'degree':
        if label_data == 'orientation':
            labels = labels*np.pi/90.
        elif label_data == 'direction':
            labels = labels*np.pi/180.

    x = np.moveaxis(x, axis, -1)
    labels = np.moveaxis(labels, axis, -1)

    # compute population vector labels using numpy broadcasting (leading dimensions) 
    axes_labels = tuple(range(-len(labels.shape),0))
    xsin = np.sum( x * np.sin(labels),axis=axes_labels )
    xcos = np.sum( x * np.cos(labels),axis=axes_labels )
    decoded = np.arctan2(xsin, xcos)

    # convert decoded to data/unit
    if label_unit == 'radian':
        if label_data == 'orientation':
            decoded /= 2.
    elif label_unit == 'degree':
        if label_data == 'orientation':
            decoded /= np.pi/90.
        elif label_data == 'direction':
            decoded /= np.pi/180.

    return decoded


def collapse(x, 
             collapse_groups,
             collapse_axis=-1, 
             collapse_func=np.mean,
             collapse_kwargs={},
             return_list=False,
             return_labels=False,
             ):
    """collapse the input in a stacked format given the groups labels.

    inputs
    -------
        x [..., n, ...] : list of arrays to collapse.
        collapse_groups [list] : group labels, same length (n) as the collapse axis
        collapse_axis : axis to collapse the data. if 0, collapse the first axis
        func : function to collapse the data

    returns
    -------
        res : collapsed data [n_group1][n_group2]...[n_groupN][ collapse_func(x_subgroup,...) ]
    """
    if not isinstance(collapse_groups, list):
        collapse_groups = [collapse_groups]

    if not isinstance(x, (list, tuple)): x = [x]

    # unique combinations of group labels
    i_groups = []
    l_groups = []
    for group in collapse_groups:
        l_group, i_group = np.unique(group, return_inverse=True)
        l_groups.append(l_group)
        i_groups.append(i_group[:, None])
    i_groups = np.concatenate(i_groups,axis=-1)
    u_groups = np.unique(i_groups, axis=0)

    # collapse
    res = dict()
    for u_group in u_groups:
        l_group = [l[u] for u,l in zip(u_group,l_groups)]
        idx = np.all(i_groups == u_group, axis=1)
        set_dict(
            res, l_group, 
            collapse_func(
                *[np.compress(idx, _x, axis=collapse_axis) for _x in x],
                **collapse_kwargs
            ),
        )

    if return_list:
        res = recursive_list(res)
    
    if return_labels:
        labels = []
        for u_group in u_groups:
            l_group = [l[u] for u,l in zip(u_group,l_groups)]
            labels.append(l_group)
        return res, labels

    return res


def find_roots(x, y, circular=True):
    """find approximate x-values (roots) of a continuous function y such that y(x)=0

    inputs
    -------
        x : x-values in the acsending order
        y : y-values corresponding to x-values
        circular : whether y is circular (wrap around the first value)
    """

    if circular:
        x = np.concatenate([x, x[:1]])
        y = np.concatenate([y, y[:1]])

    # identify points where y changes sign (indicating a root is between two points).
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    diffx = wrap(np.diff(x), period=180.)
    
    # compute the x-values where the sign change occurs, approximating the root positions.
        # x[:-1][s] : x-values at the start of each sign-changing interval.
        # np.diff(x)[s] : distance between consecutive x-values where sign change occurs.
    return x[:-1][s] + diffx[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def find_fixed_points(ssb_fun, x=None, dx=None):
    """find fixed points of stimulus-specific bias function

    inputs
    -------
        ssb_fun : stimulus-specific bias function (circular)
        x : stimulus range. If None, use the default stimulus list.
        dx : step size for numerical differentiation

    returns
    -------
        fixed_points : fixed points estimates of the stimulus-specific bias function
        gradients : gradients around the fixed points
    """
    if x is None: 
        x = np.linspace(0, 180, 96, endpoint=False)
    if dx is None:
        dx = x[1] - x[0]
    
    kappa = ssb_fun(x)
    kappa = kappa / np.max(np.abs(kappa))

    # find the fixed points
    fixed_points = find_roots(x, kappa)

    # compute the gradient at the nearest x-values to the fixed points
    idx = np.abs(x - fixed_points.reshape((-1, 1))).argmin(axis=-1)
    gradients = np.gradient( kappa, dx )[idx]

    return fixed_points, gradients


def find_critical_stimuli(ssb_fun, 
                          stim_list,
                          reject_overlap=True,
                          tol=8.0,
                          grad_tol=0.0,
                          x=None, 
                          dx=None):
    """find converging and diverging stimuli based on the stimulus-specific bias function

    inputs
    -------
        ssb_fun : stimulus-specific bias function (circular)
        stim_list : stimulus list among which to find the critical stimuli
        reject_overlap : whether to reject the cross-overlapping critical stimuli
        tol : tolerance for labeling the stimuli as converging or diverging (in degree)

    returns
    -------
        dict('converging', 'diverging') : critical stimuli
    """
    # 1. find the fixed points
    fixed_points, gradients = find_fixed_points(ssb_fun, x=x, dx=dx)

    # 2. find the critical stimuli
    div_stimuli  = [] # diverging stimuli
    conv_stimuli = [] # converging stimuli
    for fxpt, grad in zip(fixed_points, gradients):
        dist_to_fxpt   = np.abs( wrap(stim_list - fxpt, period=180.) )
        within_tol     = dist_to_fxpt < tol
        critical_stims = stim_list[within_tol]
        if grad > grad_tol:
            div_stimuli.append(critical_stims)
        elif grad < -grad_tol:
            conv_stimuli.append(critical_stims)
    
    div_stimuli  = np.array([]) if len(div_stimuli) == 0  else np.unique(np.concatenate(div_stimuli))
    conv_stimuli = np.array([]) if len(conv_stimuli) == 0 else np.unique(np.concatenate(conv_stimuli))

    # 3. check for overlapping critical stimuli
    if reject_overlap:
        overlap = np.intersect1d(div_stimuli, conv_stimuli)
        div_stimuli  = np.setdiff1d(div_stimuli, overlap)
        conv_stimuli = np.setdiff1d(conv_stimuli, overlap)

    return {
        'diverging'  : div_stimuli,
        'converging' : conv_stimuli
    }