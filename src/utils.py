""" Utility functions
    Author: Hyunwoo Gu
"""
import os
import operator
import subprocess
from functools import reduce
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.lines import Line2D
from scipy.stats import vonmises, gamma, norm

EPS = 1e-10

# ========================================
# dataset
# ========================================
PATH = {}
PATH['data'] = {}
PATH['data']['external'] = "https://www.dropbox.com/s/ekvx5ko9swv0qfz/external.zip"
PATH['data']['processed'] = {}
PATH['data']['processed']['behavior'] = "https://www.dropbox.com/s/lny8tvz94te1pp7/behavior.zip"
PATH['data']['processed']['fmri']     = "https://www.dropbox.com/s/7juuwrhdzbz0kqy/fmri.zip"
PATH['data']['outputs'] = {}
PATH['data']['outputs']['behavior']   = "https://www.dropbox.com/s/aaf0cjw30w8p7gh/behavior.zip"
PATH['data']['outputs']['fmri']       = "https://www.dropbox.com/s/d52l5jg1mtzfb5a/fmri.zip"
PATH['data']['outputs']['rnn']        = "https://www.dropbox.com/s/tcluude70xo7omv/rnn.zip"
PATH['data']['outputs']['ddm']        = "https://www.dropbox.com/s/jkyo5jw3vykdgs9/ddm.zip"
PATH['models'] = {}
PATH['models']['ddm'] = {}
PATH['models']['ddm']['reduced']      = "https://www.dropbox.com/s/7xtbejv58zieubd/reduced.zip"
PATH['models']['ddm']['full']         = "https://www.dropbox.com/s/hzi2ac20ihnmn3g/full.zip"
PATH['models']['rnn'] = {}
PATH['models']['rnn']['hom']          = "https://www.dropbox.com/s/6xmp1qpsdyel6zm/hom.zip"
PATH['models']['rnn']['het']          = "https://www.dropbox.com/s/arn3vpinmrabe0h/het.zip"
PATH['models']['rnn']['het_emonly']   = "https://www.dropbox.com/s/rax9aylmu44rzou/het_emonly.zip"
ORIGIN = str(Path(os.path.abspath(__file__)).parent.parent.absolute())

def getFromDict(dataDict, mapList):
    """get value from nested dictionary
    https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    """
    return reduce(operator.getitem, mapList, dataDict)

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def download_dataset(path, cmd='curl'):
    """download dataset from the path"""
    path_strip  = path.strip('/')
    path_split  = path_strip.split('/')
    path_parent = '/'.join(path_split[:-1])
    data_url    = getFromDict(PATH, path_split)
    
    print(f"downloading {path_strip}...")
    if os.path.exists(f"{ORIGIN}/{path_strip}"):
        print(f"{path_strip} already exists. Skipping download...")
    else:
        mkdir(f"{ORIGIN}/{path_parent}")
        if cmd == 'wget':
            command  = f"wget {data_url} -P {ORIGIN}/{path_parent} -c -nc;"
            command += f"unzip -o {ORIGIN}/{path_strip}.zip -d {ORIGIN}/{path_parent};"
        elif cmd == 'curl':
            command  = f"curl -L -o {ORIGIN}/{path_strip}.zip {data_url};"
            command += f"unzip -o {ORIGIN}/{path_strip}.zip -d {ORIGIN}/{path_parent};"
        command += f"rm {ORIGIN}/{path_strip}.zip"
        print(command)
        subprocess.run(command, capture_output=True, shell=True)
        print(f"downloaded {path_strip}.")

# ========================================
# plotting
# ========================================
LABEL = False # label axes 
HUSL  = mc.ListedColormap(sns.color_palette("husl",24))
BLUES = mc.ListedColormap(sns.color_palette("GnBu",12))
TWLT  = mc.ListedColormap(sns.color_palette("twilight",24))
E_COLOR = np.array([41, 175, 127])/255.
L_COLOR = np.array([51, 99, 141])/255.
DIR_FIGURE = str(Path(ORIGIN+'/figures'))
mkdir(DIR_FIGURE)

def setup_matplotlib(setup_type='article'):
    """setup matplotlib parameters for publishable quality figures
    https://github.com/adrian-valente/populations_paper_code
    """
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['axes.titlepad'] = 24
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Helvetica'

    if setup_type=='article':
        plt.rcParams['axes.labelsize'] = 19
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['font.size'] = 14
    elif setup_type=='poster':
        plt.rcParams['axes.labelsize'] = 25
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['font.size'] = 18

def set_size(size, ax=None):
    """set size of the figure, without changing the aspect ratio
    https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    """
    if not ax: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    w, h = size
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def draw_publish_axis(ax, xrange, yrange, xticks, yticks, xwidth=2.5, ywidth=2):
    _xmin, _ = ax.get_xaxis().get_view_interval()
    _ymin, _ = ax.get_yaxis().get_view_interval()
    if xrange is not None:
        ax.add_artist(Line2D(xrange, (_ymin,_ymin), color='black', linewidth=xwidth, solid_capstyle='butt', fillstyle='full'))
    if yrange is not None:
        ax.add_artist(Line2D((_xmin,_xmin), yrange, color='black', linewidth=ywidth, solid_capstyle='butt', fillstyle='full'))
    ax.set_frame_on(False)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    plt.tight_layout()

# ========================================
# statistics
# ========================================
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

def psi(x, m, s, lam):
    return lam + norm.cdf(x, loc=m, scale=s) * (1.-lam*2)

def nan(size, **kwargs):
    """make a nan matrix of the given size"""
    return np.full(size, np.nan, **kwargs)

def wrap(x, period=2*np.pi):
    """wrap the circular data to the range of [0, 2pi]"""
    return (x - period/2.) % (period) - period/2.

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

# ========================================
# BOLD data decoding & modeling
# ========================================
def pop_vector_decoder(x, labels, unit='degree', data='orientation'):
    """circular 'labeled-line' decoding from the channel or unit response vectors, given 'labels'.

    Inputs
    ----------
        x : array-like, (..., n_channel)
            population 'neural' signals for each of channels or units
        labels : (n_channel,) 
            circular 'labels' in the specified unit (degree/radian) and data (orientation/direction)

    Returns
    -------
        decoded : array-like 
            decoded labels
    
    References
    Georgopoulos, A. P., Schwartz, A. B., & Kettner, R. E. (1986). Neuronal population coding of movement direction. Science, 233(4771), 1416-1419.
    """

    if unit == 'radian':
        if data == 'orientation':
            labels = labels*2.
    elif unit == 'degree':
        if data == 'orientation':
            labels = labels*np.pi/90.
        elif data == 'direction':
            labels = labels*np.pi/180.

    # compute population vector labels
    s, c    = np.sin(labels), np.cos(labels)
    decoded = np.arctan2(x @ s, x @ c)

    # convert decoded to data/unit
    if unit == 'radian':
        if data == 'orientation':
            decoded /= 2.
    elif unit == 'degree':
        if data == 'orientation':
            decoded /= np.pi/90.
        elif data == 'direction':
            decoded /= np.pi/180.

    return decoded

def asvoid(arr):
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

def inNd(a, b, assume_unique=False):
    a = asvoid(a)
    b = asvoid(b)
    return np.in1d(a, b, assume_unique)

def column_roll(A,shift):
    """numpy roll in a specfic axis"""
    n_row = len(A)
    for i in range(n_row):
        A[i] = np.roll(A[i],shift)
    return A

def glover(t):
    a1, a2, b1, b2, c = 6., 16., 1., 1., 1/6.
    h1 = gamma.pdf(t, a=a1, scale=1./b1)
    h2 = gamma.pdf(t, a=a2, scale=1./b2)
    h  = h1 - c*h2
    return h

def conv_operator(Ts,Us,ds):
    t0 = min(Ts[0],  Us[0])
    t1 = max(Ts[-1], Us[-1])
    tt = np.arange(t0,t1+EPS,step=ds)

    conv_matrix = np.zeros((len(tt), len(tt)))
    for i_row in range(len(tt)-1,-1,-1):
        conv_matrix[i_row,:(i_row+1)] = glover(tt)[::-1][((len(tt)-1)-i_row):]

    idx_i = np.argmin(np.abs(tt.reshape((-1,1)) - Us.reshape((1,-1))), axis=0)
    idx_j = np.argmin(np.abs(tt.reshape((-1,1)) - Ts.reshape((1,-1))), axis=0)

    return conv_matrix[idx_i][:,idx_j]
    
# ========================================
# stimulus-specific bias
# ======================================== 
def derivative_von_mises(x, mu, kappa):
    """derivative of von-Mises density functions"""
    return (kappa*np.sin(mu-x)*np.exp(kappa*np.cos(mu-x)) / (np.i0(kappa)*2*np.pi)) 
    
def stimulus_specific_bias(x, w, n_basis=12, p_basis=6.):
    """estimate smooth stimulus-specific bias function κ 
        based on the measured at n different stimuli
        
    Intputs
    ----------
        x (size n) : stimulus (in degree, [0, 180])
        w (size m) : weights of the von-Mises basis functions
        p_basis : precision parameters of basis functions
    
    Returns
    -------
        kappa_hat : stimulus-specific bias function evaluated at x (in degree)
    """

    c_basis = np.linspace(0, 2*np.pi, n_basis, endpoint = False)
    
    x    = (x*2.*np.pi/180).reshape((-1,1))
    X    = derivative_von_mises(x, c_basis, p_basis)
    kappa_hat = w[0] + np.sum(X*w[1:], axis=-1)
    
    return kappa_hat


# ========================================
# Euler-Maruyama simulation 
# ======================================== 
# parameters
par = {
    # hyperparameters
    'T'      : 18,            # total time of interest  
    'ds'     : np.pi/48.,     # stimulus space discretization
    'stims'  : np.linspace(0,2*np.pi, num=24, endpoint=False), # stimulus space
    'relref' : np.array([-21,-4,0,4,21]) * np.pi/90.,          # reference space
    'thres'  : np.pi/10,      # threshold that divides near & far references
    'sig'    : 0.2,           # smoothness of decision-conditioning mask
    'n_time' : 10,            # number of time discretization for 1sec
    'n_trial': 100,           # number of particles simulated
    'eps'    : 1e-10,         # small number for numerical stability

    # parameters (placeholder) (degree-space)
    'w_K'    : 0.5,
    'w_E'    : 1.5,
    'w_P'    : 0.0,
    'w_D'    : 0.0,
    'w_r'    : 0.0,
    'w_b'    : 0.5,
    'lam'    : 0.0,
    's'      : 0.5,

    # stimulus-specific bias (drift) function
    'K'      : lambda x: 0,

    # encoding functions
    'F'      : lambda x: x,
    'F_inv'  : lambda x: x,
}


def run_euler(dmT, **par):
    """Simulate drift-diffusion dynamics using Euler-Maruyama method
        dmT: timing of decision-making
    """

    # relevant functions
    K     = par['K'] # drift function
    F     = par['F'] # stimulus-to-measurement function
    F_inv = par['F_inv'] # measurement-to-stimulus function

    # relevant parameters (in radians)
    lam = par['lam']
    w_K = par['w_K']*np.pi/90.
    w_E = par['w_E']*np.pi/90.
    w_P = par['w_P']*np.pi/90.
    w_D = par['w_D']*np.pi/90.
    w_r = par['w_r']*np.pi/90.
    w_b = par['w_b']*np.pi/90.

    # time and space discretization
    ts  = np.linspace(0,par['T'],par['n_time']*par['T']) # time space
    dt  = ts[1] - ts[0]
    dmI = abs(ts - dmT).argmin()
    stim_wrap = wrap(par['stims'])

    # results placeholder
    res_e  = nan((len(par['stims']),len(par['relref']),par['n_trial'])) # encoding
    res_m  = nan((len(par['stims']),len(par['relref']),par['n_time']*par['T'],par['n_trial'])) # memory
    res_dm = nan((len(par['stims']),len(par['relref']),par['n_trial'])) # decision

    # encoding simulation
    p_e = np.sqrt(1./np.clip(w_E, par['eps'], None)) # encoding precision
    for i_theta, v_theta in enumerate(stim_wrap):
        m0 = vonmises.rvs(loc=F(v_theta), kappa=p_e, size=len(par['relref'])*par['n_trial'])
        m0 = wrap(F_inv(m0 % (2*np.pi)))
        res_e[i_theta] = m0.reshape((len(par['relref']),par['n_trial']))

    # memory simulation
    for i_s, v_s in enumerate(par['stims']):
        for i_r, v_r in enumerate(par['relref']):
            for i_t, _ in enumerate(ts): 
                
                if i_t == 0: 
                    # encoding
                    res_m[i_s, i_r, i_t, :] = res_e[i_s, i_r, :]
                else:
                    # memory dynamics
                    _th  = wrap(res_m[i_s, i_r, i_t-1, :])
                    _dth = w_K*K(_th)*dt + w_D*np.sqrt(dt)*np.random.normal(size=par['n_trial'])
                    res_m[i_s, i_r, i_t, :] = _th + _dth

                if i_t == dmI:
                    # decision-making
                    m_r = v_r + np.random.normal(scale=par['sig'], size=par['n_trial'])
                    dist = wrap(res_m[i_s, i_r, i_t, :]-v_s-m_r)
                    res_dm[i_s, i_r, :] = np.where(dist > 0, 1., -1.)

                    if abs(v_r) > par['thres']:
                        res_m[i_s, i_r, i_t, :] += w_b*res_dm[i_s, i_r, :] + w_r*v_r
                    else:
                        res_m[i_s, i_r, i_t, :] += w_b*res_dm[i_s, i_r, :] + w_r*v_r
                        
    res_m = wrap(res_m)

    # production simulation
    res_p = res_m[:,:,-1,:]
    res_p = res_p + w_P*np.random.normal(size=res_p.shape)
    res_p = wrap(res_p)

    # lapse
    if lam > 0:
        lap_idx = np.random.choice(2, size=res_dm.shape, p=[1.-lam*2.,lam*2.])
        res_dm[lap_idx==1] = np.random.choice([-1.,1.], size=np.sum(lap_idx))

    return res_e, res_m, res_p, res_dm
