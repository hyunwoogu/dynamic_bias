"""
Utility functions for plotting.
"""
import logging
import colorsys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from pathlib import Path
from .dataset import mkdir, ORIGIN

LABEL = True # label axes 
HUSL  = mc.ListedColormap(sns.color_palette("husl",24))
BLUES = mc.ListedColormap(sns.color_palette("GnBu",12))
TWLT  = mc.ListedColormap(sns.color_palette("twilight",24))
GRAYS = mc.ListedColormap(sns.color_palette("gray_r",5))
SPECTRAL = mc.ListedColormap(sns.color_palette("Spectral_r",12))
E_COLOR = np.array([51, 99, 141])/255.
L_COLOR = np.array([41, 175, 127])/255.
DIR_FIGURE = str(Path(ORIGIN+'/figures'))
mkdir(DIR_FIGURE)

def setup_matplotlib(setup_type='article'):
    """setup matplotlib parameters for publishable quality figures
    https://github.com/adrian-valente/populations_paper_code
    """
    log = logging.getLogger("fontTools.subset")
    log.setLevel(logging.CRITICAL)

    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['axes.titlepad'] = 24
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Helvetica Neue'

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

def interp_colormap(color_l=[1,1,1,0], *, 
                    color_u, n=256):
    """naively interpolates between two RGBA colors 
        starting from color_l and ending at color_u
    """
    color_l = np.array(color_l)
    color_u = np.array(color_u)
    lspace  = np.linspace(0,1,num=n)[:,None]
    rgba    = (1-lspace)*color_l + lspace*color_u
    return ListedColormap(rgba)

def lighten_color(c, intensity=1):
    """given color RGB tuple, update luminosity
         luminosity <-  1 - intensity * (1-luminosity) 
    """
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - intensity*(1 - c[1]), c[2])

def kde1d(x, x_data, bw=0.1):
    """compute 1D kernel density estimate"""
    kde = gaussian_kde(x_data, bw_method=bw)
    return kde(x)

def draw_publish_axis(ax, xrange, yrange, xticks, yticks, xwidth=2.5, ywidth=2, tight_layout=True):
    """inspired by Justin L. Gardner's MATLAB code"""
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
    if tight_layout:
        plt.tight_layout()

def se_line(x, x_data, scale, n=50):
    """compute standard error line given x"""
    mx = np.mean(x_data)
    se = scale*(1/n + (x-mx)**2/np.sum((x_data-mx)**2))
    se = np.sqrt(se)
    return se

def centered_ellipse(par, x,y):
    """centered ellipse loss function"""
    a,b,th = par
    t1 = (x*np.cos(th)+y*np.sin(th))**2 / a**2 
    t2 = (x*np.sin(th)-y*np.cos(th))**2 / b**2     
    loss = np.sum((1-(t1+t2))**2)
    return loss

def fit_centered_ellipse(x, y):
    res = minimize(centered_ellipse, [1,1,0], args=(x,y),
                   bounds=[[0, np.inf],[0, np.inf],[-np.pi, np.pi]])['x']
    return {'w': res[:2], 'phi': res[2]}

def harmonics(freq=1, phase=0, num=24, endpoint=False):
    """generate a 2D matrix of sine and cosine harmonics based on given frequency and phase"""
    t = np.linspace(0, 2*np.pi, num=num, endpoint=endpoint)
    cos_wave = np.cos(freq * t + phase)
    sin_wave = np.sin(freq * t + phase)
    return np.vstack((cos_wave, sin_wave))

def rotate(vec, phi=0, w=[1.,1.], flip=None, late_flip=True):
    """warped rotation transformation
        vec [n,2] : input vector
    """
    vec = np.array(vec)
    
    # rotation matrix
    R = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)],
    ])

    # flip around the state space axes
    if flip is not None:
        if late_flip:
            if 'x' in list(flip):
                R = np.array([[-1.,0],[0,1.]]) @ R
            if 'y' in list(flip):
                R = np.array([[1.,0],[0,-1.]]) @ R
        else:
            if 'x' in list(flip):
                R = R @ np.array([[-1.,0],[0,1.]])
            if 'y' in list(flip):
                R = R @ np.array([[1.,0],[0,-1.]])

    # warping
    W = np.array([
        [w[0], 0],
        [0, w[1]]
    ])
    return (R @ W @ (vec.T)).T
    
def fit_rotation(x, n=24):
    """fit rotation given [n,2] basis vectors based on dot product with harmonics"""
    h = harmonics(num=n).T
    f1 = lambda phi : -np.sum(x * rotate( h, phi[0] ))
    f2 = lambda phi : -np.sum(x * rotate( h, phi[0], flip=['x'] ))
    fit1 = minimize(f1, [0], bounds=[[-np.pi, np.pi]]) # ccw direction
    fit2 = minimize(f2, [0], bounds=[[-np.pi, np.pi]]) # cw direction
    
    if fit1.fun < fit2.fun:
        return fit1.x[0], []
    else:
        return fit2.x[0], ['x']

def arrow(ps, pe, project=True):
    """compute the arrow vector from ps to pe"""
    if project:
        tgntslp = -ps[0]/ps[1]    
        incrvec = np.array([1,tgntslp]) / np.sqrt(1+tgntslp**2)
        incrmnt = np.dot( pe-ps, incrvec ) 
        dp      = incrmnt*incrvec

    else:
        dp = pe - ps
        dp = dp / np.linalg.norm(dp)
    
    return ps, dp