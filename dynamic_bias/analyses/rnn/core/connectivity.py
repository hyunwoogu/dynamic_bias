"""utility functions for inspecting connectivity
"""
import numpy as np
from scipy.optimize import minimize

vec_t0 = np.linspace(0,np.pi*2,num=24,endpoint=False)
vec_c0 = np.cos(vec_t0)
vec_s0 = np.sin(vec_t0)
mat_0  = np.array([vec_s0, vec_c0])
V      = mat_0.T

def make_template( theta,w1,w2,b ) :
    vec_t1 = theta+np.linspace(0,np.pi*2,num=24,endpoint=False)
    vec_c1 = np.cos(vec_t1)
    vec_s1 = np.sin(vec_t1)
    mat_1  = np.array([vec_s1, vec_c1])
    ident  = np.array([[w1,0],[0,w2]])
    return mat_0.T @ ident @ mat_1 + b

def frob_loss(params, J_true, hom=True):
    if hom:
        th,w,b = params
        J_pred = make_template( th, w, w,  b )
    else:
        th,w1,w2,b = params
        J_pred = make_template( th, w1,w2, b )
    loss   = np.sum( (J_true-J_pred)**2 )
    return loss

def rot_pham(th, w1, w2=None, hom=True):
    # positive phase means counter-clockwise direction
    if w1 > 0:
        phase = (th-np.pi) % (2*np.pi) - np.pi
    else:
        phase = th % (2*np.pi) - np.pi
        
    if hom:
        ampli = np.abs(w1)
    else:
        ampli = np.abs(w1), np.abs(w2)
        
    return (phase, ampli)
    
def lowrank_J(J, hom=True, return_rot_info=True):
    J_true = {}
    J_true['J00'] =  J['J11'][:24,:24]
    J_true['J01'] =  J['J11'][:24,24:]
    J_true['J10'] =  J['J11'][24:,:24]
    J_true['J11'] =  J['J11'][24:,24:]
    J_true['J02'] = (J['J12'][:24,:24]+J['J12'][:24,24:])/2.
    J_true['J12'] = (J['J12'][24:,:24]+J['J12'][24:,24:])/2.
    J_true['J20'] = (J['J21'][:24,:24]+J['J21'][24:,:24])/2.
    J_true['J21'] = (J['J21'][:24,24:]+J['J21'][24:,24:])/2.
    J_true['J22'] = (J['J22'][:24,:24]+J['J22'][:24,24:]+J['J22'][24:,:24]+J['J22'][24:,24:])/4.

    J_pred = {}
    R_info = {}
    for k_J, v_J in J_true.items():
        if hom:
            x0     = [0., 1., 0.]
            args   = v_J
            bounds = [[-np.pi,+np.pi],[-5.,5.],[-5.,5.]]
        else:
            x0     = [0., 1., 1., 0.]
            args   = (v_J, False)
            bounds = [[-np.pi,+np.pi],[-5.,5.],[-5.,5.],[-5.,5.]]
        
        res = minimize(
            fun    = frob_loss,
            x0     = x0,
            args   = args,
            bounds = bounds
        )
        
        if hom:
            J_pred[k_J] = make_template( res['x'][0], res['x'][1], res['x'][1], res['x'][2] )
        else:
            J_pred[k_J] = make_template( res['x'][0], res['x'][1], res['x'][2], res['x'][3] )

        if return_rot_info:
            R_info[k_J] = rot_pham( *res['x'][:-1], hom=hom )
            
    return J_pred, R_info