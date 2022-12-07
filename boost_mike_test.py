import sys
import scri
import sxs
import numpy as np
import quaternion
from quaternion import rotate_vectors
from matplotlib import pyplot as plt
from scri import flux
import functools
from scri import waveform_base
from scipy.integrate import trapz as integrate

SA_path="/home/khushal/Documents/Python/SXS data/SXS:BBH:0126/"
Waveform_path="rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N4.dir"
md = sxs.metadata.Metadata.from_file(SA_path + "metadata.txt")    
h = scri.SpEC.read_from_h5(SA_path +  Waveform_path)
M = md['remnant_mass']

s = np.array([1.0,0.0,0.0])
x = []
errors = []

def lorentz_transformation(v, w, t,M):
    beta = np.linalg.norm(v)
    g_m = (1/(1 - beta**2))**(1/2)
    Boost_matrix = [[g_m , - g_m * beta, 0,0], [ -g_m * beta, g_m, 0, 0], [0,0,1,0],[0,0,0,1]]    
    for j in range(3):
        M[0,j+1,:] = w[:,j]
        M[j+1,0,:] = M[0,j+1,:]
    M = np.einsum('mi,ilk->mlk', Boost_matrix, np.einsum('ijk, jl -> ilk', M, Boost_matrix)) 
    return M

for β in range(1,10):
    x.append(10**(-β))
    v = 10**(-β)*s    
    hp = h.transform(boost_velocity=v)
    t = len(hp.t)    
    w1 = scri.boost_flux(hp)
    w2 = scri.boost_flux(h.interpolate(hp.t))
    M = np.zeros((4,4,t))
    B_M = lorentz_transformation(v, w2, t, M)
    w2 = np.c_[B_M[0,1,:],B_M[0,2,:],B_M[0,3,:]]
    error = np.linalg.norm((w1 - w2),axis = 1)
    errors.append(np.abs(integrate(error, hp.t)) / (hp.t[-1] - hp.t[0]))
    
plt.xlabel("Order of magnitude of translation")
plt.ylabel("Time averaged error")
plt.title("Boost flux")
plt.loglog(np.transpose(x),errors,'+')
plt.show()