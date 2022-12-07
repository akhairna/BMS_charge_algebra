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
from scipy.optimize import curve_fit

SA_path="/home/khushal/Documents/Python/SXS data/SXS:BBH:0126/"
Waveform_path="rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N4.dir"
md = sxs.metadata.Metadata.from_file(SA_path + "metadata.txt")    
h = scri.SpEC.read_from_h5(SA_path +  Waveform_path)
M = md['remnant_mass']

t = 1.0
x = []
errors = []
             
for β in range(1,10):
    x.append(10**(-β))
    hp = h.transform(time_translation=10**(-β)*t)
    w1 = scri.boost_flux(hp)
    w2 = scri.boost_flux(h)
    error = np.linalg.norm((w1 - w2),axis = 1)
    errors.append(np.abs(integrate(error, hp.t)) / (hp.t[-1] - hp.t[0]))
        
def func(x, a, b, c):
    return a * np.exp(-b * x) + c    
    
xdata = np.transpose(x)    
ydata = errors    
popt, pcov = curve_fit(func, xdata, ydata)    
plt.xlabel("Order of magnitude of translation")
plt.ylabel("Time averaged error")
plt.title("Boost flux")
plt.loglog(xdata, func(xdata, *popt),'r-',errors,'+',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.show()




