import scri
import sxs
import spherical_functions as sf
from spherical_functions import LM_index
from matplotlib import pyplot as plt
import numpy as np
from spherical_functions import clebsch_gordan as CG
from scri import flux
import functools
import numpy as np
import quaternion
from quaternion import rotate_vectors    

SA_path="/home/khushal/Documents/Python/SXS data/SXS:BBH:0023/"
Sim_path=""
Waveform_path="rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N2.dir"
md = sxs.metadata.Metadata.from_file(SA_path + Sim_path + "metadata.txt")    

h = scri.SpEC.read_from_h5(SA_path + Sim_path + Waveform_path)
R = np.quaternion(10,57,12,4).normalized()
M = quaternion.as_rotation_matrix(~R)


E_dot = scri.energy_flux(h)
P_dot = np.transpose(np.matmul(M, np.transpose(scri.momentum_flux(h)))) 
L_dot = np.transpose(np.matmul(M, np.transpose(scri.angular_momentum_flux(h))))
B_dot = np.transpose(np.matmul(M, np.transpose(scri.boost_flux(h))))


hp = h.rotate_decomposition_basis(R)
E_dotp = scri.energy_flux(hp)
P_dotp = scri.momentum_flux(hp) 
L_dotp = scri.angular_momentum_flux(hp)
B_dotp = scri.boost_flux(hp)     
    
Q_dot = np.c_[E_dot, P_dot, L_dot, B_dot]
Q_dotp = np.c_[E_dotp, P_dotp, L_dotp, B_dotp]

C = Q_dot - Q_dotp
D = Q_dotp + Q_dot

x = hp.t
y = np.log(np.abs(C/D))

fig, axs = plt.subplots(2, 2)
fig.suptitle('Rotation')
axs[0,0].plot(x, (y[:,0]))
axs[0,0].set_title('Energy')
axs[0, 1].plot(x, y[:,1:4], 'tab:orange')
axs[0, 1].set_title('Momentum')
axs[1, 0].plot(x, y[:,4:7], 'tab:green')
axs[1, 0].set_title('Angular Momentum')
axs[1, 1].plot(x, y[:,7:], 'tab:red')
axs[1, 1].set_title('Boost')

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Fractional error')

plt.plot()
plt.show()
