import sys
import scri
import sxs
import numpy as np
import quaternion
from matplotlib import pyplot as plt

SA_path="/home/khushal/Documents/Python/SXS data/SXS:BBH:0023/"
Sim_path=""
Waveform_path="rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N4.dir"
md = sxs.metadata.Metadata.from_file(SA_path + Sim_path + "metadata.txt")    
h = scri.SpEC.read_from_h5(SA_path + Sim_path + Waveform_path)
M = md['remnant_mass']

v = [0.0001,0.0,0.0]
beta = np.linalg.norm(v)
g_m = (1/(1 - beta**2))**(1/2)


hp = h.transform(boost_velocity = v)

E_dotp = scri.energy_flux(hp)
P_dotp = scri.momentum_flux(hp) 
L_dotp = scri.angular_momentum_flux(hp)
B_dotp = scri.boost_flux(hp)
Q_dotp = np.c_[E_dotp, P_dotp, L_dotp, B_dotp]


Boost_matrix = [[g_m ,  -g_m * beta, 0,0], [ -g_m * beta, g_m, 0, 0], [0,0,1,0],[0,0,0,1]]

E_dot = scri.energy_flux(h.interpolate(hp.t))
P_dot = scri.momentum_flux(h.interpolate(hp.t)) 
L_dot = scri.angular_momentum_flux(h.interpolate(hp.t))
B_dot = scri.boost_flux(h.interpolate(hp.t))

Bondi_p_dot = np.c_[E_dot, P_dot]

Bondi_p_dot = np.transpose(np.matmul(Boost_matrix, np.transpose(Bondi_p_dot)))    

M = np.zeros((4,4,len(hp.t)))
for j in range(3):
    M[0,j+1,:] = B_dot[:,j]
    M[j+1,0,:] = M[0,j+1,:]
M[1,2,:] = - L_dot[:,2]
M[1,3,:] = L_dot[:,1]
M[2,3,:] = - L_dot[:,0]
M[2,1,:] = - M[1,2,:]
M[3,1,:] = - M[1,3,:]
M[3,2,:] = - M[2,3,:]
    

M = np.einsum('mi,ilk->mlk', Boost_matrix, np.einsum('ijk, jl -> ilk', M, Boost_matrix))   
    
Q_dot = np.c_[Bondi_p_dot, M[3,2,:], M[1,3,:], M[2,1,:], M[0,1,:],M[0,2,:],M[0,3,:]]                      
    
C = Q_dot - Q_dotp
D = Q_dotp + Q_dot

x = hp.t
y = np.log(np.abs(C/D))

fig, axs = plt.subplots(2, 2)
fig.suptitle('Boost')
axs[0,0].plot(x, (y[:,0]))
axs[0,0].set_title('Energy')
axs[0, 1].plot(x, y[:,1:4], 'tab:orange')
axs[0, 1].set_title('Momentum')
axs[1, 0].plot(x, y[:,4:7], 'tab:green')
axs[1, 0].set_title('Angular Momentum')
axs[1, 1].plot(x, y[:,7:], 'tab:red')
axs[1, 1].set_title('Boost')

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='log Fractional error')

plt.plot()
plt.show()
