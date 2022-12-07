import sys
import scri
import sxs
import numpy as np
import csv
import cmath
import quaternion
from quaternion import rotate_vectors
from matplotlib import pyplot as plt
from scri import flux
import functools
from scri import waveform_base
from spherical_functions import clebsch_gordan as CG
from scri import WaveformModes


SA_path="/home/khushal/Documents/Python/SXS data/SXS:BBH:0023/"
Waveform_path="rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N4.dir"
md = sxs.metadata.Metadata.from_file(SA_path + "metadata.txt")    
h = scri.SpEC.read_from_h5(SA_path +  Waveform_path)
M = md['remnant_mass']

s = [0.0,0.0,0.5]

hp = h.transform(space_translation = s)
E_dotp = scri.energy_flux(hp)
P_dotp = scri.momentum_flux(hp) 
L_dotp = scri.angular_momentum_flux(hp)
B_dotp = scri.boost_flux(hp)

def boost_s_dep(h,c, hdot=None):    
    from scri import h as htype
    from scri import hdot as hdottype

    if not isinstance(h, WaveformModes):
        raise ValueError(
            "Boost fluxes can only be calculated from a `WaveformModes` object; "
            + "`h` is of type `{0}`.".format(type(h))
        )
    if (hdot is not None) and (not isinstance(hdot, WaveformModes)):
        raise ValueError(
            "Boost fluxes can only be calculated from a `WaveformModes` object; "
            + "`hdot` is of type `{0}`.".format(type(hdot))
        )
    if h.dataType == htype:
        if hdot is None:
            hdot = h.copy()
            hdot.dataType = hdottype
            hdot.data = h.data_dot
        elif hdot.dataType != hdottype:
            raise ValueError(
                "Input argument `hdot` is expected to have data of type `hdot`; "
                + "this `hdot` waveform data has type `{0}`".format(h.data_type_string)
            )
    else:
        raise ValueError(
            "Input argument `h` is expected to have data of type `h`; "
            + "this `h` waveform data has type `{0}`".format(h.data_type_string)
        )

    boost_s_dep = np.zeros((hdot.n_times, 3), dtype=float)
    
    _, hdot_chiz_hdot = flux.matrix_expectation_value(hdot, functools.partial(flux.p_z, s=-2), hdot)
    _, hdot_chiplus_hdot = flux.matrix_expectation_value(hdot, functools.partial(flux.p_plus, s=-2), hdot)
    _, hdot_chiminus_hdot = flux.matrix_expectation_value(hdot, functools.partial(flux.p_minus, s=-2), hdot)

    boost_flux_plus = - (1 / 2) * np.multiply(c, hdot_chiplus_hdot) 
    boost_flux_minus = - (1 / 2) * np.multiply(c, hdot_chiminus_hdot)
    
    # This is the component in the x direction. x = 0.5 * ( plus + minus).real
    boost_s_dep[:, 0] = (0.5) * (boost_flux_plus + boost_flux_minus).real

    # This is the component in the y direction. y = 0.5 * ( plus - minus).imag
    boost_s_dep[:, 1] = (0.5) * (boost_flux_plus - boost_flux_minus).imag

    # Component in the z direction. Only the real part of the complex value is taken into account.
    boost_s_dep[:, 2] = - (1 / 2) * (np.multiply(c, hdot_chiz_hdot) ).real

    # A final factor of \frac{-1}{32 \pi} should be included that is present outside the integral expression of flux.

    boost_s_dep /= (-32) * np.pi

    return boost_s_dep


c = np.linalg.norm(s)
E_dot = scri.energy_flux(h.interpolate(hp.t))
P_dot = scri.momentum_flux(h.interpolate(hp.t)) 
L_dot = scri.angular_momentum_flux(h.interpolate(hp.t))
B_dot = scri.boost_flux(h.interpolate(hp.t)) + boost_s_dep(hp,- c)

Q_dot = np.c_[E_dot, P_dot, L_dot, B_dot]
Q_dotp = np.c_[E_dotp, P_dotp, L_dotp, B_dotp]

C = Q_dot - Q_dotp
D = Q_dotp + Q_dot

x = hp.t
y = np.log(np.abs(C/D))

fig, axs = plt.subplots(2, 2)
fig.suptitle('Space Translation')
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
