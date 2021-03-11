from sklearn import datasets, svm, metrics, model_selection
import matplotlib.pyplot as plt
# from sdm_schrodinger1D import SDMSchrodinger1D, np, ne
from scipy.signal import fftconvolve
from scipy.signal import blackman
import numpy as np
import sys


def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result

def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    #test
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.fft(minus_one * A)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    return result


params = dict(
    # integer to factorize
    N=7 * 17,

    # parameters of Gauss sum
    M=5,
    L=30,

    # carrier frequency
    omega0=0.1,

    # transform limited pulse
    TL_laser_filed="F * sin(omega0 * t) * sin(pi * t / t_final) ** 2",

    # F=float(sys.argv[1]),
    # the final time of propagation (= 8 periods of laser oscillations)
    t_final=12 * 2. * np.pi / 0.1,

    # parameters of propagator
    V="-1. / sqrt(X ** 2 + 1.37)",
    diff_V="X * (X ** 2 + 1.37) ** (-1.5)",

    pi=np.pi,

    abs_boundary="sin(0.5 * pi * (X + X_amplitude) / X_amplitude) ** (0.05 * dt)",

    X_gridDIM=2 * 1024,
    X_amplitude=140.,

)

# update dt such that it matches the chosen number of timesteps
# parameternames='-%s-intensity-%s-steps' % (sys.argv[1], sys.argv[2])
# steps = 2**int(sys.argv[2])
steps=2
params['dt'] = params['t_final'] / steps
times=np.arange(steps)*params['dt']
print(params['t_final'])

###################################################################
#
#   Converting input data to field
#
####################################################################
#load test data. This only loads 1 and 0 data to make binary classification easier
digits=datasets.load_digits(return_X_y=0)
# digits=datasets.fetch_olivetti_faces()
print(type(digits.data))
print(np.array(digits).shape)
# randtest=0
# testn=np.size(digits.data[:,1])
testn=1
randtest=0
print(testn)
encode= np.transpose(digits.data[randtest:randtest+testn,:])
encode =encode/np.sum(encode,axis=0)
print(encode.shape)
#Truncate this encoding so the last entry is always non-zero. This is necessary to ensure that the scaling by omega0 is accurate.
# encode=np.trim_zeros(encode,trim='b')

#Here we'll try directly using the testdigit as a direct mapping in frequency space. We'll take the number of time steps and use that. The input electric field will therefore be
#
# print(consistencycheck.size)
# space=int(steps/testdigit[testindex,:].size)
# zeros=np.zeros(space*testdigit[testindex,:].size)
# zeros[::space]=consistencycheck
# encodedfield=zeros

#Space frequencies to ensure they are distinguishable.
# inomega = params['omega0']*np.arange(encode[:,0].size)/encode[:,0].size
# print(inomega.size)
# encodewave=np.array([[(params['F']/16)*np.real(np.exp(1j*w*t))*np.sin(np.pi*t/params['t_final'])**2 for w in inomega] for t in times])
# timefield=np.dot(encodewave,encode)
# np.save('inputfield',timefield)
# omega = (np.arange(steps) - steps/ 2) * 2*np.pi / (params['t_final']*params['omega0'])
#
# # plt.title("input field")
# # plt.plot(times,timefield)
# # plt.show()
#
#
# ###################################################################
# #
# #   Running field through HHG
# #
# ####################################################################
#
# #Hoping for the best, and that I've correctly identified the part of sdm_schrodinger that does what I want it to
# # initialise output field
# spectra=[]
# pulse=[]
# dxdt=[]
# pav=[]
# dpdt=[]
# fav=[]
#
#
#
# # frequency range in units of omega0
#
# # set initial condition to be the ground state
# for j in range(0,testn):
#     output= SDMSchrodinger1D(field=timefield[:,j],**params).set_groundstate()
#     #Propagate to generate the output field
#     output.propagate_field(steps)
#     x_av=output.X_average
#     spectra.append(np.abs(FT(x_av)*blackman(steps))**2)
#     pulse.append(x_av)
#     dxdt.append(np.gradient(x_av, params['dt']))
#     pav.append(output.X_average_RHS)
#     dpdt.append(np.gradient(output.P_average, params['dt']))
#     fav.append(output.P_average_RHS)
# # spectra = np.array([np.abs(FT(output[j].X_average * blackman(steps))) ** 2 for j in range(0,testn)])
# spectra=np.array(spectra)
# pulse=np.array(pulse)
# dxdt=np.array(dxdt)
# pav=np.array(pav)
# dpdt=np.array(dpdt)
# fav=np.array(fav)
# np.save('./output/spectra/spectra'+parameternames,spectra)
# np.save('./output/pulse/pulse'+parameternames,pulse)
# np.save('./output/Ehrenfest/dxdt'+parameternames,dxdt)
# np.save('./output/Ehrenfest/pav'+parameternames,pav)
# np.save('./output/Ehrenfest/dpdt'+parameternames,dpdt)
# np.save('./output/Ehrenfest/fav'+parameternames,fav)
#
#


# # PLOTTING EHRENFEST THEOREMS
# dxdt = np.array([np.gradient(output[j].X_average, params['dt']) for j in range(0,testn)])
# pav = np.array([output[j].X_average_RHS for j in range(0,testn)])
# dpdt = np.array([np.gradient(output[j].P_average, params['dt']) for j in range(0,testn)])
# fav = np.array([output[j].P_average_RHS for j in range(0,testn)])
# plt.subplot(131)
# plt.title("Verify the first Ehrenfest theorem")
# plt.plot(
# times,
# np.transpose(dxdt),
# '-r',
# label='$d\\langle\\hat{x}\\rangle / dt$'
# )
# plt.plot(times, np.transpose(pav), '--b', label='$\\langle\\hat{p}\\rangle$')
# plt.legend()
# plt.ylabel('momentum')
# plt.xlabel('time $t$ (a.u.)')
#
# plt.subplot(133)
# plt.title("Verify the second Ehrenfest theorem")
#
# plt.plot(
# times,
# np.transpose(dpdt),
#     '-r',
#     label='$d\\langle\\hat{p}\\rangle / dt$'
# )
# plt.plot(times, np.transpose(fav), '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
# plt.legend()
# plt.ylabel('force')
# plt.xlabel('time $t$ (a.u.)')
#
# plt.subplot(133)
# plt.title("The expectation value of the hamiltonian")
# plt.show()
#
#
# # Analyze how well the energy was preserved
# h = np.transpose(np.array([output[j].hamiltonian_average for j in range(0,testn)]))
# print(
#     "\nHamiltonian is preserved within the accuracy of %.2e percent" % (100. * (1. - h.min() / h.max()))
# )
#
# plt.plot(times, h)
# plt.ylabel('energy')
# plt.xlabel('time $t$ (a.u.)')
#
# plt.show()