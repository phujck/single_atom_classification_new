import numpy as np
import wavelets
import transform
import lattice_definition as ld
import matplotlib.pyplot as plt
import sys, os
import scipy.integrate
#from matplotlib import cm as cm

sys.stdout = open(os.devnull, 'w')
##########################-INPUTS-##########################

'''input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a)'''
lat = ld.hhg(field=32.9, nup=3, ndown=3, nx=6, ny=0, U=0.52*3., t=.52, F0=10., a=4., bc='pbc')
laser = ld.pulse(lat, envelope="sin", carrierx="sin", carriery="sin", cycles=5., ellipticity=None, CEP=0., sigma=None)
delta = 1.e-2
'''minimum intensity included in the spectrogram. set to False to include all intensities'''
min_spec = -7.
'''maximum harmonic to be included in the spectrogram'''
max_harm = 50
'''spacings to be calculated in the spectrograms. =1 means using every point, =10 every 10th point etc'''
take = 1
'''path where current data is located'''
file_path = '/users/k1623514/lustre/data/HHG/6x6/U_5_4/U_5.0/'
'''source=0 if data is from exact simulations; source=1 if from tvmc; source=2 takes current from Jfile'''
source = 2
#Jfile = 'tracked_edge_current_x.dat'
Jfile = 'bulk_current_x.dat'

#file_path = '/users/k1623514/code/exact_hhg/sim_data/'
'''plot_mott=True includes the 1D excitations and tthreshold includes the time when Fth is reached'''
plot_mott = False
tthreshold = False
'''pulse=0 doesn't include a pulse, =1 includes the vector potential, =2 includes the field strength'''
pulse = 2
'''title of plot. if not required then title=None'''
title = None
'''plot wavelet in the time and frequency domains'''
plot_wavelet = False
'''save spectrogram to file_path'''
save = False

'''type of wavelet. options are 'Morlet', 'Paul', 'DOG', 'Ricker' (aka 'Marr' or 'Mexican_hat')'''
wvlet = 'Morlet'
'''w0 for Morlet is the nondimensional frequency constant. default is 6'''
w0 = 8
'''the scale resolution'''
dj = 0.1
'''the order. For Paul the default is m=4 and for Dog it's m=2'''
m = 4
'''whether to unbias the power spectrum. default is False'''
unbias = False
'''disregard wavelet power outside the cone of influence when computing global wavelet spectrum. default is False'''
mask_coi = False
'''wavelet in time or frequency domain. they give the same result'''
frequency = True

############################################################
sys.stdout = sys.__stdout__

'''Ut = U/t'''
def correlation(x,Ut):
    return (4./Ut)*np.log(x+np.sqrt(x**2.-1.))/np.cosh(2.*np.pi*x/Ut)

def mott(x,Ut):
    if Ut==0.0:
        return 0.0
    else:
        return (16./Ut)*np.sqrt(x**2.-1.)/np.sinh(2.*np.pi*x/Ut)

'''F(t) = -dA(t)/dt, with time in units of cycles'''
def fstrength(time):
    field = -lat.F0*(np.cos(2.*np.pi*time)*(np.sin(np.pi*time/laser.cycles)**2.) + \
                 (1./laser.cycles)*np.cos(np.pi*time/laser.cycles)*np.sin(np.pi*time/laser.cycles)*np.sin(2.*np.pi*time))
    return field

def findfield(E,thresh):
    for k, f in enumerate(E):
        if thresh < abs(f):
            return k
    return 0.0

if source==0:
	if lat.U.is_integer():
		Un = 'current_' + str(int(lat.U)) + '.dat'
	else:
		Un = 'current_' + str(lat.U) + '.dat'
elif source==1:
	Un = 'current.dat'
elif source==2:
	Un = Jfile
    
eJ = []
with open(file_path + Un, 'r') as f:
    info = f.readlines()
for i in info:
    eJ.append(float(i.split()[0]))
eJ = np.array(eJ)
at = np.gradient(eJ, delta)[::take]
tc = np.linspace(0., laser.cycles, len(at))

if wvlet=='Morlet':
    wave = wavelets.Morlet(w0=w0)
elif wvlet=='Paul':
    wave = wavelets.Paul(m=m)
elif wvlet=='DOG':
    wave = wavelets.DOG(m=m)
elif wvlet=='Ricker' or wvlet=='Marr' or wvlet=='Mexican_hat':
    wave = wavelets.Ricker()
else:
    print('Invalid choice of wavelet')

analysis = transform.WaveletAnalysis(data=at, time=tc, dt=delta*take, dj=dj, wavelet=wave, 
                                     unbias=unbias, mask_coi=mask_coi, frequency=frequency, axis=-1)

#scales = analysis.scales
#wk = w0/(lat.field*scales)
wk = analysis.fourier_frequencies/lat.freq
wk = wk[wk<=max_harm]
L = len(at)
nrm = 2.**(np.ceil(np.log2(L)))
nrm = L**2./(nrm - L)
power = analysis.wavelet_power/nrm
power = np.log10(power[-len(wk):])
wmax, tmax = np.unravel_index(np.argmax(power[:len(wk),:], axis=None), power.shape)
maxp = power[wmax, tmax]
tmax *= take*delta*lat.freq
print('\nPeak emission occurs at time = ' + str(round(tmax, 1)) + ' cycles, at harmonic = ' + str(round(wk[wmax], 1)) + ', at intensity = ' + str(round(maxp, 1)) + '\n')

fig, ax = plt.subplots()
T, S = np.meshgrid(tc, wk)
if min_spec:
	power = np.ma.array(power, mask=power<min_spec)
if plot_mott:
	excit = scipy.integrate.quad(mott, 1., np.inf, args=(lat.U))[0]
	excit_max = (excit + 8.)/lat.field
	excit /= lat.field
	plt.axhline(excit, linestyle='--', color='k')
	plt.axhline(excit_max, linestyle='--', color='k')
if tthreshold:
	excit = scipy.integrate.quad(mott, 1., np.inf, args=(lat.U))[0]
	x = scipy.integrate.quad(correlation, 1., np.inf, args=(lat.U))[0]
	Fth = excit*x/2.
	tinc = np.arange(0., laser.cycles, 1.e-2)
	E = [fstrength(t) for t in tinc]
	tFth = 1.e-2*findfield(E, Fth/lat.a)
	plt.axvline(tFth, linestyle='--', color='k')
	#Fth2 = excit**2./8.
	#tFth2 = 1e-2*findfield(E, Fth2/lat.a)
	#plt.axvline(tFth2, linestyle='--')
if pulse>0:
	tc2 = np.linspace(0., laser.cycles/lat.freq, len(at))	
	cpulse = []
	for ctime in tc2:
		cpulse.append(laser.calc_phi(ctime)[0])
	cpulse = np.array(cpulse)
	if pulse==2:
		cpulse = -np.gradient(cpulse, delta)
	cpulse /= np.max(cpulse)
	bst = .9*max_harm
	ax.plot(tc, bst+cpulse, color='red')
cmat = ax.contourf(T, S, power, 100, cmap='RdYlBu_r')
fig.colorbar(cmat)
ax.set_ylabel('Harmonic')
ax.set_xlabel('Time [cycles]')
if title!=None:
	ax.set_title(title)
#ax.grid(True)
#if mask_coi:
#    coi_time, coi_scale = analysis.coi
#    ax.fill_between(x=coi_time, y1=coi_scale, y2=analysis.scales.max(), color='gray', alpha=0.3)
plt.show()
    
#t,w = np.meshgrid(tc,wk)
#powern = np.ma.masked_where(power<min_spec,power)
#cm.RdYlBu_r.set_bad(color='white', alpha=None)
#plt.pcolormesh(t,w,powern,cmap='RdYlBu_r')
#plt.colorbar()
#plt.xlabel('Time [cycles]')
#plt.ylabel('Harmonic Order')
#plt.title('Time-Resolved Emission')
#plt.show()

if plot_wavelet:
	tm = np.linspace(-5., 5., 1000)
	wavey = wave.time(tm)
	plt.plot(tm, wavey.real, label='Real')
	plt.plot(tm, wavey.imag, label='Imag')
	plt.xlabel('Time')
	plt.legend()
	plt.title(wvlet + ' wavelet in time domain')
	plt.show()

	wkm = analysis.w_k
	wavey = wave.frequency(wkm)
	plt.plot(wkm, wavey.real)
	plt.xlabel('Harmonic')
	plt.title(wvlet + ' wavelet in frequency domain')
	plt.show()

#fig, ax = plt.subplots()
#T, S = np.meshgrid(tc, Fp)
#if min_spec:
#	power = np.ma.array(power, mask=power<min_spec)
#cmat = ax.contourf(T, S, power, 100, cmap='RdYlBu_r')
#fig.colorbar(cmat)
#ax.set_ylabel('Fourier period')
#ax.set_xlabel('Time [cycles]')
#ax.grid(True)
#if mask_coi:
#    coi_time, coi_scale = analysis.coi
#    ax.fill_between(x=coi_time, y1=coi_scale, y2=analysis.scales.max(), color='gray', alpha=0.3)
#plt.show()

if save:
	plt.matplotlib.use('Agg')
	fig.savefig(file_path + 'spectrogram.png', bbox_inches='tight')
