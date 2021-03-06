from sklearn import datasets, svm, metrics, model_selection, linear_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import blackman
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from numba import njit

params = {
    'axes.labelsize': 40,
    # 'legend.fontsize': 28,
    'legend.fontsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    # 'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True,
    # 'figure.figsize': (12, 16),
    'figure.figsize': (20, 12),
    'lines.linewidth': 3,
    'lines.markersize': 15

}

plt.rcParams.update(params)

# IGNORING WARNINGS. DON'T LEAVE THIS ON!
# import warnings
# warnings.filterwarnings("ignore")


# def iFT(A):
#     """
#     Inverse Fourier transform
#     :param A:  1D numpy.array
#     :return:
#     """
#     A = np.array(A)
#     minus_one = (-1) ** np.arange(A.size)
#     result = np.fft.ifft(minus_one * A)
#     result *= minus_one
#     result *= np.exp(1j * np.pi * A.size / 2)
#     return result
#
# def FT(A):
#     """
#     Fourier transform
#     :param A:  1D numpy.array
#     :return:
#     """
#     #test
#     A = np.array(A)
#     minus_one = (-1) ** np.arange(A.size)
#     result = np.fft.fft(minus_one * A)
#     result *= minus_one
#     result *= np.exp(-1j * np.pi * A.size / 2)
#     return result

"""set up system parameters"""
F = 10**(-2)  # laser intensity
x_range = 1  # range over which x coords can be drawn
y_range = 1  # range over which y coords can be drawn
radius = 0.8  # radius of the classification circle.
data_n = 1000  # number of data points in set
omega_0=0.06
omega_1=2*omega_0
omega_2=3*omega_0
# outfile = './Data/fieldstrength={}_n={},x_range={},y_range={},radius={}.npz'.format(F, data_n, x_range, y_range, radius)
outfile = './Data/fieldstrength={}_n={},x_range={},y_range={},radius={},x_omega={},y_omega={}.npz'.format(F,data_n,x_range,y_range,radius,omega_1,omega_2)
full_data = np.load(outfile, allow_pickle=True)
spectra = full_data['spectra']
harmonics = full_data['harmonics']
coords = full_data['data']
targets = full_data['targets']
data_F = full_data['field_strength']

assert F == data_F

# print(coords)
# print(targets)
# print('sall cool')
omega_laser = 0.06
dt = 0.05

t_final = 12 * 2. * np.pi / omega_laser
times = np.linspace(0, t_final, dt)


@njit
def laser(t, coords, F):
    """
    The strength of the laser field.
    Always add an envelop to the laser field to avoid all sorts of artifacts.
    We use a sin**2 envelope, which resembles the Blackman filter
    """
    x_dat, y_dat = coords
    return F * (np.sin(omega_laser * t) + x_dat * np.sin(2 * omega_laser * t) + y_dat * np.sin(
        4 * omega_laser * t)) * np.sin(np.pi * t / t_final) ** 2


inputpulse = [[laser(t, co, F) for t in times] for co in coords]



def analysis(X, Y, tuned_parameters, test_fraction=0.8):
    # Use a support vector machine estimator with a grid search for hyperparameters
    # split data into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(X, Y, test_size=test_fraction)

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        classifier = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=6,
                                  scoring='%s_macro' % score)
        classifier.fit(train_data, train_target)

        print("Best parameters set found on development set:")
        print()
        print(classifier.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        expected, predicted = test_target, classifier.predict(test_data)
        print(classification_report(expected, predicted))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print()


def fscore(X, Y, tuned_parameters, train_fraction=0.5):
    # Use a support vector machine estimator with a grid search for hyperparameters
    # split data into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(X, Y, test_size=1 - train_fraction)
    classifier = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=4)
    classifier.fit(train_data, train_target)
    expected, predicted = test_target, classifier.predict(test_data)
    return (f1_score(predicted, expected, average='macro'))

def fscorelinparams(X, Y, tuned_parameters, train_fraction=0.5):
    # Use a support vector machine estimator with a grid search for hyperparameters
    # split data into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(X, Y, test_size=1 - train_fraction)
    classifier = SVC(tuned_parameters, cv=8, n_jobs=4)
    classifier.fit(train_data, train_target)
    print(classifier.coef_)
    expected, predicted = test_target, classifier.predict(test_data)
    return (f1_score(predicted, expected, average='macro'))


raw_parameters_rbf = [{'kernel': ['rbf'], 'gamma': [1],
                       'C': [1e3]}]
input_parameters_rbf = [{'kernel': ['rbf'], 'gamma': [1e3],
                         'C': [1e3]}]
# high_parameters = [{'kernel': ['rbf'], 'gamma': [1e6, 3e6, 5e6],
#                     'C': [6000, 6200, 5800]}]

raw_parameters = [{'kernel': ['linear'], 'C': [1e3]}]
lin_parameters = [{'kernel': ['linear'], 'C': [1e3]}]
high_parameters = [{'kernel': ['linear'], 'C': [1e3]}]
pulse_parameters = [{'kernel': ['linear'], 'C': [5e3]}]
full_parameters = [{'kernel': ['linear'], 'C': [1e3]}]
input_parameters = [{'kernel': ['linear'], 'C': [5e6]}]


# fraction=np.arange(0.005,0.3,0.005)
# fhigh= [fscore(cutspectra,targets,high_parameters,k) for k in fraction]
# print('High Harmonic analysis done')
# flin= [fscore(linspectra,targets,lin_parameters,k) for k in fraction]
# print('Linear Harmonic analysis done')
# fraw=[fscore(coords,targets,raw_parameters,k) for k in fraction]
# print('input linear kernel done')
# frawrbf=[fscore(coords,targets,raw_parameters_rbf,k) for k in fraction]
# print('input Gaussian Kernel Done')
# ffull=[fscore(newspectra,targets,pulse_parameters,k) for k in fraction]
# # finpulse=[fscore(inputpulse,targets,input_parameters,k) for k in fraction]
# # finpulserbf=[fscore(inputpulse,input_parameters_rbf,k) for k in fraction]
# #
# # # # print(fscore(cutspectra,high_parameters,0.7))
# # fraction=1-fraction
# #
# plt.plot(fraction,ffull, label='Optical Output (Full Spectrum)')
# plt.plot(fraction,fhigh, label='Optical Output ($\\omega>\\omega_c$)')
# # plt.plot(fraction,finpulse, label='Input Pulse (Time Domain)')
# plt.plot(fraction,flin, label='Optical Output ($\\omega<\\omega_c$)')
# plt.plot(fraction,fraw, label='Input Data (Linear Kernel)')
# plt.plot(fraction,frawrbf, label='Input Data (Gaussian Kernel)')
# plt.xlabel('Training Data Fraction')
# plt.ylabel('F1 Score')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('./Plots/circlefscorewithfraction.pdf',bbox_inches='tight')
# plt.show()
## analysis(coords,targets,raw_parameters,0.5)
# analysis(coords,targets,raw_parameters_rbf,0.5)
# analysis(linspectra,targets, lin_parameters,0.5)
# analysis(cutspectra,targets, high_parameters,0.5)
# analysis(spectra,targets,full_parameters,0.5)
# analysis(inputpulse,targets,input_parameters_rbf)
#
#

"""setup lin and non-line spectra"""
spectral_cutoff = 10
omega_c=4.01
linharmonics = np.array([a for a in harmonics if 0 < a < omega_c])
cutoffomega = np.array([a for a in harmonics if 1 < a < spectral_cutoff])
lindex = np.argmax(harmonics > omega_c)
cutdex = np.argmax(harmonics > spectral_cutoff)
zeroindex = int(spectra[0].size / 2)
linspectra = []
cutspectra = []
newspectra = []
for spectrum in spectra:
    newspectra.append(spectrum[zeroindex:])
    linspectra.append(spectrum[zeroindex:lindex])
    cutspectra.append(spectrum[lindex:cutdex])
bottom_f = -1.9
top_f = -0.1
number_f = 30
fraction = 10**np.linspace(bottom_f, top_f, number_f)
# fhigh = [fscore(cutspectra, targets, high_parameters, k) for k in fraction]
print('High Harmonic analysis done')
# flin = [fscore(linspectra, targets, lin_parameters, k) for k in fraction]
print('Linear Harmonic analysis done')
fraw=[fscore(coords,targets,raw_parameters,k) for k in fraction]
# print('input linear kernel done')
frawrbf = [fscore(coords, targets, raw_parameters_rbf, k) for k in fraction]
print('input Gaussian Kernel Done')
ffull = [fscore(newspectra, targets, pulse_parameters, k) for k in fraction]
# finpulse=[fscore(inputpulse,targets,input_parameters,k) for k in fraction]
# finpulserbf=[fscore(inputpulse,input_parameters_rbf,k) for k in fraction]
#
# # # print(fscore(cutspectra,high_parameters,0.7))
# fraction=1-fraction
# outfile = (
#     './Data/fractionscores:bottomf={}_topf={}_numberf={}_omegac={},fieldstrength={}_n={},x_range={},y_range={},radius={}.npz'.format(
#         bottom_f, top_f, number_f, omega_c,F, data_n, x_range, y_range, radius))
#
# fract_dict=dict(fraction=fraction,ffull=ffull,fhigh=fhigh,flin=flin,fraw=fraw,frawrbf=frawrbf)
# np.savez(outfile, **fract_dict)
# plt.plot(fraction, ffull, label='Optical Output (Full Spectrum)')
# plt.plot(fraction, fhigh, label='Optical Output ($\\omega>\\omega_c$)')
# # plt.plot(fraction,finpulse, label='Input Pulse (Time Domain)')
# plt.plot(fraction, flin, label='Optical Output ($\\omega<\\omega_c$)')
# plt.plot(fraction,fraw, label='Input Data (Linear Kernel)')
# plt.plot(fraction, frawrbf, label='Input Data (Gaussian Kernel)')
plt.semilogx(fraction, ffull, label='Single Atom Computer',marker='^')
# plt.semilogx(fraction, fhigh, label='Optical Output ($\\omega>\\omega_c$)')
# plt.semilogx(fraction, flin, label='Optical Output ($\\omega<\\omega_c$)')
plt.semilogx(fraction,fraw, label='Linear Kernel',marker='o')
plt.semilogx(fraction, frawrbf, label='Gaussian Kernel',marker='v')
plt.xlabel('Training Data Fraction')
plt.ylabel('F1 Score (Test Data)')
plt.xlim(0.99*10**-2,1.001)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./Plots/circlefscorewithlowfraction.pdf', bbox_inches='tight')
plt.show()

intensity_linspec = []
intensity_highspec = []
intensity_fullspec = []
# intensities=np.linspace(0.001,0.02,20)
# intensities = np.linspace(0.0004, 0.0008, 3)
# intensities = np.append(intensities, np.linspace(0.001, 0.02, 20))

bottom_I = -4
top_I = -2
number_I = 30
fraction_I = 0.2
# intensities = 10 ** np.linspace(bottom_I, top_I, number_I)
# intensities = 10 ** np.linspace(-5, -2, 30)
# intensities=10**np.linspace(-4,-2,10)
# intensities = 10 ** np.linspace(-4, -2, 10)
intensities=10**np.linspace(-4,-1.5,15)


# for F in intensities:
#     outfile = './Data/fieldstrength={}_n={},x_range={},y_range={},radius={}.npz'.format(F, data_n, x_range, y_range,
#                                                                                         radius)
#     full_data = np.load(outfile, allow_pickle=True)
#     spectra = full_data['spectra']
#     harmonics = full_data['harmonics']
#     coords = full_data['data']
#     targets = full_data['targets']
#     data_F = full_data['field_strength']
#
#     # assert F==data_F
#
#     print(coords)
#     print(targets)
#     print('sall cool')
#     omega_laser = 0.06
#     dt = 0.05
#
#     t_final = 12 * 2. * np.pi / omega_laser
#     times = np.linspace(0, t_final, dt)
#
#     """setup lin and non-line spectra"""
#     spectral_cutoff = 10
#     omega_c=4.01
#     linharmonics = np.array([a for a in harmonics if 0 < a < omega_c])
#     cutoffomega = np.array([a for a in harmonics if 1 < a < spectral_cutoff])
#     lindex = np.argmax(harmonics > omega_c)
#     cutdex = np.argmax(harmonics > spectral_cutoff)
#     zeroindex = int(spectra[0].size / 2)
#     linspectra = []
#     cutspectra = []
#     newspectra = []
#     for spectrum in spectra:
#         newspectra.append(spectrum[zeroindex:])
#         linspectra.append(spectrum[zeroindex:lindex])
#         cutspectra.append(spectrum[lindex:cutdex])
#     intensity_fullspec.append(fscore(newspectra, targets, high_parameters, fraction_I))
#     intensity_highspec.append(fscore(cutspectra, targets, high_parameters, fraction_I))
#     intensity_linspec.append(fscore(linspectra, targets, high_parameters, fraction_I))
#
#
#
# outfile = './Data/intensityscores_bottomI={}_topI={}_numberI={}_fractionI={}_n={},omegac={}_x_range={},y_range={},radius={}.npz'.format(
#         bottom_I, top_I, number_I, fraction_I,omega_c,data_n, x_range, y_range, radius)
# intensity_dict=dict(intensities=intensities,ffull=intensity_fullspec,fhigh=intensity_highspec,flin=intensity_linspec)
# np.savez(outfile, **fract_dict)
# # plt.plot(intensities, intensity_fullspec, label='Optical Output (Full Spectrum)')
# # plt.plot(intensities, intensity_highspec, label='Optical Output ($\\omega>\\omega_c$)')
# # # plt.plot(fraction,finpulse, label='Input Pulse (Time Domain)')
# # plt.plot(intensities, intensity_linspec, label='Optical Output ($\\omega<\\omega_c$)')
#
# plt.semilogx(intensities, intensity_fullspec, label='Optical Output (Full Spectrum)')
# plt.semilogx(intensities, intensity_highspec, label='Optical Output ($\\omega>\\omega_c$)')
# plt.semilogx(intensities, intensity_linspec, label='Optical Output ($\\omega<\\omega_c$)')
# plt.xlabel('Intensity')
# plt.ylabel('F1 Score')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('./Plots/circlefscorewithintensity.pdf', bbox_inches='tight')
# plt.show()


"""plotting F score with intensity"""
# for fraction_I in [0.1,0.5,0.8]:
for omegas in [3,4,5]:
    omega_2=omega_0*omegas
    fraction_I=0.9
    intensity_fullspec = []
    data_n=200
    for F in intensities:
        outfile = './Data/fieldstrength={}_n={},x_range={},y_range={},radius={},x_omega={},y_omega={}.npz'.format(F,data_n,x_range,y_range,radius,omega_1,omega_2)

        full_data = np.load(outfile, allow_pickle=True)
        spectra = full_data['spectra']
        harmonics = full_data['harmonics']
        coords = full_data['data']
        targets = full_data['targets']
        data_F = full_data['field_strength']

        # assert F==data_F

        print(coords)
        print(targets)
        print('sall cool')
        omega_laser = 0.06
        dt = 0.05

        t_final = 12 * 2. * np.pi / omega_laser
        times = np.linspace(0, t_final, dt)

        """setup lin and non-line spectra"""
        spectral_cutoff = 30
        omega_c=omegas+0.5
        linharmonics = np.array([a for a in harmonics if 0 < a < omega_c])
        cutoffomega = np.array([a for a in harmonics if omega_c < a < spectral_cutoff])
        lindex = np.argmax(harmonics > omega_c)
        print(lindex)
        cutdex = np.argmax(harmonics > spectral_cutoff)
        print(cutdex)

        zeroindex = int(spectra[0].size / 2)
        linspectra = []
        cutspectra = []
        newspectra = []
        for spectrum in spectra:
            # print(len(spectrum[zeroindex:]))
            newspectra.append(spectrum[zeroindex:])
            linspectra.append(spectrum[zeroindex:lindex])
            cutspectra.append(spectrum[lindex:cutdex])
        intensity_fullspec.append(fscore(cutspectra, targets, high_parameters, fraction_I))
        # intensity_highspec.append(fscore(cutspectra, targets, high_parameters, fraction_I))
        # intensity_linspec.append(fscore(linspectra, targets, high_parameters, fraction_I))



    # outfile = './Data/intensityscores_bottomI={}_topI={}_numberI={}_fractionI={}_n={},omegac={}_x_range={},y_range={},radius={}.npz'.format(
    #         bottom_I, top_I, number_I, fraction_I,omega_c,data_n, x_range, y_range, radius)
    # intensity_dict=dict(intensities=intensities,ffull=intensity_fullspec,fhigh=intensity_highspec,flin=intensity_linspec)
    # np.savez(outfile, **fract_dict)
    # plt.plot(intensities, intensity_fullspec, label='Optical Output (Full Spectrum)')
    # plt.plot(intensities, intensity_highspec, label='Optical Output ($\\omega>\\omega_c$)')
    # # plt.plot(fraction,finpulse, label='Input Pulse (Time Domain)')
    # plt.plot(intensities, intensity_linspec, label='Optical Output ($\\omega<\\omega_c$)')

    plt.semilogx(intensities, intensity_fullspec, label='$\\omega_y={:d} \\omega_0$'.format(omegas), marker='o')
plt.xlabel('Field Amplitude (a.u.)')
plt.ylabel('F1 Score (Test Set)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./Plots/circlefscoreintensityvaryingfractions.pdf', bbox_inches='tight')
plt.show()
