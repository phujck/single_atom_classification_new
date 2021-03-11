from sklearn import datasets, svm, metrics, model_selection, linear_model
import matplotlib.pyplot as plt
from sdm_schrodinger1D import SDMSchrodinger1D, np, ne
from scipy.signal import fftconvolve
from scipy.signal import blackman
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,f1_score
from sklearn.svm import SVC
# IGNORING WARNINGS. DON'T LEAVE THIS ON!
# import warnings
# warnings.filterwarnings("ignore")


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

def analysis(data,tuned_parameters,test_fraction=0.8):
    # Use a support vector machine estimator with a grid search for hyperparameters
    # split data into train and test sets
    train_data, test_data, train_target, test_target = train_test_split(data, digits.target[:testn], test_size=test_fraction)


    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        classifier = GridSearchCV(SVC(), tuned_parameters, cv=5,
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
        expected, predicted =test_target, classifier.predict(test_data)
        print(classification_report(expected, predicted))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print()

def fscore(data,tuned_parameters,test_fraction=0.5):
    # Use a support vector machine estimator with a grid search for hyperparameters
    # split data into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(data, digits.target[:testn], test_size=test_fraction)
    classifier = GridSearchCV(SVC(), tuned_parameters, cv=2)
    classifier.fit(train_data, train_target)
    expected, predicted =test_target, classifier.predict(test_data)
    return(f1_score(predicted,expected,average='macro'))


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

    F=0.06,
    # the final time of propagation (= 7 periods of laser oscillations)
    t_final=8 * 2. * np.pi / 0.06,

    # parameters of propagator
    V="-1. / sqrt(X ** 2 + 1.37)",
    diff_V="X * (X ** 2 + 1.37) ** (-1.5)",

    pi=np.pi,

    abs_boundary="sin(0.5 * pi * (X + X_amplitude) / X_amplitude) ** (0.05 * dt)",

    X_gridDIM=2 * 1024,
    X_amplitude=140.,

)

# update dt such that it matches the chosen number of timesteps
steps = 2**12
params['dt'] = params['t_final'] / steps
times=np.arange(steps)*params['dt']
print(params['t_final'])

###################################################################
#
#   Converting input data to field
#
####################################################################
#load test data. This only loads 1 and 0 data to make binary classification easier
digits=datasets.load_digits(n_class=10,return_X_y=0)
testn=digits.data.size
# testn=300
spectra=np.load('fullspectra.npy')
spectralow=np.load('fullspectra.npy')
print(spectralow.shape)
print('highressize %s, lowressize %s' %spectra.shape, spectralow.shape)
spectra=spectra[:testn,:]
pulse=np.load('rawpulse.npy')
inputpulse=np.transpose(np.load('inputfield.npy'))
print(pulse.shape)
print(digits.data.size)
print(spectra.shape)

#Truncate this encoding so the last entry is always non-zero. This is necessary to ensure that the scaling by omega0 is accurate.
# encode=np.trim_zeros(encode,trim='b')

#Here we'll try directly using the testdigit as a direct mapping in frequency space. We'll take the number of time steps and use that. The input electric field will therefore be
#
# print(consistencycheck.size)
# space=int(steps/testdigit[testindex,:].size)
# zeros=np.zeros(space*testdigit[testindex,:].size)
# zeros[::space]=consistencycheck
# encodedfield=zeros


omega = (np.arange(steps) - steps/ 2) * 2*np.pi / (params['t_final']*params['omega0'])
# plt.semilogy(
#     omega,
#     np.transpose(spectra),
#     label="smooth $\\langle X \\rangle$"
# )
# plt.xlim([0,30])
# plt.title("output field")
# plt.show()
cutoff=4.5
linomega=np.array([a for a in omega if 0<a<1.01])
print(linomega)
cutoffomega=np.array([a for a in omega if 1<a<cutoff])
print(spectra.shape)
print(omega.size)
buffer=4
lincut= int((params['t_final']*params['omega0'])/(2*np.pi) )+buffer
cutspectra=spectra[:,int(spectra[0,:].size/2)+lincut:int(spectra[0,:].size/2)+lincut+cutoffomega.size]
linspectra=spectra[:,int(spectra[0,:].size/2):int(spectra[0,:].size/2)+linomega.size]
# fullspectra=spectra[:,int(spectra[0,:].size/2):int(spectra[0,:].size/2)+cutoffomega.size]
# print(cutspectra.size)
# print(linspectra.size)
# cutspectra=spectra[:,int(spectra.size/2):int(spectra.size/2)+lincut]
cutspectra[cutspectra < 4e-8] = 0.
# plt.semilogy(
#     cutoffomega,
#     np.transpose(cutspectra),
#     # np.transpose(fullspectra),
#     label="smooth $\\langle X \\rangle$"
# )
# plt.xlim([0,7])
# plt.title("output field")
# plt.show()
# plt.semilogy(
#     linomega,
#     np.transpose(linspectra),
#     label="smooth $\\langle X \\rangle$"
# )
# plt.xlim([0,7])
# plt.title("output field")
# plt.show()
# print(cutspectra.shape)





raw_parameters_rbf = [{'kernel': ['rbf'], 'gamma': [1e-4],
                         'C': [1000]}]
input_parameters_rbf = [{'kernel': ['rbf'], 'gamma': [1e3],
                    'C': [1e3]}]
high_parameters = [{'kernel': ['rbf'], 'gamma': [1e6, 3e6, 5e6],
                    'C': [6000, 6200, 5800]}]

raw_parameters = [{'kernel': ['linear'], 'C': [1e-4]}]
lin_parameters = [{'kernel': ['linear'], 'C': [5e-1]}]
high_parameters =[{'kernel': ['linear'], 'C': [1e10]}]
pulse_parameters=[{'kernel': ['linear'], 'C': [5e3]}]
full_parameters=[{'kernel': ['linear'], 'C': [5e5]}]
input_parameters=[{'kernel': ['linear'], 'C': [5e6]}]
analysis(digits.data[:testn],raw_parameters_rbf,0.95)
# analysis(linspectra, lin_parameters,0.5)
analysis(cutspectra, high_parameters,0.9)
# analysis(pulse,pulse_parameters,0.9)
# analysis(fullspectra,full_parameters,0.4)
# analysis(inputpulse,input_parameters_rbf)

fraction=np.arange(0.1,0.99,0.1)
fhigh= [fscore(cutspectra,high_parameters,k) for k in fraction]
# flin= [fscore(linspectra,lin_parameters,k) for k in fraction]
fraw=[fscore(digits.data[:testn],raw_parameters,k) for k in fraction]
frawrbf=[fscore(digits.data[:testn],raw_parameters_rbf,k) for k in fraction]
# fpulse=[fscore(pulse,pulse_parameters,k) for k in fraction]
finpulse=[fscore(inputpulse,input_parameters,k) for k in fraction]
finpulserbf=[fscore(inputpulse,input_parameters_rbf,k) for k in fraction]

# # print(fscore(cutspectra,high_parameters,0.7))
fraction=1-fraction

plt.plot(fraction,fhigh, label='Optical Output (High Harmonics)')
# plt.plot(fraction,fpulse, label='Optical Output (Time Domain)')
# plt.plot(fraction,flin, label='Linear')
plt.plot(fraction,fraw, label='Input Data (Linear Kernel)')
plt.plot(fraction,frawrbf, label='Input Data (Gaussian Kernel)')
plt.xlabel('Data Fraction Used for Training')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

plt.plot(fraction,fhigh, label='Optical Output (High Harmonics)')
# plt.plot(fraction,fpulse, label='Optical Output (Time Domain)')
# plt.plot(fraction,flin, label='Linear')
plt.plot(fraction,fraw, label='Input Data (Linear Kernel)')
plt.plot(fraction,frawrbf, label='Input Data (Gaussian Kernel)')
plt.plot(fraction,finpulse, label='Input Pulse (Linear Kernel)')
plt.plot(fraction,finpulserbf, label='Input Pulse (Gaussian Kernel)')
plt.xlabel('Data Fraction Used for Training')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
# plt.plot(times,np.transpose(pulse)[:,0:4])
# plt.show()


print('linear omegas %s' % cutoffomega.size)
# lincutoffs=np.arange(2,linomega.size, 1)
# flincutoffs=[fscore(spectra[:,int(spectra[0,:].size/2):int(spectra[0,:].size/2)+k],lin_parameters,0.5)for k in lincutoffs]
# plt.plot(((cutoff)*highcutoffs/cutoffomega.size),fcutoffs)
# plt.xlabel('$ \\frac{\omega_l}{\omega_0}$')
# plt.ylabel('F1 Score')
# plt.show()

#
# highcutoffs=np.arange(0,cutoffomega.size-1)
# fcutoffs=[fscore(spectra[:,int(spectra[0,:].size/2)+lincut+ k:int(spectra[0,:].size/2)+lincut+ cutoffomega.size],high_parameters,0.5)for k in highcutoffs]
# plt.plot((1+(cutoff-1)*highcutoffs/cutoffomega.size),fcutoffs)
# plt.xlabel('$ \\frac{\omega_l}{\omega_0}$')
# plt.ylabel('F1 Score')
# plt.show()


#
# pulsecutoffs=np.arange(steps-1)
# pulsecutoffs=[fscore(pulse[:,0:1+k],pulse_parameters,0.5)for k in pulsecutoffs]
# plt.plot((times/params['t_final']),pulsecutoffs)
# plt.xlabel('$ \\frac{t_c}{t_f}$')
# plt.ylabel('F1 Score')
# plt.show()










