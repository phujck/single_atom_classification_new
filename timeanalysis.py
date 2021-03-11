from sklearn import datasets, svm, metrics, model_selection, linear_model
from sdm_schrodinger1D import SDMSchrodinger1D, np, ne
from scipy.signal import fftconvolve
from scipy.signal import blackman
import matplotlib.pyplot as plt
import sys
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

def analysis(data,tuned_parameters,test_fraction=0.4):
    # Use a support vector machine estimator with a grid search for hyperparameters
    train_data, test_data, train_target, test_target = train_test_split(data, digits.target[:testn], test_size=test_fraction)



    classifier = GridSearchCV(SVC(), tuned_parameters, cv=5)
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
    # expected, predicted =test_target, classifier.predict(test_data)
    # print(classification_report(expected, predicted))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    print()

def fscore(data,tuned_parameters,test_fraction=0.3):
    # Use a support vector machine estimator with a grid search for hyperparameters
    train_data, test_data, train_target, test_target = train_test_split(data, digits.target[:testn], test_size=test_fraction)



    classifier = GridSearchCV(SVC(), tuned_parameters, cv=10)
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

    # the final time of propagation (= 7 periods of laser oscillations)
    t_final=12 * 2. * np.pi / 0.1,

    # parameters of propagator
    V="-1. / sqrt(X ** 2 + 1.37)",
    diff_V="X * (X ** 2 + 1.37) ** (-1.5)",

    pi=np.pi,

    abs_boundary="sin(0.5 * pi * (X + X_amplitude) / X_amplitude) ** (0.05 * dt)",

    X_gridDIM=2 * 1024,
    X_amplitude=140.,

)
steps = 2**int(12)
cmap = plt.get_cmap('jet_r')
params['dt'] = params['t_final'] / steps
times=np.arange(steps)*params['dt']
digits=datasets.load_digits()
testn=np.size(digits.data[:,1])

encode = np.transpose(digits.data[:testn, :])
encode = encode / np.sum(encode, axis=0)
# testn=1

omega = (np.arange(steps) - steps/ 2) * 2*np.pi / (params['t_final']*params['omega0'])
# plt.semilogy(
#     omega,
#     np.transpose(spectra),
#     label="smooth $\\langle X \\rangle$"
# )
# plt.xlim([0,30])
# plt.title("output field")
# plt.show()
# print(omega.size)
# cutspectra=spectra[:,int(spectra[0,:].size/2)+lincut:int(spectra[0,:].size/2)+lincut+cutoffomega.size]
# linspectra=spectra[:,int(spectra[0,:].size/2):int(spectra[0,:].size/2)+linomega.size]
raw_parameters = [{'kernel': ['linear'], 'C': [1e-4,]}]
lin_parameters = [{'kernel': ['linear'], 'C': [1e-4]}]
high_parameters =[{'kernel': ['linear'], 'C': [1e10]}]
pulse_parameters=[{'kernel': ['linear'], 'C': [1e3]}]
full_parameters=[{'kernel': ['linear'], 'C': [5e5]}]
input_parameters=[{'kernel': ['linear'], 'C': [5e6]}]
raw_parameters_rbf = [{'kernel': ['rbf'], 'gamma': [1e-4],
                         'C': [1000]}]
fscorepulse=[]
fscorehighspectra=[]
fscorelinspectra=[]
fscoreinput=[]

fscorepulsehigh=[]
fscorehighspectrahigh=[]
fscorelinspectrahigh=[]
fscoreinputhigh=[]
intensities=[]
c=0
for j in range(2,12):
    color = cmap((float(j)-7)/20)
    # if j <10:
    #     j=j*0.001
    # else:
    #     j=(j-9)*0.01
    j= 5 +j*5
    print('Intensity:')
    j = int(j)
    # if j ==0.0072:
    #     j=0.0074
    # if j==0.0078:
    #     j=0.008
    params['t_final']=j*2. * np.pi / 0.1
    print(j)
    parameternames='-%s-time-%s-steps.npy' % (0.004,j)
    parameternameshigh='intensity-%s-time-%s-steps-%s.npy' % (0.004,j,12)

    params['dt'] = params['t_final'] / steps
    omega = (np.arange(steps) - steps / 2) * 2 * np.pi / (params['t_final'] * params['omega0'])
    omegahigh=(np.arange(steps*2) - steps) * 2 * np.pi / (params['t_final'] * params['omega0'])
    cutoff = 6
    linomega = np.array([a for a in omega if 0 < a < 1.01])
    cutoffomega = np.array([a for a in omega if 1 < a < cutoff])
    linomegahigh = np.array([a for a in omegahigh if 0 < a < 1.01])
    cutoffomegahigh = np.array([a for a in omegahigh if 1 < a < cutoff])
    print(cutoffomega)
    intensities.append(j)
    print(parameternames)
    pulse=np.load('./output/pulse/pulse'+parameternames)
    spectra = np.load('./output/spectra/spectra'+parameternames)
    pulsehigh=np.load('./output/pulse/pulse'+parameternameshigh)
    spectrahigh = np.load('./output/spectra/spectra'+parameternameshigh)
    # for k in range(0,pulse[:,1].size):
    #     spectra.append(np.abs(FT(pulse[k,:]*blackman(steps)))**2)
    spectra=np.array(spectra)
    buffer = 3
    lincut = int((params['t_final'] * params['omega0']) / (2 * np.pi)) + buffer
    cutspectra=spectra[:,int(spectra[0,:].size/2)+lincut:int(spectra[0,:].size/2)+lincut+cutoffomega.size]
    linspectra=spectra[:,int(spectra[0,:].size/2):int(spectra[0,:].size/2)+linomega.size]
    cutspectra[cutspectra < 1e-9] = 0.

    cutspectrahigh=spectrahigh[:,int(spectrahigh[0,:].size/2)+lincut:int(spectrahigh[0,:].size/2)+lincut+cutoffomegahigh.size]
    linspectrahigh=spectrahigh[:,int(spectrahigh[0,:].size/2):int(spectrahigh[0,:].size/2)+linomegahigh.size]
    cutspectrahigh[cutspectrahigh < 1e-9] = 0.

    inomega = params['omega0'] * (np.arange(encode[:, 0].size) + 1) / encode[:, 0].size
    print(inomega.size)
    encodewave = np.array(
        [[0.04 * np.real(np.exp(1j * w * t)) * np.sin(np.pi * t / params['t_final']) ** 2 for w in inomega] for
         t in times])
    timefield = np.dot(encodewave, encode)
    timefield=np.transpose(timefield)


    encodewavehigh = np.array(
        [[0.04 * np.real(np.exp(1j * w * t)) * np.sin(np.pi * t / params['t_final']) ** 2 for w in inomega] for
         t in times])
    timefieldhigh = np.dot(encodewavehigh, encode)
    timefieldhigh=np.transpose(timefieldhigh)
    # analysis(pulse, pulse_parameters)
    # analysis(linspectra, lin_parameters)
    # # # analysis(cutspectra, high_parameters)

    fscorepulse.append(fscore(pulse, pulse_parameters))
    fscorehighspectra.append(fscore(cutspectra,high_parameters))
    fscorelinspectra.append(fscore(linspectra,lin_parameters))
    fscoreinput.append(fscore(timefield,input_parameters))

    fscorepulsehigh.append(fscore(pulsehigh, pulse_parameters))
    fscorehighspectrahigh.append(fscore(cutspectrahigh, high_parameters))
    fscorelinspectrahigh.append(fscore(linspectrahigh, lin_parameters))
    fscoreinputhigh.append(fscore(timefieldhigh, input_parameters))


    #
    # if c <1 or c>10:
    #     plt.plot(times,pulse[0,:],color=color,label='intensity %s' % (j))
    #     # plt.plot(times,pulse[1,:],color=color,linestyle='--')
    # else:
    #     plt.plot(times,pulse[0,:],color=color)
    #     # plt.plot(times,pulse[1,:],color=color,linestyle='--')
    #
    # plt.xlabel('Time')
    # plt.ylabel('Pulse Intensity')
    # plt.legend()
    #
    # if c<1 or c>10:
    #     plt.semilogy(omega,spectra[0,:],color=color,label='input intensity %s' % (j))
    # else:
    #     plt.semilogy(omega,spectra[0,:],color=color)
    #
    # # plt.semilogy(omega,spectra[1,:],color=color,linestyle='--')
    # plt.xlabel('$\\frac{\omega}{\omega_0}$')
    # plt.ylabel('Pulse Intensity')
    # plt.xlim([0, 20])
    # plt.legend()
    # #
    # if c<1 or c>29:
    #     plt.plot(omega,spectra[0,:],color=color,label='input intensity %s' % (j))
    # else:
    #     plt.plot(omega,spectra[0,:],color=color)

    # plt.semilogy(omega,spectra[1,:],color=color,linestyle='--')
    # plt.xlabel('$\\frac{\omega}{\omega_0}$')
    # plt.ylabel('Pulse Intensity')
    # plt.xlim([0, 20])
    # plt.legend()
    c+=1
    print(c)




# plt.show()
intensities=np.array(intensities)
p=plt.plot(intensities,fscorepulse, label='output pulse')
plt.plot(intensities,fscorepulsehigh, linestyle='dashed', color=p[-1].get_color())

p=plt.plot(intensities,fscorehighspectra, label='output spectra (high harmonics)')
plt.plot(intensities,fscorehighspectrahigh, linestyle='dashed', color=p[-1].get_color())

p=plt.plot(intensities,fscorelinspectra, label='output spectra (linear)')
plt.plot(intensities,fscorelinspectrahigh, linestyle='dashed', color=p[-1].get_color())

p=plt.plot(intensities,fscoreinput, label='input pulse')
plt.plot(intensities,fscoreinputhigh, linestyle='dashed', color=p[-1].get_color())


plt.plot(intensities,np.ones(intensities.size)*fscore(digits.data[:testn],raw_parameters),linestyle='dashdot', label='input data (Linear)')
plt.plot(intensities,np.ones(intensities.size)*fscore(digits.data[:testn],raw_parameters_rbf),linestyle='dashdot', label='input data (rbf)')
plt.xlabel('periods of frequency $\omega_0$')
plt.ylabel('F score')
plt.legend()
plt.show()