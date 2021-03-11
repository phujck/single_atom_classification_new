
import sys
import os
from itertools import repeat
from imag_time_propagation import ImgTimePropagation, np, fftpack
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from scipy.signal import blackman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # enable log color plot
from numba import njit  # compile python
# from tqdm import tqdm # progress bar
from sklearn import datasets
import multiprocessing


if __name__ == '__main__':
    def _getThreads():
        """ Returns the number of available threads on a posix/win based system """
        if sys.platform == 'win32':
            return (int)(os.environ['NUMBER_OF_PROCESSORS'])
        else:
            return (int)(os.popen('grep -c cores /proc/cpuinfo').read())


    threads_available = _getThreads()
    threads = threads_available
    os.environ["OMP_NUM_THREADS"] = '{}'.format(threads)
    os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(threads)
    os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
    os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)




    ########################################################################################################################
    #
    #   Utilities: Define functions for testing and visualizing
    #
    ########################################################################################################################

    def test_propagation(sys, t_final):
        """
        Run tests for the specified propagators and plot the probability density
        of the time dependent propagation
        :param sys: class that propagates
        """
        iterations = 748
        steps = int(np.ceil(t_final / sys.dt / iterations))

        # display the propagator
        plt.imshow(
            [np.abs(sys.propagate(steps)) ** 2 for _ in range(iterations)],
            origin='lower',
            norm=LogNorm(vmin=1e-12, vmax=0.1),
            aspect=0.4, # image aspect ratio
            extent=[sys.x.min(), sys.x.max(), 0., sys.t]
        )
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')
        plt.colorbar()


    def test_Ehrenfest1(sys):
        """
        Test the first Ehenfest theorem for the specified quantum system
        """
        times = sys.dt * np.arange(len(sys.x_average))

        dx_dt = np.gradient(sys.x_average, sys.dt)

        print("{:.2e}".format(np.linalg.norm(dx_dt - sys.x_average_rhs)))

        plt.plot(times, dx_dt, '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
        plt.plot(times, sys.x_average_rhs, '--b', label='$\\langle\\hat{p}\\rangle$')
        plt.legend()
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')


    def test_Ehrenfest2(sys):
        """
        Test the second Ehenfest theorem for the specified quantum system
        """
        times = sys.dt * np.arange(len(sys.p_average))

        dp_dt = np.gradient(sys.p_average, sys.dt)

        print("{:.2e}".format(np.linalg.norm(dp_dt - sys.p_average_rhs)))

        plt.plot(times, dp_dt, '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
        plt.plot(times, sys.p_average_rhs, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
        plt.legend()
        plt.ylabel('force')
        plt.xlabel('time $t$ (a.u.)')


    def frft(x, alpha):
        """
        Implementation of the Fractional Fourier Transform (FRFT)
        :param x: array of data to be transformed
        :param alpha: parameter of FRFT
        :return: FRFT(x)
        """
        k = np.arange(x.size)

        y = np.hstack([
            x * np.exp(-np.pi * 1j * k**2 * alpha),
            np.zeros(x.size, dtype=np.complex)
        ])
        z = np.hstack([
            np.exp(np.pi * 1j * k**2 * alpha),
            np.exp(np.pi * 1j * (k - x.size)**2 * alpha)
        ])

        G = fftpack.ifft(
            fftpack.fft(y, overwrite_x=True) * fftpack.fft(z, overwrite_x=True),
            overwrite_x=True
        )

        return np.exp(-np.pi * 1j * k**2 * alpha) * G[:x.size]



    ########################################################################################################################
    #
    #   Define functions propagating to calculate the HHG spectrum
    #
    ########################################################################################################################

    def get_hhg_spectrum(F, omega_0=0.06,omega_1=0.12,omega_2=0.18, data=np.zeros(10), get_omega=False, test_mode=False):
        """
        Evaluate the HHG spectrum
        :param F: the amplitude of the laser field strength
        :param get_omega: boolean flag whether to return the omegas
        :param coords: tuple of data to be encoded into laser

        :return:
        """

        ####################################################################################################################
        #
        # Define parameters of an atom (a single-electron model of Ar) in the external laser field
        #
        ####################################################################################################################
        # laser field frequency
        omega_laser = omega_0
        omegas=np.linspace(omega_1,omega_2,len(data))

        # the final time of propagation (= 8 periods of laser oscillations)
        t_final = 12 * 2. * np.pi / omega_laser
        # the amplitude of grid
        x_amplitude = 300.

        # the time step
        dt = 0.05


        @njit
        def laser(t):
            """
            The strength of the laser field.
            Always add an envelop to the laser field to avoid all sorts of artifacts.
            We use a sin**2 envelope, which resembles the Blackman filter
            """
            # laser = 2*np.sin(omega_laser * t)
            laser = np.sin(omega_laser * t)
            for freq,dat in zip(omegas,data):
                laser+=dat*np.sin(freq*t)
            laser*=F*np.sin(np.pi * t / t_final) ** 2
            return laser

        @njit
        def encode_laser(t):
            """
            The strength of the laser field.
            Always add an envelop to the laser field to avoid all sorts of artifacts.
            We use a sin**2 envelope, which resembles the Blackman filter
            """
            # laser = 2*np.sin(omega_laser * t)
            laser = 0
            for freq, dat in zip(omegas, data):
                laser += dat * np.sin(freq*t)
            laser *= F * np.sin(np.pi * t / t_final) ** 2
            return laser

        # pulse_times = np.linspace(0, t_final, t_final / dt)
        # tl_pulse = [F * np.sin(omega_laser * t) * np.sin(np.pi * t / t_final) ** 2 for t in pulse_times]
        # encode_pulse = [encode_laser(t) for t in pulse_times]
        # full_pulse = [laser(t) for t in pulse_times]
        # plt.plot(tl_pulse)
        # plt.plot(encode_pulse)
        # plt.plot(full_pulse)
        # plt.show()
        # np.save('tl_input', tl_pulse)
        # np.save('encoding_input', encode_pulse)
        # np.save('full_input', full_pulse)

        @njit
        def v(x, t=0.):
            """
            Potential energy.

            Define the  potential energy as a sum of the soft core Columb potential
            and the laser field interaction in the dipole approximation.
            """
            return -1. / np.sqrt(x ** 2 + 2.37) + x * laser(t)


        @njit
        def diff_v(x, t=0.):
            """
            the derivative of the potential energy
            """
            return x * (x ** 2 + 2.37) ** (-1.5) + laser(t)


        @njit
        def k(p, t=0.):
            """
            Non-relativistic kinetic energy
            """
            return 0.5 * p ** 2


        @njit
        def diff_k(p, t=0.):
            """
            the derivative of the kinetic energy for Ehrenfest theorem evaluation
            """
            return p

        @njit
        def abs_boundary(x):
            """
            Absorbing boundary similar to the Blackman filter
            """
            return np.sin(0.5 * np.pi * (x + x_amplitude) / x_amplitude) ** (0.05 * dt)

        sys_params = dict(
            dt=dt,
            x_grid_dim=2 * 1024,
            x_amplitude=x_amplitude,

            k=k,
            diff_k=diff_k,
            v=v,
            diff_v=diff_v,

            abs_boundary=abs_boundary,
        )



        ####################################################################################################################
        #
        # propagate
        #
        ####################################################################################################################

        sys = ImgTimePropagation(**sys_params)

        # Set the ground state wavefunction as the initial condition
        sys.get_stationary_states(1)
        sys.set_wavefunction(sys.stationary_states[0])

        if test_mode:
            plt.title("No absorbing boundary, $|\\Psi(x, t)|^2$")
            test_propagation(sys, t_final)
            plt.show()

            plt.subplot(121)
            print("\nError in the first Ehrenfest relation: ")
            test_Ehrenfest1(sys)

            plt.subplot(122)
            print("\nError in the second Ehrenfest relation: ")
            test_Ehrenfest2(sys)

            plt.show()
        else:
            steps = int(np.ceil(t_final / sys.dt))
            sys.propagate(steps)
            print("F = {} Completed!".format(F))



        ####################################################################################################################
        #
        # Evaluate the spectrum
        #
        ####################################################################################################################

        # N = len(sys.x_average)
        # k = np.arange(N)
        #
        # # frequency range
        # omega = (k - N / 2) * np.pi / (0.5 * sys.t)
        #
        # # spectra of the
        # spectrum = np.abs(
        #     # used windows fourier transform to calculate the spectra
        #     # rhttp://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
        #     fftpack.fft((-1) ** k * blackman(N) * sys.x_average)
        # ) ** 2
        # spectrum /= spectrum.max()

        N = len(sys.p_average_rhs)
        k = np.arange(N)

        # generate the desired momentum grid
        omega_amplitude = 2.

        domega = 2. * omega_amplitude / N
        omega = (k - N / 2) * domega

        delta = sys.dt * domega / (2. * np.pi)
        #
        # plt.plot(sys.p_average_rhs)
        # plt.show()
        # np.save('output_pulse', sys.p_average_rhs)
        spectrum = np.abs(
            frft(blackman(N) * sys.p_average_rhs * np.exp(np.pi * 1j * k * N * delta), delta)
        ) ** 2
        spectrum /= spectrum.max()

        return (spectrum, omega / omega_laser) if get_omega else spectrum

    #
    # print('getting data')
    # data=circle_data(1,1,0.8,5000)
    # print(data)
    # print(data.coords)
    # data.plot()
    # print('now evolving')
    # F=0.04
    # spectrum, harmonic = get_hhg_spectrum(F, get_omega=True)


    ########################################################################################################################
    #
    #   Run simulations
    #
    ########################################################################################################################

    # spectrum, harmonic = get_hhg_spectrum(0.04, get_omega=True, test_mode=True)
    #
    # plt.semilogy(harmonic, spectrum)
    # plt.ylabel('spectrum (arbitrary units)')
    # plt.xlabel('frequency / $\\omega_L$')
    # plt.xlim([0, 45.])
    # plt.ylim([1e-15, 1.])
    # plt.show()

    # fields_strengths = np.linspace(0, 0.04, 400)

    # ignore the zero
    # fields_strengths = fields_strengths[1:]



    # spectra = [[get_hhg_spectrum(F,a), a] for a in data.coords]

    digits = datasets.load_digits(return_X_y=0)
    encode = np.transpose(digits.data)
    DigitData = np.transpose(encode / np.sum(encode, axis=0))
    # DigitData=digits.data
    omega_0=0.06
    omega_1=2*omega_0
    omega_2=3*omega_0
    pool = multiprocessing.get_context("fork").Pool(threads_available)
    # spectra=pool.map(pool_spectra,data.coords)
    # for F in np.linspace(0.0002,0.001,5):
    # intensities=10**np.linspace(-5,-2,10)
    # intensities=[10**-2]
    # get_hhg_spectrum(0.001,2 * omega_0, 8 * omega_0,data=DigitData[28,:], get_omega=True)
    # intensities=10**np.linspace(-4,-1.5,15)
    intensities=[10**-1]
    #
    for F in intensities:
        spectrum, harmonic = get_hhg_spectrum(F, get_omega=True)
        for omega_2 in [4*omega_0,10*omega_0]:
            # print(data)
            # print(data.coords)
            # data.plot()
            print('now evolving')
            print(DigitData.shape)
            # F = 0.04
            spectra=pool.starmap(get_hhg_spectrum,zip(repeat(F),repeat(omega_0),repeat(omega_1),repeat(omega_2),DigitData,repeat(False),repeat(False)))
            print(len(spectra))
            # pool.join()
            # spectrum, harmonic = get_hhg_spectrum(F, get_omega=True)
            # spectra.append(spectrum)

            # spectra = np.array(spectra)
            # for spectrum in spectra:
            #     # print(spectra.size)
            #     plt.semilogy(harmonic, spectrum)
            # plt.ylabel('spectrum (arbitrary units)')
            # plt.xlabel('frequency / $\\omega_L$')
            # # plt.legend()
            # # plt.xlim([0, 45.])
            # plt.ylim([1e-15, 1.])
            # plt.show()


            data_dict=dict(spectra=spectra,harmonics=harmonic,min_omega=omega_1,max_omega=omega_2,field_strength=F)
            outfile = './Data/Digits:fieldstrength={},minomega={},maxomega={}.npz'.format(F,omega_1,omega_2)
            np.savez(outfile, **data_dict)

    # # plot the results
    # plt.imshow(
    #     spectra.T,
    #     origin='lower',
    #     norm=LogNorm(vmin=1e-14, vmax=1e0),
    #     aspect=0.001, # image aspect ratio
    #     extent=[fields_strengths.min(), fields_strengths.max(), harmonic.min(), harmonic.max()]
    # )
    # plt.ylim([0, 45.])
    # plt.show()