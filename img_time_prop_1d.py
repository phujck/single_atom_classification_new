import numpy as np
import pyfftw
import pickle
from numpy import linalg  # Linear algebra for dense matrix
from numba import njit
from multiprocessing import cpu_count

def imag_time_prop_1d(*, x_grid_dim, x_amplitude, v, k, dt, init_wavefunction=None, nsteps=5000,
                    abs_boundary=1., fftw_wisdom_fname='fftw.wisdom', **kwargs):
    """
    Imaginary time propagator of the 1D Schrodinger equation to get the ground state and ground state energy

    :param x_grid_dim: the grid size
    :param x_amplitude: the maximum value of the coordinates
    :param v: the potential energy (as a function)
    :param k: the kinetic energy (as a function)
    :param dt: initial time increment
    :param init_wavefunction: initial guess for wavefunction
    :param epsilon: relative error tolerance
    :param abs_boundary: absorbing boundary
    :param fftw_wisdom_fname: File name from where the FFT wisdom will be loaded from and saved to
    :param kwargs: ignored

    :return: wavefunction, ground state energy
    """
    print("\nStarting imaginary time propagation")

    # Check that all attributes were specified
    # make sure self.x_amplitude has a value of power of 2
    assert 2 ** int(np.log2(x_grid_dim)) == x_grid_dim, \
        "A value of the grid size (x_grid_dim) must be a power of 2"

    ####################################################################################################
    #
    #   Initialize Fourier transform for efficient calculations
    #
    ####################################################################################################

    # Load the FFTW wisdom
    # try:
    #     with open(fftw_wisdom_fname, 'rb') as fftw_wisdow:
    #         pyfftw.import_wisdom(pickle.load(fftw_wisdow))
    # except FileNotFoundError:
    #     pass

    # allocate the array for wave function
    wavefunction = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

    # allocate the array for wave function in momentum representation
    wavefunction_p = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

    # allocate the array for calculating the momentum representation for the energy evaluation
    wavefunction_p_ = pyfftw.empty_aligned(x_grid_dim, dtype=np.complex)

    # parameters for FFT
    fft_params = {
        "flags": ('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'),
        "threads": cpu_count(),                                       #removed cpu_count from here
        "planning_timelimit": 60,
    }

    # FFT
    fft = pyfftw.FFTW(wavefunction, wavefunction_p, **fft_params)

    # iFFT
    ifft = pyfftw.FFTW(wavefunction_p, wavefunction, direction='FFTW_BACKWARD', **fft_params)

    # fft for momentum representation
    fft_p = pyfftw.FFTW(wavefunction_p, wavefunction_p_, **fft_params)

    # Save the FFTW wisdom
    # with open(fftw_wisdom_fname, 'wb') as fftw_wisdow:
    #     pickle.dump(pyfftw.export_wisdom(), fftw_wisdow)

    ####################################################################################################
    #
    #   Initialize grids
    #
    ####################################################################################################

    # get coordinate step size
    dx = 2. * x_amplitude / x_grid_dim

    # generate coordinate range
    x = (np.arange(x_grid_dim) - x_grid_dim / 2) * dx

    # generate momentum range as it corresponds to FFT frequencies
    p = (np.arange(x_grid_dim) - x_grid_dim / 2) * (np.pi / x_amplitude)

    # tha array of alternating signs for going to the momentum representation
    minues = (-1) ** np.arange(x_grid_dim)

    # evaluate the potential energy
    try:
        v = v(x)
    except TypeError:
        v = v(x, 0.)

    v_min = v.min()
    v -= v_min

    # evaluate the kinetic energy
    try:
        k = k(p)
    except TypeError:
        k = k(p, 0.)

    k_min = k.min()
    k -= k_min

    # pre-calculate the absorbing potential and the sequence of alternating signs
    abs_boundary = (abs_boundary if isinstance(abs_boundary, (float, int, np.ndarray)) else abs_boundary(x))

    # precalucate the exponent of the potential and kinetic energy
    img_exp_v = (-1) ** np.arange(x.size) * abs_boundary * np.exp(-0.5 * dt * v)
    img_exp_k = np.exp(-dt * k)

    # initial guess for the wave function
    wavefunction[:] = (np.exp(-v) + 0j if init_wavefunction is None else init_wavefunction)

    @njit
    def get_energy(psi, pis_p):
        """
        Calculate the energy for a given wave function and its momentum representaion
        :return: float
        """
        density = np.abs(psi) ** 2
        density /= density.sum()

        energy = np.sum(v * density)

        # get momentum density
        density = np.abs(pis_p) ** 2
        density /= density.sum()

        energy += np.sum(k * density)

        return energy + v_min


    for _ in range(nsteps):

        wavefunction *= img_exp_v

        # going to the momentum representation
        wavefunction_p = fft(wavefunction)

        wavefunction_p *= img_exp_k

        # going back to the coordinate representation
        wavefunction = ifft(wavefunction_p)

        wavefunction *= img_exp_v

        wavefunction /= linalg.norm(wavefunction) * np.sqrt(dx)

    # get the wave function in the momentum representation for getting the energy
    # wavefunction_p[:] = wavefunction
    np.copyto(wavefunction_p, wavefunction)
    wavefunction_p *= minues
    wavefunction_p_ = fft_p(wavefunction_p)

    # calculate the energy
    energy = get_energy(wavefunction, wavefunction_p_)

    print("\n\nFinal current ground state energy = {:.4e}".format(energy))

    return wavefunction, energy
