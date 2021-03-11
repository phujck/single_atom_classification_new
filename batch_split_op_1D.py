import numpy as np
import pyfftw
import pickle
from numpy import linalg  # Linear algebra for dense matrix
from numba import njit
from multiprocessing import cpu_count
# from numba.core.registry import CPUDispatcher
# from types import FunctionType

class BatchSplitOp1D(object):
    """
    The second-order split-operator propagator of the 1DSchrdoinger equation in the coordinate representation
    with the time-dependent Hamiltonian

        H = K(p, t) + V(x, t)

    """
    def __init__(self, *, x_grid_dim, x_amplitude, v, k, dt, batch_size=1,
                diff_k=None, diff_v=None, t=0, abs_boundary=1.,
                 fftw_wisdom_fname='fftw.wisdom', **kwargs):
        """
        :param x_grid_dim: the grid size
        :param batch_size: the number of system to be propagated in the batch (i.e., batch size)
        :param x_amplitude: the maximum value of the coordinates
        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)
        :param diff_k: the derivative of the potential energy for the Ehrenfest theorem calculations
        :param diff_v: the derivative of the kinetic energy for the Ehrenfest theorem calculations
        :param t: initial value of time
        :param dt: initial time increment
        :param abs_boundary: absorbing boundary
        :param fftw_wisdom_fname: File name from where the FFT wisdom will be loaded from and saved to
        :param kwargs: ignored
        """

        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.v = v
        self.k = k
        self.diff_v = diff_v
        self.t = t
        self.dt = dt
        self.abs_boundary = abs_boundary
        self.num_propagators = batch_size

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
        self.wavefunction = pyfftw.empty_aligned((batch_size, x_grid_dim), dtype=np.complex)

        # allocate the array for wave function in momentum representation
        self.wavefunction_p = pyfftw.empty_aligned((batch_size, x_grid_dim), dtype=np.complex)

        # allocate the array for calculating the momentum representation for the energy evaluation
        self.wavefunction_p_ = pyfftw.empty_aligned((batch_size, x_grid_dim), dtype=np.complex)

        # parameters for FFT
        self.fft_params = {
            "flags": ('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'),
            "threads": cpu_count(),                                       #Removed cpu_count from here
            "planning_timelimit": 60,
        }

        # FFT
        self.fft = pyfftw.FFTW(self.wavefunction, self.wavefunction_p, **self.fft_params)

        # iFFT
        self.ifft = pyfftw.FFTW(self.wavefunction_p, self.wavefunction, direction='FFTW_BACKWARD', **self.fft_params)

        # fft for momentum representation
        self.fft_p = pyfftw.FFTW(self.wavefunction_p, self.wavefunction_p_, **self.fft_params)

        # # Save the FFTW wisdom
        # with open(fftw_wisdom_fname, 'wb') as fftw_wisdow:
        #     pickle.dump(pyfftw.export_wisdom(), fftw_wisdow)

        ####################################################################################################
        #
        #   Initialize grids
        #
        ####################################################################################################

        # Check that all attributes were specified
        # make sure self.x_amplitude has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"

        # get coordinate step size
        dx = self.dx = 2. * self.x_amplitude / self.x_grid_dim

        # generate coordinate range
        self.x = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * self.dx
        x = self.x = self.x[np.newaxis, :]

        # generate momentum range as it corresponds to FFT frequencies
        self.p = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * (np.pi / self.x_amplitude)
        p = self.p = self.p[np.newaxis, :]

        # list of self.dt to monitor how the adaptive step method is working
        self.time_increments = []

        ####################################################################################################
        #
        # Codes for efficient evaluation
        #
        ####################################################################################################

        # Decide whether the potential depends on time
        try:
            v(x, 0)
            self.time_independent_v = False
        except TypeError:
            self.time_independent_v = True

        # Decide whether the kinetic energy depends on time
        try:
            k(p, 0)
            self.time_independent_k = False
        except TypeError:
            self.time_independent_k = True

        # pre-calculate the absorbing potential and the sequence of alternating signs

        abs_boundary = (abs_boundary if isinstance(abs_boundary, (float, int, np.ndarray)) else abs_boundary(x))
        abs_boundary = (-1) ** np.arange(self.wavefunction.shape[1]) * abs_boundary

        # Cache the potential if it does not depend on time
        if self.time_independent_v:
            pre_calculated_v = v(x)

            self.pre_calculated_expV = abs_boundary * np.exp(-0.5j * dt * pre_calculated_v)
        else:
            self.pre_calculated_expV = np.zeros_like(self.wavefunction)

            @njit
            def expV(pre_calculated_expV, t):
                """
                function to efficiently evaluate
                    wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
                """
                pre_calculated_expV[:] = abs_boundary * np.exp(-0.5j * dt * (v(x, t + 0.5 * dt)))

            self.expV = expV

        # Cache the kinetic energy if it does not depend on time
        if self.time_independent_k:
            pre_calculated_k = k(p)

            self.expK = np.exp(-1j * dt * pre_calculated_k)
        else:
            @njit
            def expK(wavefunction, t):
                """
                function to efficiently evaluate
                    wavefunction *= exp(-1j * dt * k)
                """
                wavefunction *= np.exp(-1j * dt * k(p, t + 0.5 * dt))

            self.expK = expK

        ####################################################################################################

        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        if diff_k and diff_v:
            # Get codes for efficiently calculating the Ehrenfest relations

            # Cache the potential if it does not depend on time
            if self.time_independent_v:
                @njit
                def get_v_average(density, t):
                    return np.sum(pre_calculated_v * density, axis=1)

                pre_calculated_diff_v = diff_v(x)
                @njit
                def get_p_average_rhs(density, t):
                    return np.sum(density * pre_calculated_diff_v, axis=1)

            else:
                @njit
                def get_v_average(density, t):
                    return np.sum(v(x, t) * density, axis=1)

                @njit
                def get_p_average_rhs(density, t):
                    return np.sum(density * diff_v(x, t), axis=1)

            self.get_p_average_rhs = get_p_average_rhs
            self.get_v_average = get_v_average

            # Cache the kinetic energy if it does not depend on time
            if self.time_independent_k:
                pre_calculated_diff_k = diff_k(p)
                @njit
                def get_x_average_rhs(density, t):
                    return np.sum(pre_calculated_diff_k * density, axis=1)

                @njit
                def get_k_average(density, t):
                    return np.sum(pre_calculated_k * density, axis=1)

            else:
                @njit
                def get_x_average_rhs(density, t):
                    return np.sum(diff_k(p, t) * density, axis=1)

                @njit
                def get_k_average(density, t):
                    return np.sum(k(p, t) * density, axis=1)

            self.get_x_average_rhs = get_x_average_rhs
            self.get_k_average = get_k_average

            @njit
            def get_x_average(density):
                return np.sum(x * density, axis=1)

            self.get_x_average = get_x_average

            @njit
            def get_p_average(density):
                return np.sum(p * density, axis=1)

            self.get_p_average = get_p_average

            # since the variable time propagator is used, we record the time when expectation values are calculated
            self.times = []

            # Lists where the expectation values of x and p
            self.x_average = []
            self.p_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for x and p
            self.x_average_rhs = []
            self.p_average_rhs = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # sequence of alternating signs for getting the wavefunction in the momentum representation
            self.minus = (-1) ** np.arange(self.x_grid_dim)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.is_ehrenfest = True
        else:
            # Since diff_v and diff_k are not specified, we are not going to evaluate the Ehrenfest relations
            self.is_ehrenfest = False

            # still evaluate the average of x
            @njit
            def get_x_average(wavefunction):
                return np.sum(x * np.abs(wavefunction) ** 2, axis=1)  * dx

            self.x_average = []

            self.get_x_average = get_x_average

    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """
        for _ in range(time_steps):
            # advance the wavefunction by dt
            self.single_step_propagation()

            # calculate the Ehrenfest theorems
            self.get_ehrenfest()

        return self.wavefunction

    def single_step_propagation(self):
        """
        Propagate the wavefunction, saved in self.wavefunction_next, by a single time-step
        :param dt: time-step
        :return: None
        """
        wavefunction = self.wavefunction

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        if not self.time_independent_v:
            self.expV(self.pre_calculated_expV, self.t)

        wavefunction *= self.pre_calculated_expV

        # going to the momentum representation
        wavefunction = self.fft(wavefunction)

        # efficiently evaluate
        #   wavefunction *= exp(-1j * dt * k)
        if self.time_independent_k:
            wavefunction *= self.expK
        else:
            self.expK(wavefunction, self.t)

        # going back to the coordinate representation
        wavefunction = self.ifft(wavefunction)

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        wavefunction *= self.pre_calculated_expV

        # make a time increment
        self.t += self.dt

        # normalize
        norms = linalg.norm(wavefunction, axis=1).reshape(-1, 1)
        norms *= np.sqrt(self.dx)
        wavefunction /= norms

    def get_ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        if self.is_ehrenfest:
            # save the current time
            self.times.append(self.t)

            # alias
            density = self.wavefunction_p

            # evaluate the coordinate density
            np.abs(self.wavefunction, out=density)
            density *= density
            # normalize
            density /= density.sum()

            # save the current value of <x>
            self.x_average.append(
                self.get_x_average(density.real)
            )

            self.p_average_rhs.append(
                -self.get_p_average_rhs(density.real, self.t)
            )

            # save the potential energy
            self.hamiltonian_average.append(
                self.get_v_average(density.real, self.t)
            )

            # calculate density in the momentum representation
            np.copyto(density, self.wavefunction)
            density *= self.minus
            density = self.fft_p(density)

            # get the density in the momentum space
            np.abs(density, out=density)
            density *= density
            # normalize
            density /= density.sum()

            # save the current value of <p>
            self.p_average.append(self.get_p_average(density.real))

            self.x_average_rhs.append(self.get_x_average_rhs(density.real, self.t))

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += self.get_k_average(density.real, self.t)
        else:
            # Evaluate the average of x
            self.x_average.append(
                self.get_x_average(self.wavefunction).real
            )

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numpy array or function specifying the wave function
        :return: self
        """
        # if isinstance(wavefunc, (CPUDispatcher, FunctionType)):
        #     self.wavefunction[:] = wavefunc(self.x)
        #
        # elif isinstance(wavefunc, np.ndarray):
        #     # wavefunction is supplied as an array
        #     self.wavefunction[:] = wavefunc


        if isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array
            self.wavefunction[:] = wavefunc
        else:
            raise ValueError("wavefunc must be either function or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dx)

        return self
