from hub_lats import *
        
class hhg(Lattice):
    def __init__(self,field,nup,ndown,nx,ny,U,t=0.52,F0=10.,a=4.,lat_type='square',bc=None,nnn=False):
        Lattice.__init__(self,nx,ny,lat_type,nnn,bc)
        self.nup = nup
        self.ndown = ndown
        self.ne = nup+ndown
        #input units: THz (field), eV (t, U), MV/cm (peak amplitude), Angstroms (lattice cst) 
        #converts to a'.u, which are atomic units but with energy normalised to t, so
        #Note, hbar=e=m_e=1/4pi*ep_0=1, and c=1/alpha=137
        factor = 1./(t*0.036749323)
        self.factor = factor
        self.U = U/t
        self.t = 1.
        #field is the angular frequency, and freq the frequency = field/2pi
        self.field = field*factor*0.0001519828442
        self.freq = self.field/(2.*3.14159265359)
        self.a = (a*1.889726125)/factor
        self.F0 = F0*1.944689151e-4*(factor**2)
        assert self.nup<=self.nsites,'Too many ups!'
        assert self.ndown<=self.nsites,'Too many downs!'

'''envelope options are "sin", "cos", "unity", "gaussian", and for the carrier wave are "sin", "cos", "zero", "gaussian"
ellipticity is between -1 and 1 and multiplies the vector potential by 1/sqrt(1+ellipticity^2) in x direction and ellipticity/sqrt(1+ellipticity^2) 
in y direction. ellipticity=1 and -1 are right-handed and left-handed circular polarisation. =0 is linear polarisation. 
if ellipticity=None then the 2D vector potentials are not multiplied by any factor.'''
class pulse(object):
    def __init__(self, lat, envelope="sin", carrierx="sin", carriery="sin", cycles=5., ellipticity=0.0, CEP=0.0, sigma=None):
        self.lat = lat
        self.envelope = envelope
        self.carrierx = carrierx
        self.carriery = carriery
        self.cycles = cycles
        self.ellipticity = ellipticity
        self.CEP = CEP
        self.sigma = sigma

    def calc_phi(self, time):
        phi = self.lat.a*self.lat.F0/self.lat.field
        if self.envelope=="sin":
            phi *= np.sin(self.lat.field*time/(2.*self.cycles))**2.
        elif self.envelope=="cos":
            phi *= np.cos(self.lat.field*time/(2.*self.cycles))**2.
        elif self.envelope=="gaussian":
            phi *= np.exp(-time**2./(2.*self.sigma**2.))
        elif self.envelope=="unity":
            pass

        def carrier(cwave):
            if cwave=="sin":
                return np.sin(self.lat.field*time+self.CEP)
            elif cwave=="cos":
                return np.cos(self.lat.field*time+self.CEP)
            elif cwave=="gaussian":
                return np.exp(-time**2./(2.*self.sigma**2.))
            elif cwave=="zero":
                return 0.

        if self.lat.dim==1:
            return (phi*carrier(self.carrierx), 0.0)
        elif self.ellipticity==None:
            return (phi*carrier(self.carrierx), phi*carrier(self.carriery))
        else:
            ep1 = 1./np.sqrt(1.+self.ellipticity**2.)
            ep2 = self.ellipticity*ep1
            return (phi*ep1*carrier(self.carrierx), phi*ep2*carrier(self.carriery))

    '''phase=True plots the phase, =False plots the electric field'''
    def plot_pulse(self, phase=True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        ts = np.arange(0., self.cycles/self.lat.freq, 0.1)
        phix, phiy = [], []
        for t in ts:
            phi = self.calc_phi(t)
            phix.append(phi[0])
            phiy.append(phi[1])
        ts = np.linspace(0., self.cycles, len(ts))
        if self.lat.dim==1:
            if phase:
                plt.plot(ts, phix)
                plt.ylabel('Phase')
            else:
                E = -np.gradient(phix)
                plt.plot(ts, E/max(E))
                plt.ylabel('Normalised Electric Field')
            plt.xlabel('Time [cycles]')
        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            if phase:
                ax.plot(ts, phix, phiy)
                ax.set_ylabel('$\phi_x$')
                ax.set_zlabel('$\phi_y$')
            else:
                Ex = -np.gradient(phix)
                Ey = -np.gradient(phiy)
                ax.plot(ts, Ex/max(Ex), Ey/max(Ey))
                ax.set_ylabel('$E_x$')
                ax.set_zlabel('$E_y$')
            ax.set_xlabel('Time [cycles]')
        plt.show()
        return

#current=current timestep, total=total number of timesteps to be calculated
def progress(total,current):
    if total<10:
        print("Simulation Progress: " + str(round(100*current/total)) + "%")
    elif current%int(total/10)==0:
        print("Simulation Progress: " + str(round(100*current/total)) + "%")
    return
