import numpy as np
from jax        import jacfwd, jacrev #!TODO: compare numerical results from between these two
 
#####################################
#              CLASSES              #
#####################################

class Lyapunov():

    def __init__(self, f : callable, h_0 : np.ndarray):
        """
        STATE VARIABLES
        f   := function over which Lyapunov simulations will be done
                callable
        h   := initial state to be evolved by f
                (N, ) np.ndarray
        Jac := differential Df function \R^n -> \R^n
                Callable
        """

        self.fn  = f
        self.Jac = jacfwd(f) 
        self.h   = h_0

    def set_h(self, h):
        self.h = h

    def set_J(self, Jac):
        self.Jac = Jac

    def evolve(self, t_sim : float, delta_t : float):
        """
        Evolves h from time t to time t + t_sim with timesteps of delta_t

        INPUTS
            t_sim   := total amount of elapsed time desired
            delta_t := time interval for each step taken

        OUTPUTS
            none

        CLASS VARIABLE CHANGES
            self.h  := updated to h_{t_sim} along discrete intervals of delta_t
        """
        s_sim   = int(t_sim/delta_t)
        for i in range(s_sim):
            self.h = self.fn(self.h)
            D      = self.Jac(self.h)

    def get_spectra(self, m : int, s_ons : int = 1, t_sim : float = 100, delta_t : float = 0.1):
        """
        Calculate the first m Lyapunov exponents via reorthnormalization procedure (Benettin et al 1980)
            Procedure detailed in Supplemental of Engelken et al 2023: https://journals.aps.org/prresearch/supplemental/10.1103/PhysRevResearch.5.043044/supplement02.pdf

        INPUTS                        
            m       := desired number of Lyapunov exponents from the reorthnormalization procedure
                        int, default value of 10
            s_ons   := number of cycles between each QR-decomposition, equal to t_ONS/delta_T
                        int, default value of 1
            t_sim   := total time of the simulation
                        float, default value of 10
            delta_t := time interval between each call of self.fn
                        float, default value of 0.1

        OUTPUTS
            spectra := spectrum of the first m Lyapunov spectra sorted from largest to smallest value
                        list, len(spectra) == m
            k       := characteristic condition number

        CLASS VARIABLE CHANGES
            self.spectra := spectra
        """

        # LOOPING
        spectra = [0]*m
        s_sim   = int(t_sim/delta_t)

        h    = self.h
        Q, R = np.linalg.qr(np.random.randn(len(h), m)) # initialization of Q

        for s in range(s_sim):
            h = self.fn(h)
            D = self.Jac(h)
            Q = D @ Q

            if s % s_ons == 0:
                Q, R = np.linalg.qr(Q)
                for i in range(m):
                    spectra[i] += np.log(abs(R[i][i]))
                
        k = abs(R[0][0]/R[m - 1][m - 1]) # characteristic number from the last step in the ONS trials

        spectra.sort(reverse = True)
        spectra = np.array(spectra)
        spectra /= t_sim
        return spectra, k
    
#####################################
#              METHODS              #
#####################################
def entropy(spec) -> float:
    # Get the Kolmogorov-Sinai entropy of a spectrum
    # spec is a list or 1D numpy array

    h = 0
    for i in range(len(spec)):
        if spec[i] > 1:
            h += spec[i]

    return h

def dim(spec) -> float:
    # Get the Kaplan-Yorke attractor dimensionality of a spectrum
    # spec is a descending list or 1D numpy array

    d = 0
    k = 0
    sum = 0

    # find k
    for i in range(len(spec)):
        if k + spec[i] < 0:
            break
        else:
            k += 1
            sum += spec[i]
    
    d = k + sum/abs(spec[k + 1])    

    return d