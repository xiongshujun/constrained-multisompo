import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jax import jacfwd, jacrev #!TODO: compare numerical results from between these two

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
        Q   := initial orthonormal system in tangent space along the trajectory
                (N, N) np.ndarray
        J   := differential Df function \R^n -> \R^n
                Callable
        """

        self.fn = f
        self.J  = jacfwd(f) 
        self.h  = h_0
        self.Q  = np.identity(len(h_0))

    def set_h(self, h):
        self.h = h

    def set_Q(self, Q):
        self.Q = Q

    def set_J(self, Jac):
        self.J = Jac

    def evolve(self, t_sim : float, delta_t : float):
        """
        Evolves h and Q from time t to time t + t_sim with timesteps of delta_t

        INPUTS
            t_sim   := total amount of elapsed time desired
            delta_t := time interval for each step taken

        OUTPUTS
            none

        CLASS VARIABLE CHANGES
            self.h  := updated to h_{t_sim} along discrete intervals of delta_t
            self.Q  := updated to Q_{t_sim} along discrete intervals of delta_t
        """
        s_sim   = int(t_sim/delta_t)
        for i in range(s_sim):
            self.h = self.fn(self.h)
            D      = self.J(self.h)
            self.Q = D @ self.Q

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

            
        CLASS VARIABLE CHANGES
            self.spectra := spectra

        """

        # LOOPING
        spectra = [0]*m
        s_sim   = int(t_sim/delta_t)

        h = self.h
        Q = self.Q
        for s in range(s_sim):
            h = self.fn(h)
            D = self.J(h)
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

"""
Code structure of getting the Lyapunov spectrum of a dynamical system
1) Initialize Lyapunov() object
        fn  : function defining the dynamical system
        h_0 : initial value
2) Call evolve() for some t_transient, delta_t to make sure the system is on the attractor and Q is convergent onto the Oseledets matrix
        Save the resulting self.h and self.Q values externally as h_0 and Q_0 respectiely for testing
3) Find an acceptable reorthonormalization interval s_ONS = O(log(k_2)/(\lambda_max - \lambda_min))/delta_t
        k_2 is some acceptable condition number
            k_2 = R^{s + s_ONS}_11/R^{s + S_ONS}_mm := ratio of the smallest and largest singular values in the QR-decomposition of Q_{s + s_ONS}
        s_ONS = t_ONS/delta_t := defines every number of steps that QR-decomposition is performed at

        Supplemental of Engelken et al details two methods to find s_ONS
            A) Get a rough estimate of the spectrum using a short t_sim and small t_ONS.
                    Then, repeat with longer simulation time and a t_ONS based on Lyapunov spectrum of this estimate
            B) Iteratively adapt t_ONS on a short simulation run to get an acceptable condition number
        There are other methods cited in Supplemental as well. This code base will be using method (A) for now.
4) Call get_spectra() for the desired spectra in the form of a python list
"""

#####################################
#            REPRODUCING            #
#             ENGELKAN              #
#####################################

# Network parameters
# Note that in Engelken et al, parameters are as follows
    # delta_t = 0.1 \tau
    # t_ONS   = \tau
    # t_sim   = 10^3 \tau
N = 100
g = 1
tau = 10
delta_t = 0.1*tau

J = np.random.normal(loc = 0, scale = g**2, size = (N, N))
h = np.random.randn(N)
phi = jnp.tanh

t_transient = 100*tau
t_sim       = 1000*tau

# Define the dynamical function
def nn(h : np.ndarray, phi: callable = phi, J : np.ndarray = J, tau : float = tau) -> np.ndarray:
    
    # conversions for jax calls
    # if isinstance()
    grad = -1*h + J @ phi(h)
    grad /= tau
    return grad

L = Lyapunov(f = nn, h_0 = h) # initialization
L.evolve(t_sim = t_transient, delta_t = delta_t) # evolve to a point where the system is on a transient
h_0 = L.h # save model parameters
Q_0 = L.h # save model parameters

# Find s_ONS
spec, k = L.get_spectra(m = N, s_ons = int(tau/delta_t), delta_t = delta_t)
s_ons = int((np.log(k)/(spec[0] - spec[-1]))/delta_t)

# Run full simulation + plot results
spec, _ = L.get_spectra(m = N, s_ons = s_ons, t_sim = t_sim, delta_t = delta_t)

plt.scatter(np.arange(len(spec)), spec)
plt.show()