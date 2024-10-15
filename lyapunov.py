import argparse

import numpy     as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from jax        import jacfwd, jacrev #!TODO: compare numerical results from between these two

#####################################
#           DOCUMENTATION           #
#####################################
"""
PARAMETRICIZATION

args.fn  := choice of which evolution rule the network as a whole will use. Engelken et al use a simple Sompolinsky model which we will build upon.
            Refer to "Learning Models" section for details on how each of these evolution rules are parameterized
args.N   := size of the network, which can be vaied across experiments
args.phi := activation function of each neuron (default is tanh(x) as in the Engelken et al paper, which also later tests out ReLU(x))
            !TODO: the original Sompolinsky Crisanti and Summers paper uses tanh(gx), see if this changes anything
args.g   := gain value.
            !TODO: see how to vary gain between regions
args.tau := rate-time constant

Note that in Engelken et al, algorithm parameters are as follows
    delta_t = 0.1 \tau
    t_ONS   = \tau
    t_sim   = 10^3 \tau
    Since dt = delta_t is used in controlling how much h_{t + 1} evolves from h_t (a value between 0 and 1), tau must be between 0 and 10
J := connectivity matrix drawn from a Gaussian of mean 0 and variance = g^2
    
PROCEDURE FOR GETTING SPECTRUM
1) Initialize Lyapunov() object
        fn  : function defining the dynamical system
        h_0 : initial value
2) Call evolve() for some t_transient, delta_t to make sure the system is on the attractor and Q is convergent onto the Oseledets matrix
        Save the resulting self.h externally as h_0 for testing
3) Find an acceptable reorthonormalization interval s_ONS = O(log(k_2)/(\lambda_max - \lambda_min))/delta_t
        k_2 is some acceptable condition number
            k_2 = R^{s + s_ONS}_11/R^{s + S_ONS}_mm := ratio of the smallest and largest singular values in the QR-decomposition of Q_{s + s_ONS}
        s_ONS = t_ONS/delta_t := defines every number of steps that QR-decomposition is performed at

        Supplemental of Engelken et al details two methods to find s_ONS
            A) Get a rough estimate of the spectrum using a short t_sim and small t_ONS.
                    Then, repeat with longer simulation time and a t_ONS based on Lyapunov spectrum of this estimate
            B) Iteratively adapt t_ONS on a short simulation run to get an acceptable condition number
        There are other methods cited in Supplemental as well. This code base uses method (A) for now.
4) Call get_spectra() for the desired spectra in the form of a python list sorted descending

PLOTTING
Plots spectrum as in Engelken et al 2023
    Lyapunov exponent value against i/N (normalized eigenvector number)
    Dotted line along y = 0
Figures saved to 'figures/' folder, titled with model type, activation type, gain, and tau parameters
"""

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
#            REPRODUCING            #
#             ENGELKAN              #
#####################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a classification model on tiled methane data.")

    parser.add_argument('--fn',             type=str,
                                            help="Evolution rule for the network",
                                            default="single_sompo",
                                            choices=['single_sompo', 'chaudhuri', 'multi_sompo', 'multi_sompo_low_rank'])
    parser.add_argument('--N',              type=int,
                                            help="Size of network",
                                            default=100)
    parser.add_argument('--phi',            type=str,
                                            help="Activation Function",
                                            default="tanh",
                                            choices=['tanh', 'ReLU'])
    parser.add_argument('--g',              type=int,
                                            default=10,
                                            help="Gain")
    parser.add_argument('--tau',            type=float,
                                            default=1,
                                            help="Rate-time constant")

    args = parser.parse_args()

    ############################################### SETUP ###############################################
    N           = args.N
    g           = args.g    
    tau         = args.tau    
    if args.phi == 'tanh':
        phi = np.tanh
    elif args.phi == 'ReLU':
        def phi(x):
            return x * (x > 0)
    
    # Note that in Engelken et al, parameters are as follows
        # delta_t = 0.1 \tau
        # t_ONS   = \tau
        # t_sim   = 10^3 \tau
    delta_t     = 0.1*tau
    t_transient = 100*tau
    t_sim       = 1000*tau

    # define J and remove self-coupling
    J = np.random.normal(loc = 0, scale = g**2, size = (N, N))     
    for i in range(len(J)):
        J[i][i] = 0
    h = np.random.randn(N)
    phi = jnp.tanh

    ######################################
    #              LEARNING              #
    #              FUNCTION              #
    ######################################
    """
    All learning functions should take in the same information
        h   := neural state
        phi := activation function
        J   := connectivity matrix
        tau := rate-time constant
        dt  := time discretization (delta_t)
    """
    def rnn(h : np.ndarray, phi: callable = phi, J : np.ndarray = J, tau : float = tau, dt : float = delta_t) -> np.ndarray:
        
        rnn 
        grad = -1*h*(1 - dt) + J @ phi(h)*dt
        grad /= tau
        return grad
    
    if args.fn  == 'single_sompo':
        fn  = rnn

    ########################################### SIMULATION ###########################################

    L = Lyapunov(f = fn, h_0 = h) # initialization
    L.evolve(t_sim = t_transient, delta_t = delta_t) # evolve to a point where the system is on a transient
    h_0 = L.h # save model parameters

    # Find s_ONS
    spec, k = L.get_spectra(m = N, s_ons = int(tau/delta_t), delta_t = delta_t)
    s_ons = int((np.log(k)/(spec[0] - spec[-1]))/delta_t) + 1

    # Run full simulation + plot results
    spec, _ = L.get_spectra(m = N, s_ons = s_ons, t_sim = t_sim, delta_t = delta_t)
    m       = len(spec)

    ###########################################$ PLOTTING ############################################

    fig, ax = plt.subplots()
    x_axis  = np.arange(m)/m
    plt.scatter(x_axis, spec) # plot Lyapunov exponents
    plt.plot(x_axis, np.zeros(m), 'k--') # plot the zero line

    plt.xlabel("i/N")
    plt.ylabel("Value of Lyapunov exponent")
    ax.set_title(f"Lyapunov spectra of {args.fn} system using {args.phi} neurons and g={args.g}")
    
    fig.savefig(f'figures/spectra_{args.fn}_{args.phi}_gain_{args.g}_tau_{args.tau}.png')
    plt.show(block = True)