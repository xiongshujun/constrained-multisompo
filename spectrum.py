import os
import os.path      as op

import numpy     as np
import jax.numpy as jnp
import pickle

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from datetime   import datetime

from utils import Lyapunov, entropy, dim

def get_spectra(fn : str = "single_sompo", N : int = 100, phi : str = "tanh", g : float = 10, tau : float = 1, exp : str = None):

    """
    Given a set of parameters, find the Lyapunov spectrum of the system, its entropy, and its attractor dimensionality.        

    INPUTS
        fn  := choice of which evolution rule the network as a whole will use. Engelken et al use a simple Sompolinsky model which we will build upon.
                Refer to "Learning Models" section for details on how each of these evolution rules are parameterized
        N   := size of the network, which can be vaied across experiments
        phi := activation function of each neuron (default is tanh(x) as in the Engelken et al paper, which also later tests out ReLU(x))
                    !TODO: the original Sompolinsky Crisanti and Summers paper uses tanh(gx), see if this changes anything
        g   := gain value.
                    !TODO: see how to vary gain between regions
        tau := rate-time constant
        exp := name of a custom experiment to override the default expanme construction

    OTHER PARAMETERS
        Note that in Engelken et al, algorithm parameters are as follows
            delta_t = 0.1 \tau
            t_ONS   = \tau
            t_sim   = 10^3 \tau
            Since dt = delta_t is used in controlling how much h_{t + 1} evolves from h_t (a value between 0 and 1), tau must be between 0 and 10
        J := connectivity matrix drawn from a Gaussian of mean 0 and variance = g^2

    OUTPUTS
        spec    := Lyapunov spectrum
        h       := Kolmogorov-Sinai entropy
        d       := Kaplan-Yorke attractor dimensionality

        Saves the spectrum as a figure and pickles the data collected (spectrum, fn, g, tau, H, D, N)
    """
    assert fn in ['single_sompo', 'chaudhuri', 'multi_sompo', 'multi_sompo_low_rank'], "Not a valid evolution rule!"
    assert phi in ['tanh', 'ReLU'], "Not a valid activation function!"

    cur_dir  = os.path.dirname(__file__)

    if exp is None:
        expname  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        expname += f"_{fn}_{phi}_gain_{str(g)}_tau_{str(tau)}"
    else:
        expname  = exp + f"_{fn}_{phi}_gain_{str(g)}_tau_{str(tau)}"

    ############################################### SETUP ###############################################
    if phi == 'tanh':
        phi = np.tanh
    elif phi == 'ReLU':
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
    
    if fn  == 'single_sompo':
        fn  = rnn

    ######################################
    #             SIMULATION             #
    ######################################
    """
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
    """
    L = Lyapunov(f = fn, h_0 = h) # initialization
    L.evolve(t_sim = t_transient, delta_t = delta_t) # evolve to a point where the system is on a transient
    h_0 = L.h # save model parameters

    # Find s_ONS
    spec, k = L.get_spectra(m = N, s_ons = int(tau/delta_t), delta_t = delta_t)
    s_ons = int((np.log(k)/(spec[0] - spec[-1]))/delta_t) + 1

    # Run full simulation to get spectra
    spec, _ = L.get_spectra(m = N, s_ons = s_ons, t_sim = t_sim, delta_t = delta_t)
    m       = len(spec)     # size of the spectrum we get
    h       = entropy(spec) # entropy
    d       = dim(spec)     # attractor dimensionality

    # Save metadata for collation and use in other plots
    data = {'spec'      : spec,
            'fn'         : fn,
            'gain'      : g,
            'tau'       : tau,
            'entropy'   : h,
            'dimension' : d,
            'N'         : N}
    if exp is None:
        filename = op.join(cur_dir, 'data/', f'metadata_{expname}.pickle')
        os.makedirs(op.dirname(filename), exist_ok=True)
    else:
        filename = op.join(cur_dir, f'data/{exp}', f'metadata_{expname}.pickle')
        os.makedirs(op.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)

    ######################################
    #              PLOTTING              #
    ######################################
    """
    Plots spectrum as in Engelken et al 2023
        Lyapunov exponent value against i/N (normalized eigenvector number)
        Dotted line along y = 0
    Figures saved to 'figures/' folder, titled with model type, activation type, gain, and tau parameters]
    """

    fig, ax = plt.subplots()
    x_axis  = np.arange(m)/m
    plt.scatter(x_axis, spec) # plot Lyapunov exponents
    plt.plot(x_axis, np.zeros(m), 'k--') # plot the zero line

    plt.xlabel("i/N")
    plt.ylabel("Value of Lyapunov exponent")
    ax.set_title(f"Lyapunov spectra of {expname}")

    # Save and show figure
    if exp is None: 
        fig.savefig(op.join(cur_dir, 'figures/', f'spectra_{expname}.png'), dpi = 300)
    else:
        filedir = op.join(cur_dir, f'data/{exp}')
        os.makedirs(op.dirname(filedir), exist_ok=True)
        fig.savefig(op.join(cur_dir, filedir, f'spectra_{expname}.png'), dpi = 300)
    plt.show(block = True)

    return spec, h, d

############################################
#              MULTITHREADING              #
#              SINGLE RUNTIME              #
############################################
# Figure 2 of Engelken et al
for N in range(0, 10000, 100):
    get_spectra(N = N, exp = 'Fig2')

# Figure 3 of Engelken et al
for gain in range(0, 10, 0.1):
    get_spectra(g = gain, exp = 'Fig3')

# Figure 4 of Engelken et al
for gain in range(0, 1000, 10):
    get_spectra(g = gain, exp = 'Fig4')