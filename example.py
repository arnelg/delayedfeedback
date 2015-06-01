"""
Example file for time-delayed feedback simulation.
This example is a driven two-level atom, the example used in
arXiv:1502.06959.

Example usage:

In python prompt:

>>>> from qutip import *
>>>> from pylab import *
>>>> import example
>>>> times,sol = example.run()
>>>> plot(t,expect(sol,sigmap()*sigmam())
>>>> show()

"""

import numpy as np
import scipy as sp

import qutip as qt

import cascade

gamma = 1.0 # coupling strength to reservoir
phi = 1.*np.pi # phase shift in fb loop
eps = 2.0*np.pi*gamma # eps/2 = Rabi frequency
delta = 0. # detuning

# time delay
tau = np.pi/(eps)
print 'tau =',tau

dim_S = 2
Id = qt.spre(qt.qeye(dim_S))*qt.spost(qt.qeye(dim_S))

# Hamiltonian and jump operators
H_S = delta*qt.sigmap()*qt.sigmam() + eps*(qt.sigmam()+qt.sigmap())
L1 = sp.sqrt(gamma)*qt.sigmam()
L2 = sp.exp(1j*phi)*L1

# initial state
rho0 = qt.ket2dm(qt.basis(2,0))

# times to evaluate rho(t)
tlist=np.arange(0.0001,2*tau,0.01)

def run(rho0=rho0,tau=tau,tlist=tlist):
    # run feedback simulation
    opts = qt.Options()
    opts.nsteps = 1e7
    sol = np.array([rho0]*len(tlist))
    for i,t in enumerate(tlist):
        sol[i] = cascade.rhot(rho0,t,tau,H_S,L1,L2,Id,options=opts)
    return tlist,sol

def run_nofb(rho0=rho0,tlist=tlist):
    # run simulation without feedback
    sol = qt.mesolve(H_S,rho0,tlist,[L1,L2],[])
    return sol.times,sol.states

if __name__=='__main__':
    run(rho0=rho0)

