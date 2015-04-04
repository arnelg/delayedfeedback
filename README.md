This code was used to generate the results in
arXiv:1502.06959 - Time-delayed quantum feedback control.

Requirement: QuTiP and all it's requirements (see qutip.org).

Explanation of files in repository:

example.py - example file for running a simulation of a two-level atom
             coupled to a feedback loop.
cascade.py - main module that builds the generator for the cascaded
             master equation, and integrates it.
tnintegrate_c.pyx - some cython functions for greater speed

Example usage:

In python prompt:

    from qutip import *
    from pylab import *
    import example
    times,sol = example.run()
    plot(times,expect(sol,sigmap()*sigmam())
    show()
