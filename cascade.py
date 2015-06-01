"""
This module contains

- A wrapper class around qutip's Qobj class to allow for some tensor
  manipulations. Mostly to be able to perform generalized partial trace.
- Functions to integrate the master equation for k cascaded identical systems.

"""

import numpy as np
import scipy as sp

import qutip as qt


# Flag to decide if we are going to use cython for fast computation of
# tensor network generalized partial trace.
usecython = True
if usecython:
    import tnintegrate_c


class TensorQobj(qt.Qobj):
    """
    A wrapper around qutip Qobj to be able to view it as a tensor.
    This class is meant for representing super-operators.
    Each index of the tensor is a "double index" of dimension
    2*dim(H), i.e., the dimension of L(H). This convention
    is chosen to make it easier to work with qutip Qobj's of
    "super" type, representing superoperators. For example, this
    is consistent with how qutip's reshuffle works.
    """

    @property
    def nsys(self):
        # number of subsystems for the underlying Hilbert space
        return len(self.reshuffle().dims[0])

    @property
    def rank(self):
        # rank of tensor
        # each index is really a "double index"
        return self.nsys*2

    @property
    def sysdim(self):
        # dim of H
        return self.dims[0][0][0]

    @property
    def superdim(self):
        # dim of L(H)
        return self.sysdim**2

    def __mul__(self, other):
        return TensorQobj(super(TensorQobj,self).__mul__(other))

    def __rmul__(self, other):
        return TensorQobj(super(TensorQobj,self).__rmul__(other))

    def reshuffle(self):
        return TensorQobj(qt.reshuffle(self))

    def getmatrixindex(self,indices):
        # returns matrix indices of T given tensor indices
        # each subsystem has dimension self.superdim (double index)
        # indices = [i1,j1,...,iM,jM]
        if not len(indices) == self.rank:
            raise ValueError("number of indices do not match rank of tensor")
        ii = 0
        jj = 0
        idx = list(indices)
        for l in range(len(indices)/2):
            j = idx.pop()
            i = idx.pop()
            ii += i*self.superdim**l
            jj += j*self.superdim**l
        return ii,jj

    def gettensorelement(self,indices):
        # return element given tensor indices
        return self.reshuffle()[self.getmatrixindex(indices)]

    def loop(self):
        # return T reduced by one subsystem by summing over 2 indices
        out = TensorQobj(dims=[[[self.sysdim]*(self.nsys-1)]*2]*2)
        idx = [0]*out.rank
        indices = []
        sumindices = []
        for cnt in range(out.superdim**(out.rank)):
            indices.append(list(idx))
            sumindices.append(list(idx[0:-1] + [0,0] + idx[-1:]))
            idx[0] += 1
            for i in range(len(idx)-1):
                if idx[i] > self.superdim-1:
                    idx[i] = 0
                    idx[i+1] += 1
        out2 = out.reshuffle()
        if usecython:
            indices = np.array(indices,dtype=np.int_)
            sumindices = np.array(sumindices,dtype=np.int_)
            indata = np.array(self.reshuffle().data.toarray(),dtype=np.complex_)
            data = tnintegrate_c.loop(indata,out2.shape,
                        indices,sumindices,self.superdim)
            out2 = TensorQobj(data,dims=out2.dims)
        else:
            for idx in indices:
                i,j = out.getmatrixindex(idx)
                for k in range(self.superdim):
                    sumidx = idx[0:-1] + [k,k] + idx[-1:]
                    out2.data[i,j] += self.gettensorelement(sumidx)
        return out2.reshuffle()


def localop(op,l,k,dim):
    """
    Create a local operator on the l'th system by tensoring
    with identity operators on all the other k-1 systems
    """
    h = op
    id = qt.qeye(dim)
    for i in range(1,l):
        h = qt.tensor(h,id)
    for i in range(l+1,k+1):
        h = qt.tensor(id,h)
    return h


def generator(k,H,L1,L2):
    """
    Create the generator for the cascaded chain of k system copies
    """
    # create bare operators
    id = qt.qeye(H.dims[0][0])
    Id = qt.spre(id)*qt.spost(id)
    Hlist = []
    L1list = []
    L2list = []
    for l in range(1,k+1):
        h = H
        l1 = L1
        l2 = L2
        for i in range(1,l):
            h = qt.tensor(h,id)
            l1 = qt.tensor(l1,id)
            l2 = qt.tensor(l2,id)
        for i in range(l+1,k+1):
            h = qt.tensor(id,h)
            l1 = qt.tensor(id,l1)
            l2 = qt.tensor(id,l2)
        Hlist.append(h)
        L1list.append(l1)
        L2list.append(l2)
    # create Lindbladian
    L = qt.Qobj()
    #H0 = 0.5*(Hlist[0]+eps*L2list[0]+np.conj(eps)*L2list[0].dag())
    H0 = 0.5*Hlist[0]
    L0 = L2list[0]
    #L0 = 0.*L2list[0]
    L += qt.liouvillian(H0,[L0])
    E0 = Id
    for l in range(k-1):
        E0 = qt.composite(Id,E0)
        Hl = 0.5*(Hlist[l]+Hlist[l+1]+1j*(L1list[l].dag()*L2list[l+1] 
                                          -L2list[l+1].dag()*L1list[l]))
        Ll = L1list[l] + L2list[l+1]
        L += qt.liouvillian(Hl,[Ll])
    Hk = 0.5*Hlist[k-1]
    Lk = L1list[k-1]
    L += qt.liouvillian(Hk,[Lk])
    E0.dims = L.dims
    # return generator, identity superop E0
    return L,E0


def integrate(L,E0,ti,tf,Lfunc=None,opt=qt.Options()):
    """
    Basic ode integrator
    """
    def _rhs(t,y,L):
        ym = y.reshape(L.shape)
        return (L*ym).flatten()
    def _rhs_td(t,y,L,Lfunc):
        # Lfunc is time-dependent part of L
        L = L + Lfunc(t).data
        ym = y.reshape(L.shape)
        return (L*ym).flatten()

    from qutip.superoperator import vec2mat
    if Lfunc is None:
        r = sp.integrate.ode(_rhs)
        r.set_f_params(L.data)
    else:
        r = sp.integrate.ode(_rhs_td)
        r.set_f_params(L.data,Lfunc)
    initial_vector = E0.data.toarray().flatten()
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, ti)
    r.integrate(tf)
    if not r.successful():
        raise Exception("ODE integration error: Try to increase "
                        "the allowed number of substeps by increasing "
                        "the nsteps parameter in the Options class.")

    return qt.Qobj(vec2mat(r.y)).trans()


def rhot(rho0,t,tau,H_S,L1,L2,Id,drivefunc=None,options=qt.Options()):
    """
    Compute rho(t)
    """
    k= int(t/tau)+1
    s = t-(k-1)*tau
    rhovec = qt.operator_to_vector(rho0)
    dim = H_S.dims[0][0]
    # collapseop for the first system in the cascade
    L2first = localop(L2,1,k,dim)
    if drivefunc is not None:
        Hdrive = lambda t: 1j*(sp.conj(drivefunc(t))*1.0*L2first
                               -drivefunc(t)*1.0*L2first.dag())
        Ldrive = lambda t: -1j*(qt.spre(Hdrive(t))-qt.spost(Hdrive(t)))
    else:
        Ldrive = None
    G1,E0 = generator(k,H_S,L1,L2)
    E = integrate(G1,E0,0.,s,Lfunc=Ldrive,opt=options)
    if k>1:
        G2,null = generator(k-1,H_S,L1,L2)
        G2 = qt.composite(Id,G2)
        E = integrate(G2,E,s,tau,Lfunc=Ldrive,opt=options)
    E.dims = E0.dims
    E = TensorQobj(E)
    for l in range(k-1):
        E = E.loop()
    sol = qt.vector_to_operator(E*rhovec)
    return sol


