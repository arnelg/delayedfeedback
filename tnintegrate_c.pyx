import numpy as np
cimport numpy as np


def getmatrixindex(np.ndarray[np.int_t] idx, superdim):
    # returns matrix indices given tensor indices
    # each subsystem has dimension superdim (double index)
    # idx = [i1,j1,...,iM,jM]
    cdef Py_ssize_t r,l
    cdef int ii = 0
    cdef int jj = 0
    cdef int i,j
    cdef int d = superdim
    r = len(idx)
    for l in range(r/2):
        j = idx[(r-1)-l*2]
        i = idx[(r-1)-(l*2+1)]
        ii += i*d**l
        jj += j*d**l
    return ii,jj


def loop(np.ndarray[np.complex_t,ndim=2] T,
         outshape,
         np.ndarray[np.int_t,ndim=2] indices,
         np.ndarray[np.int_t,ndim=2] sumindices,
         superdim):
    # return T reduced by one subsystem by summing over 4 indices
    cdef Py_ssize_t l,i,j
    cdef int k
    cdef int n = len(indices)
    cdef int d = superdim
    cdef int r = len(indices[0,:])
    data = np.zeros(outshape,dtype=np.complex_)
    for l in range(n):
        idx = indices[l,:]
        sumidx = sumindices[l,:]
        i,j = getmatrixindex(idx,d)
        for k in range(d):
            sumidx[(r+2)-2] = k
            sumidx[(r+2)-3] = k
            data[i,j] += T[getmatrixindex(sumidx,d)]
    return data
