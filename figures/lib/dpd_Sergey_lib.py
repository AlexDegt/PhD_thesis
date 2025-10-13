# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as signal

''' Constants '''
NONZERO_NUM_1DLUT = 2
NONZERO_NUM_2DLUT = 4

def nmse(x, e):
    y = 10.0*np.log10(np.real(np.sum(e*np.conj(e)) / np.sum(x*np.conj(x))))
    return y


def aclr (x, err, fx, fe, b, Fs, hord = 512):
    w0 = b*0.5 / Fs
    h = signal.firwin(hord+1, w0)
    t = np.linspace(0, hord, hord+1) - hord/2.0
    hx = h * np.exp(1j * 2 * np.pi * fx / Fs * t)
    he = h * np.exp(1j * 2 * np.pi * fe / Fs * t)
    
    xf = signal.lfilter(hx, 1, x)
    ef = signal.lfilter(he, 1, err)
    return nmse(xf, ef)
   
    
def delay(x, d):
    s = np.copy(x) * 0.0

    if(d>0):
        s[d:len(x)] = x[0:len(x)-d]
    else:
        s[0:len(x)+d]  = x[-d:len(s)]
    return s


def cheby_polyu(x, order):
    u = np.zeros((order+1), dtype = 'complex128')
    u[0] = 1.0
    if (order > 0):
        u[1] =  x
        p = 2
        while(p<=order):
            u[p] = 2.0 * x * u[p-1] - u[p-2]
            p = p + 1
    return u

def cheby_polyu_rescaled(x, order, alpha, beta):
    """ Rescaled Chebyshev polynomial """
    assert beta > alpha
    normalization = cheby_polyu((beta+alpha)/(beta-alpha), order)
    return cheby_polyu((beta+alpha-2*x)/(beta-alpha), order)/normalization

def cheby_polyu2(x, pord):
    u = np.zeros((2*pord+1), dtype = 'complex128')
    u[0] = 1.0
    if (pord > 0):
        u[1] =  x
        p = 2
        while(p<=2*pord):
            u[p] = 2.0 * x * u[p-1] - u[p-2]
            p = p + 1
    return u


def vand1d_cheby(xa, model, pord, mlin):
    NLUT = np.size(model,1)
    a = 0.9 * abs(xa) / max(abs(xa))
    
    ncoeff = NLUT * (pord+1) + mlin
    N = len(xa)
    V = np.zeros((N, ncoeff), dtype='complex128')
    

    MD = int(np.max(np.abs(model[2:6][:])))
    
       
    dlin = int(mlin / 2)
    if dlin > MD:
        MD = dlin

    for n in range(MD,N-MD):
        index  = 0
        for m in range(NLUT):
            
            # 1D LUT description
            if model[1][m] == 1:
                
                
                # no term lut type == 0
                if model[0][m] == 0:
                    ds = int(model[3][m])
                    term = 1.0
                    #
                
                # sl lut type == 1    
                elif model[0][m] == 1:                    
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    term = xa[n-dl]
                
                # sil lut type == 2 
                elif model[0][m] == 2:                    
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    du = int(model[4][m])
                    dv = int(model[5][m])
                    term = xa[n-dl] * xa[n-du] * np.conjugate(xa[n-dv])

                    
                # abs LUT type== 4    
                elif model[0][m] == 4:
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    du = int(model[4][m])
                    term = xa[n-dl] * np.abs(xa[n-du])
                

                else:
                    print('vand2d function error! LUT number %d type is wrong!' % m)
                    return
                V[n, index:index+pord+1]   = term * cheby_polyu(a[n - ds], pord)
                index = index + pord+1
            
          
            else:
                print('vand2d function error! LUT number %d dimension is wrong!' % m)
                return
        for m in range(mlin):
            V[n, index+m]   = xa[n - m - dlin]

    return V





def vand1d_cheby2(xa, model, pord, mlin):
    NLUT = np.size(model,1)
    a = 0.9 * abs(xa) / max(abs(xa))
    
    ncoeff = NLUT * (pord+1) + mlin
    N = len(xa)
    V = np.zeros((N, ncoeff), dtype='complex128')
    

    MD = int(np.max(np.abs(model[2:6][:])))
    
       
    dlin = int(mlin / 2)
    if dlin > MD:
        MD = dlin

    for n in range(MD,N-MD):
        index  = 0
        for m in range(NLUT):
            
            # 1D LUT description
            if model[1][m] == 1:
                
                
                # no term lut type == 0
                if model[0][m] == 0:
                    ds = int(model[3][m])
                    term = 1.0
                    #
                
                # sl lut type == 1    
                elif model[0][m] == 1:                    
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    term = xa[n-dl]
                
                # sil lut type == 2 
                elif model[0][m] == 2:                    
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    du = int(model[4][m])
                    dv = int(model[5][m])
                    term = xa[n-dl] * xa[n-du] * np.conjugate(xa[n-dv])

                    
                # abs LUT type== 4    
                elif model[0][m] == 4:
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    du = int(model[4][m])
                    term = xa[n-dl] * np.abs(xa[n-du])
                

                else:
                    print('vand2d function error! LUT number %d type is wrong!' % m)
                    return
                V[n, index:index+pord+1]   = term * cheby_polyu2(a[n - ds], pord)
                index = index + pord+1
            
          
            else:
                print('vand2d function error! LUT number %d dimension is wrong!' % m)
                return
        for m in range(mlin):
            V[n, index+m]   = xa[n - m - dlin]

    return V




def vand2d(xa, xb, model, nspln, mlin, bits_interp = -1):
    
    NLUT = np.size(model,1)
    
    ncoeff = 0
    
    for m in range(NLUT):
        if model[1][m] == 1:
            ncoeff = ncoeff + nspln
        elif model[1][m] == 2:
            ncoeff = ncoeff + nspln*nspln
        else:
            print('vand2d function error! LUT number %d dimension is wrong!' % m)
            return 
    
    if mlin < 0:
        print('vand2d function error! mlin paramter must be a positive!')
    else:
        ncoeff = ncoeff + mlin
    
    axa   = np.abs(xa) * nspln
    addra = np.floor(axa)
    dxa   = axa - addra   

    axb   = np.abs(xb) * nspln
    addrb = np.floor(axb)
    dxb  = axb - addrb   
    
    if(bits_interp > 0):
        dxa = np.floor(dxa * 2.0**bits_interp) / 2.0**bits_interp
        dxb = np.floor(dxb * 2.0**bits_interp) / 2.0**bits_interp
    
    N = len(xa)
    V = np.zeros((N, ncoeff), dtype='complex128')
    

    MD = int(np.max(np.abs(model[2:6][:])))
    
       
    dlin = int(mlin / 2)
    if dlin > MD:
        MD = dlin

     
    for n in range(MD,N-MD):
        index  = 0
        for m in range(NLUT):
            
            # 1D LUT description
            if model[1][m] == 1:
                
                # no term lut type == 0
                if model[0][m] == 0:
                    ds = int(model[3][m])
                    a  = int(addra[n - ds])
                    V[n, index+a]   = (1.0 - dxa[n - ds])
                    V[n, index+a+1] = dxa[n-ds]
                
                # sl lut type == 1    
                elif model[0][m] == 1:                    
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    a  = int(addra[n - ds])
                    term = xa[n-dl]
                    V[n, index+a]   = term * (1.0 - dxa[n - ds])
                    V[n, index+a+1] = term * dxa[n-ds]
                
                # sil lut type == 2 
                elif model[0][m] == 2:                    
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    du = int(model[4][m])
                    dv = int(model[5][m])
                    term = xa[n-dl] * xa[n-du] * np.conjugate(xa[n-dv])
                    a  = int(addra[n - ds])
                    V[n, index+a]   = term * (1.0 - dxa[n - ds])
                    V[n, index+a+1] = term * dxa[n-ds]
                    
                # abs LUT type== 4    
                elif model[0][m] == 4:
                    dl = int(model[2][m])
                    ds = int(model[3][m])
                    du = int(model[4][m])
                    term = xa[n-dl] * np.abs(xa[n-du])
                    a  = int(addra[n - ds])
                    V[n, index+a]   = term * (1.0 - dxa[n - ds])
                    V[n, index+a+1] = term * dxa[n-ds]
                else:
                    print('vand2d function error! LUT number %d type is wrong!' % m)
                    return
                index = index + nspln
            
            # 2D LUT description
            elif model[1][m] == 2:
                ds = int(model[3][m])
                aa  = int(addra[n - ds])
                ab  = int(addrb[n - ds])
                a00 = index + ab* nspln + aa
                a01 = a00 + 1
                a10 = a00 + nspln
                a11 = a01 + nspln
                
                
                 # no term lut type == 0
                if model[0][m] == 0:                 
                    V[n, a00]   = (1.0 - dxa[n - ds]) * (1.0 - dxb[n - ds])
                    V[n, a01]   = (      dxa[n - ds]) * (1.0 - dxb[n - ds])
                    V[n, a10]   = (1.0 - dxa[n - ds]) * (      dxb[n - ds])
                    V[n, a11]   = (      dxa[n - ds]) * (      dxb[n - ds])
                
                # sl lut type == 1    
                elif model[0][m] == 1:                    
                    dl = int(model[2][m])
                    term = xa[n-dl]
                    V[n, a00]   = (1.0 - dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a01]   = (      dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a10]   = (1.0 - dxa[n - ds]) * (      dxb[n - ds]) * term
                    V[n, a11]   = (      dxa[n - ds]) * (      dxb[n - ds]) * term
                
                # sil lut type == 2 
                elif model[0][m] == 2:                    
                    dl = int(model[2][m])
                    du = int(model[4][m])
                    dv = int(model[5][m])
                    term = xa[n-dl] * xa[n-du] * np.conjugate(xa[n-dv])
                    
                    V[n, a00]   = (1.0 - dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a01]   = (      dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a10]   = (1.0 - dxa[n - ds]) * (      dxb[n - ds]) * term
                    V[n, a11]   = (      dxa[n - ds]) * (      dxb[n - ds]) * term
                    
                # sil lut type == 3 
                elif model[0][m] == 3:                    
                    dl = int(model[2][m])
                    du = int(model[4][m])
                    dv = int(model[5][m])
                    
                    term = xa[n-dl] * xb[n-du] * np.conjugate(xb[n-dv])
                    
                    V[n, a00]   = (1.0 - dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a01]   = (      dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a10]   = (1.0 - dxa[n - ds]) * (      dxb[n - ds]) * term
                    V[n, a11]   = (      dxa[n - ds]) * (      dxb[n - ds]) * term
                    
                # abs LUT type== 4    
                elif model[0][m] == 4:
                    dl = int(model[2][m])
                    du = int(model[4][m])
                    term = xa[n-dl] * np.abs(xa[n-du])
                    
                    V[n, a00]   = (1.0 - dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a01]   = (      dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a10]   = (1.0 - dxa[n - ds]) * (      dxb[n - ds]) * term
                    V[n, a11]   = (      dxa[n - ds]) * (      dxb[n - ds]) * term
                    
                # abs LUT type== 5    
                elif model[0][m] == 5:
                    dl = int(model[2][m])
                    du = int(model[4][m])
                    term = xa[n-dl] * np.abs(xb[n-du])
                    
                    V[n, a00]   = (1.0 - dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a01]   = (      dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a10]   = (1.0 - dxa[n - ds]) * (      dxb[n - ds]) * term
                    V[n, a11]   = (      dxa[n - ds]) * (      dxb[n - ds]) * term
                
                # abs LUT type== 6 
                elif model[0][m] == 6:
                    term = xa[n]**2
                    
                    V[n, a00]   = (1.0 - dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a01]   = (      dxa[n - ds]) * (1.0 - dxb[n - ds]) * term
                    V[n, a10]   = (1.0 - dxa[n - ds]) * (      dxb[n - ds]) * term
                    V[n, a11]   = (      dxa[n - ds]) * (      dxb[n - ds]) * term
                    
                else:
                    print('vand2d function error! LUT number %d type is wrong!' % m)
                    return
              
                index = index + nspln*nspln
            else:
                print('vand2d function error! LUT number %d dimension is wrong!' % m)
                return
        for m in range(mlin):
            V[n, index+m]   = xa[n - m - dlin]

    return V

def lut_ind_vec(xa, xb, model, nspln):
    """
        Calculates non-zero elements indicies
        within curren LUT statement vector.
        Works with 2D LUT
        xa, xb - input variables (NOT arrays)
    """
    axa   = np.abs(xa) * nspln
    addra = np.floor(axa)  

    axb   = np.abs(xb) * nspln
    addrb = np.floor(axb)
    # 1D LUT description
    if model[1][0] == 1:
        a  = int(addra)
        return a
     # 2D LUT description
    elif model[1][0] == 2:
        aa  = int(addra)
        ab  = int(addrb)
        a00 = ab* nspln + aa
        a01 = a00 + 1
        a10 = a00 + nspln
        a11 = a01 + nspln
        return np.array([a00, a01, a10, a11])


def vand_lin(x, mlin):
    dlin = int(mlin / 2)
    N = len(x)
    V = np.zeros((N, mlin), dtype='complex128')
    for n in range(dlin,N-dlin):
        for m in range(mlin):
            V[n, m]   = x[n - m + dlin]
    return V
    


def vand2d_delay2(x, model, nspln, mlin, bits_interp = -1):
    NLUT = np.size(model,1)
    V= []
    L = np.zeros((6, 1), dtype='int')
    for m in range(NLUT):
        L[0][0] = model[0][m]
        L[1][0] = model[1][m]
        L[2][0] = model[2][m]
        L[3][0] = model[3][m]
        L[4][0] = model[4][m]
        L[5][0] = model[5][m]
        
        d = L[4][0]
        
        y = delay(x, d)
        v0 = vand2d(x, y, L, nspln, 0, bits_interp)
        if np.size(V)==0:
            V = v0
        else:
            V = np.hstack((V, v0))
    v0 = vand_lin(x, mlin)
    V = np.hstack((V, v0))
    return V
      

def vand2d_delay3(x, r, model, nspln, mlin, bits_interp = -1):
    NLUT = np.size(model,1)
    V= []
    L = np.zeros((6, 1), dtype='int')
    for m in range(NLUT):
        L[0][0] = model[0][m]
        L[1][0] = model[1][m]
        L[2][0] = model[2][m]
        L[3][0] = model[3][m]
        L[4][0] = model[4][m]
        L[5][0] = model[5][m]
        
        d = L[4][0]
        
        y = delay(r, d)
        v0 = vand2d(x, y, L, nspln, 0, bits_interp)
        if np.size(V)==0:
            V = v0
        else:
            V = np.hstack((V, v0))
    v0 = vand_lin(x, mlin)
    V = np.hstack((V, v0))
    return V


                