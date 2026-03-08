from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

import dpd_lib as dpd

def plot_amam(*argv, nfig = -1, clfig = True):
    
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()
        
    plt.xlabel("n")
    plt.ylabel("|x(n)|")
    plt.title("ABS")
    plt.grid(True)
    
    if(np.mod(len(argv),2)):
        print("Attention! plot_abs function number of arguments must be EVEN!")
        return 
    for n in range(int(len(argv)/2)):
        plt.scatter(np.abs(argv[2*n]), np.abs(argv[2*n+1]), s = 0.1)
    plt.show
    return;
    
    


def plot_abs(*argv, nfig = -1, clfig = True):
    
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()
        
    plt.xlabel("n")
    plt.ylabel("|x(n)|")
    plt.title("ABS plot")
    plt.grid(True)
    
    for n in range(len(argv)):
        plt.plot(np.abs(argv[n]))
    plt.show
    return;

    
    
    
def plot_ampm(*argv, nfig = -1, clfig = True):
    
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()
        
    plt.xlabel("|x|")
    plt.ylabel("angle(x,y)")
    plt.title("AM-PM")
    plt.grid(True)
    
    if(np.mod(len(argv),2)):
        print("Attention! plot_ampm function number of arguments must be EVEN!")
        return 
    for n in range(int(len(argv)/2)):
        phi = 180.0 * np.angle(np.conj(argv[2*n+1])*argv[2*n]) / np.pi    
        plt.scatter(np.abs(argv[2*n]), phi, s = 0.1)
        
    plt.show
    return;





def plot_psd(*argv, nfft = 2048, Fs = 1.0, nfig = -1, clfig = True):
    
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()

    for z in argv:
        if(z.ndim > 1):
            frq, psd = signal.welch(z[:,0], fs = Fs,  window = 'blackman', nperseg = nfft, return_onesided = False, detrend = False)
        else:
            frq, psd = signal.welch(z, fs = Fs, window = 'blackman', nperseg = nfft, return_onesided = False, detrend = False)    
        
        frq = np.fft.fftshift(frq)
        psd = 10.0*np.log10(np.fft.fftshift(psd))
        plt.plot(frq, psd, lw = 1.0)
    
    plt.xlabel("frequency")
    plt.ylabel("Magnitude, dB")
    plt.title("Power spectral density")
    plt.grid(True)
    plt.show
    return
    

def plot_psd_maxhold(*argv, nfft = 2048, nseg = 16384, Fs = 1.0, nfig = -1, clfig = True):
    
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()

    for z in argv:
        if(z.ndim > 1):
            q = z[:,0]            
        else:
            q = z
        cnt = 0
        psd = np.zeros((nfft,1)) - 10000
        while cnt + nseg < len(q):
           seg = q[cnt:cnt+nseg] 
           frq, psd_cur = signal.welch(seg, fs = Fs, window = 'blackman', nperseg = nfft, return_onesided = False, detrend = False)    
           psd_cur = 10.0*np.log10(np.fft.fftshift(psd_cur))
           for i in range(nfft):
               if(psd_cur[i]>psd[i]):
                   psd[i] = psd_cur[i]
           cnt = cnt + nseg
                   
           
        
        frq = np.fft.fftshift(frq) 
        plt.plot(frq, psd, lw = 1.0)
    
    plt.xlabel("frequency")
    plt.ylabel("maxhold Magnitude, dB")
    plt.title("maxhold Power spectral density")
    plt.grid(True)
    plt.show
    return


def plot_nmse(x, *argv, nseg = 1024, nfig = -1, clfig = True):
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()
        
    plt.xlabel("segment number")
    plt.ylabel("NMSE, dB")
    plt.title("NMSE in time")
    plt.grid(True)
    
    for n in range(len(argv)):
        err = argv[n]
        nmse = np.zeros((int(len(x)/nseg)))
        cnt = 0
        i = 0
        while cnt + nseg < len(err):
            nmse[i] = dpd.nmse(x[cnt:cnt+nseg], err[cnt:cnt+nseg]) 
            i = i+1
            cnt = cnt + nseg
        plt.plot(nmse)
    plt.show
    return;


def plot_re(*argv, nfig = -1, clfig = True):
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()
        
    plt.xlabel("|x|")
    plt.ylabel("|y|")
    plt.title("Real part plot")
    plt.grid(True)
    
    for n in range(len(argv)):
        plt.plot(np.real(argv[n]))
    plt.show
    return; 
    
    


def plot_im(*argv, nfig = -1, clfig = True):
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()
        
    plt.xlabel("|x|")
    plt.ylabel("|y|")
    plt.title("Image part plot")
    plt.grid(True)
    
    for n in range(len(argv)):
        plt.plot(np.imag(argv[n]))
    plt.show
    return;
 

def plot_reim(*argv, nfig = -1, clfig = True):
    if (nfig > 0):
        plt.figure(nfig)
    else:
        plt.figure()    
    
    if(clfig):
        plt.clf()
        
    plt.xlabel("|x|")
    plt.ylabel("|y|")
    plt.title("Image part plot")
    plt.grid(True)
    
    for n in range(len(argv)):
        plt.plot(np.real(argv[n]))
        plt.plot(np.imag(argv[n]), '--')
    plt.show
    return;
    