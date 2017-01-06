# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 10:39:38 2017

@author: mhel
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

def stairCaseFunction(t, T):
    retVal = 0.0
    i = np.floor(t/T)
    u = (t-i*T)/T
    if(u < 0.33):
        retVal = -1.0
    elif(u < 0.66):
        retVal = 0.0
    else:
        retVal = 1.0
    return retVal    

n = 1024
t = np.linspace(0.0, 1.0, num=n)
fSine = 10.3
TSine = 1.0/fSine
nBits = 2
quantStep = 1.0/(2**nBits)
fADC = 1.0/n
fDAC = 10

window = np.hanning(n)

sineSig = np.sin(2.0*math.pi*fSine*t)
stairSig = np.array([stairCaseFunction(t[i], TSine) for i in range(len(t))])*0.76315248

signal = stairSig

truncSig = np.zeros(len(signal))
value = np.round(signal[0]/quantStep)*quantStep
for i in range(n):
    if i%fDAC == 0:
        value = np.round(signal[i]/quantStep)*quantStep
    truncSig[i] = value

truncSig = truncSig*window
signal = signal*window

concatData = np.zeros((n, 3))
concatData[:,0] = t
concatData[:,1] = signal
concatData[:,2] = truncSig

np.savetxt('timeDomain.dat', concatData, delimiter='\t')

spectrSig = 40.0*np.log10(2.0*np.absolute(np.fft.rfft(signal))/n)
spectrTrunc = 40.0*np.log10(2.0*np.absolute(np.fft.rfft(truncSig))/n)

f = np.fft.rfftfreq(n, 1.0/n)

concatData = np.zeros((len(f),6))
concatData[:,0] = f
concatData[:,1] = spectrSig
concatData[:,2] = spectrTrunc
concatData[:,3] = np.ones(len(f))*40.0*np.log10(quantStep*quantStep/12.0)
concatData[:,4] = np.ones(len(f))*40.0*np.log10(1.0*1.0/2.0)


np.savetxt('freqDomain.dat', concatData, delimiter='\t')