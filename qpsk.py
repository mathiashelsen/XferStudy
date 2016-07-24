import numpy as np
import scipy.signal 
import matplotlib.pyplot as plt

# Number of periods to calculate
N = 4096
# Oversampling ratio
Nos = 64
# Symbol rate relative to carrier
symbolRate = 0.16
# The bandwidth off the BP filter rel to symbol rate
relBW = 0.52
# Number of phases
Nphases = 8

# The temporal axis
t = np.linspace(0.0, N, num=N*Nos)

# The symbols
symbols = (2.0*np.pi/Nphases)*(np.random.random_integers(0, Nphases-1, symbolRate*N+1))

# The actual waveform
x = [np.sin(2.0*np.pi*t[i] + symbols[(i/Nos)*symbolRate]) for i in range(N*Nos)]
# Add some noise for good measure
x = x + np.random.normal(loc=0.0, scale=1.0, size=len(x))

# Internal constants for decimation and filtering
LPorder = (Nos*2)
q = int(round(Nos/symbolRate))
offset = q/2

# Calculate in-phase and quadrature reponse, apply low pass filter
I = np.sin(2.0*np.pi*t)*x
I = np.convolve(I, np.ones((LPorder,))/LPorder, mode='same')
Q = np.cos(2.0*np.pi*t)*x
Q = np.convolve(Q, np.ones((LPorder,))/LPorder, mode='same')

# Save data
concatData = np.zeros( (len(x), 6) )
concatData[:,0] = x
concatData[:,1] = [ symbols[(i/Nos)*symbolRate] for i in range(N*Nos) ]
concatData[:,2] = I
concatData[:,3] = Q
np.savetxt('raw_wfrm.dat', concatData, delimiter='\t')

# Decimate IQ response
newT = t[offset::q]
concatData = np.zeros( (len(newT), 3) )
concatData[:,0] = newT
concatData[:,1] = I[offset::q]
concatData[:,2] = Q[offset::q]
np.savetxt('dwn_smpld_iq.dat', concatData, delimiter='\t')

# Calculate the FFT
x_hat = np.fft.rfft(x)
power = 20.0*np.log10(np.absolute(x_hat))
freq = np.fft.rfftfreq(len(x), t[1]-t[0])
print freq[0], freq[1]-freq[0]

# Calculate where the carrier is found and the admissible BW 
indexCenter = int(round(1.0/(freq[1]-freq[0])))
widthSideband = int(round(relBW*symbolRate/(freq[1]-freq[0])))
lo = indexCenter - widthSideband
up = indexCenter + widthSideband

# Use only the portion of the spectrum allowed by the brickwall filter
x_hat_prime = np.zeros(len(x_hat)) + 1j*np.ones(len(x_hat))*1.0e-6
x_hat_prime[lo:up] = x_hat[lo:up]
power_prime = 20.0*np.log10(np.absolute(x_hat_prime))

# Inverse FFT
x_prime = np.fft.irfft(x_hat_prime)
# IQ calculation
I_prime = np.sin(2.0*np.pi*t)*x_prime
I_prime = np.convolve(I_prime, np.ones((LPorder,))/LPorder, mode='same')
Q_prime = np.cos(2.0*np.pi*t)*x_prime
Q_prime = np.convolve(Q_prime, np.ones((LPorder,))/LPorder, mode='same')

# Save IQ 
concatData = np.zeros( (len(x_prime), 6) )
concatData[:,0] = x_prime
concatData[:,1] = [ symbols[(i/Nos)*symbolRate] for i in range(N*Nos) ]
concatData[:,2] = I_prime
concatData[:,3] = Q_prime

np.savetxt('prime_wfrm.dat', concatData, delimiter='\t')

# Decimate and save IQ data
concatData = np.zeros( (len(newT), 3) )
concatData[:,0] = newT
concatData[:,1] = I_prime[offset::q]
concatData[:,2] = Q_prime[offset::q]
np.savetxt('dwn_smpld_iq_prime.dat', concatData, delimiter='\t')

# Save original spectrum and cropped spectrum
concatData = np.zeros( (len(power), 3) )
concatData[:,0] = freq
concatData[:,1] = power
concatData[:,2] = power_prime
np.savetxt('spectrum.dat', concatData, delimiter='\t')

