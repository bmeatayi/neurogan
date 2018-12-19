import numpy as np
from scipy.stats import norm
import scipy.signal as signal
import numpy.matlib
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

nT = 30 # Temporal filter length
nX = 40 # Spatial filter length

N = 30 # Number of neurons

temp_filt = np.zeros((N, nT))
spat_filt = np.zeros((N, nX))

# Define filters manually
for i in range(N-15):
    temp_filt[i,:] = norm.pdf(np.linspace(-15,15,nT), loc=i-12.5+i/1.5, scale=.8)
    spat_filt[i,:] = norm.pdf(np.linspace(-10,10,nX), i-7.5, scale=.8) - .8 * norm.pdf(np.linspace(-10,10,nX), i-7.5, 1)

for i in range(N-15, N):
    temp_filt[i,:] = norm.pdf(np.linspace(-5,5,nT), loc=.45*i-10, scale=.8) - .6*norm.pdf(np.linspace(-5,5,nT), loc=.45*i-9, scale=.8)
    spat_filt[i,:] = norm.pdf(np.linspace(-15,15,nX), loc=1.5*i-33, scale=.7) - .5*norm.pdf(np.linspace(-15,15,nX), loc=1.5*i-33, scale=1.5)

bias = -3 - 2*np.random.uniform(size=N) # Bias
print(bias)
np.save('bias.npy', bias)

W = np.zeros((N, nT, nX))

for i, filts in enumerate(zip(temp_filt, spat_filt)):
    a, b = filts
    f = np.matmul(a[:,np.newaxis], b[np.newaxis,:])
    f = 3 * f/np.linalg.norm(f.flatten())
    W[i,:,:] = f
print(W.shape)
np.save('W.npy', W)

# Stimulation
nBin = 15000
nTrial = 200
stim = np.random.uniform(-1, 1, size=(nBin,nX))
print(stim.shape)

fr = np.zeros((N, nBin))
data = np.zeros((nTrial, nBin, N))
fr_all = np.zeros((nTrial, nBin, N))

for j in range(nTrial):
    print(f'trial{j} from {nTrial}')
    shared_noise = np.random.randn(nBin, 1) * .5
    for i, filt in enumerate(W):
        act = signal.convolve2d(stim, filt, mode='valid')
        act = np.concatenate((np.zeros((29, 1)), act))
        #         print(act.shape, bias.shape, shared_noise.shape)
        pr = sigmoid(act + bias[i] + shared_noise).flatten()
        fr[i, :] = pr
    fr_all[j, :, :] = fr.transpose()
    data[j, :, :] = np.random.binomial(n=1, p=fr.transpose(), size=(nBin, N))

print(data.shape)

stim = stim[:,:,np.newaxis]
print(stim.shape)

np.save('data.npy', data)
np.save('stim.npy', stim)