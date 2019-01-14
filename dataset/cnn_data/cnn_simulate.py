import os
import sys
import numpy as np
import torch
from scipy.stats import norm

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from models.gen_cnn import GeneratorCNN

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
ShortTensor = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor
torch.set_default_tensor_type(FloatTensor)


nW = 40
nH = 40
nT = 5
nF1 = 3
nF2 = 2
f1sz = 11
f2sz = 7
nCell = 10
sharedNoiseDim = 3

# Create filters
x = norm.pdf(np.linspace(-1, 1, f1sz), loc=0, scale=.5)
y = norm.pdf(np.linspace(-1, 1, f1sz), loc=0, scale=.2)
z = norm.pdf(np.linspace(-1, 1, nT), loc=0, scale=.8)
xy = np.matmul(x[:, np.newaxis], y[:, np.newaxis].T)
xyz = np.tile(xy[np.newaxis, :, :], (nT, 1, 1)) * z[:, np.newaxis, np.newaxis]
convFilt1 = 3 * xyz / np.linalg.norm(xyz.flatten())

x = norm.pdf(np.linspace(-1, 1, f2sz), loc=0, scale=.3)
y = norm.pdf(np.linspace(-1, 1, f2sz), loc=0, scale=.7)
z = norm.pdf(np.linspace(-1, 1, nT - 2), loc=0, scale=.8)
xy = np.matmul(x[:, np.newaxis], y[:, np.newaxis].T)
xyz = np.tile(xy[np.newaxis, :, :], (nT - 2, 1, 1)) * z[:, np.newaxis, np.newaxis]
convFilt2 = 3 * xyz / np.linalg.norm(xyz.flatten())

fcSz = (2, 24, 24)
fcFilt = np.zeros((nCell, *fcSz))
for i in range(nCell):
    x = norm.pdf(np.linspace(0, fcSz[1], fcSz[1]), loc=2 + i * 2, scale=4)
    y = norm.pdf(np.linspace(0, fcSz[2], fcSz[2]), loc=2 + i * 2, scale=2)
    fcFilt[i, 0, :, :] = np.matmul(x[:, np.newaxis], y[:, np.newaxis].T)

    x = norm.pdf(np.linspace(0, fcSz[1], fcSz[1]), loc=2 + i * 2, scale=2)
    y = norm.pdf(np.linspace(0, fcSz[2], fcSz[2]), loc=2 + i * 2, scale=5)
    fcFilt[i, 1, :, :] = np.matmul(x[:, np.newaxis], y[:, np.newaxis].T)

fcFilt = .07 * fcFilt.reshape((fcFilt.shape[0], -1))

simulator = GeneratorCNN(nw=nW, nh=nH, nl=nT,
                         n_filters=(nF1, nF2),
                         kernel_size=(f1sz, f2sz),
                         n_cell=nCell)
simulator(torch.rand(size=(10, 3)), torch.rand(size=(10, 5, 40, 40)))
simulator.eval()
# simulator.conv1.weight.shape: torch.Size([3, 5, 11, 11])
# simulator.conv1.bias.shape: torch.Size([3])

# simulator.conv2.weight.shape: torch.Size([2, 3, 7, 7])
# simulator.conv2.bias.shape: torch.Size([2])

# x_conv2.shape: torch.Size([10, 2, 24, 24])
# simulator.fc.weight.shape: torch.Size([10, 1152])

simulator.conv1.weight.data = torch.tensor(convFilt1).unsqueeze(0).repeat((nF1, 1, 1, 1)).type(FloatTensor)
simulator.conv1.bias.data = torch.zeros(nF1).type(FloatTensor)

simulator.conv2.weight.data = torch.tensor(convFilt2).unsqueeze(0).repeat((nF2, 1, 1, 1)).type(FloatTensor)
simulator.conv2.bias.data = torch.zeros(nF2).type(FloatTensor)

simulator.fc.weight.data = torch.tensor(fcFilt).type(FloatTensor)
fcBias = torch.rand(nCell) - 5.0
simulator.fc.bias.data = fcBias

simulator.shared_noise.data = torch.tensor([0.0, 0.35, .2])

# Create stimulation matrix
stimLength = 35000
stim = (torch.rand(stimLength, nW, nH) - .5) * 2
nRepeat = 300
spikes = torch.zeros(nRepeat, stimLength, nCell).type(ShortTensor)
for i in range(stimLength - nT + 1):
    print(f"{i}/{stimLength}")
    z = torch.randn(nRepeat, sharedNoiseDim)
    spikes[:, i + 4, :] = simulator.generate(z, stim[i:i + 5, :, :].unsqueeze(0).repeat((nRepeat, 1, 1, 1))).squeeze()

spikes = spikes.cpu().numpy()
stim = stim.cpu().numpy()

np.save('stim.npy', stim)
np.save('spike.npy', spikes)
np.save('convFilt1.npy', convFilt1)
np.save('convFilt2.npy', convFilt2)
np.save('fcFilt.npy', fcFilt)
np.save('fcBias.npy', fcBias.cpu().numpy())
