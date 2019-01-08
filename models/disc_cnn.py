import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DiscriminatorCNN_proj(nn.Module):
    def __init__(self, nw=40, nh=40, nl=5,
                 n_filters=(15, 9),
                 kernel_size=(15, 11),
                 n_cell=1):
        super(DiscriminatorCNN_proj, self).__init__()
        self.nW = nw
        self.nH = nh
        self.nL = nl
        self.nFiltL1 = n_filters[0]
        self.nFiltL2 = n_filters[1]
        self.szFiltL1 = kernel_size[0]
        self.szFiltL2 = kernel_size[1]
        self.nCell = n_cell
        self.l3_filt_shape = None

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=self.nL,
                                                      out_channels=self.nFiltL1,
                                                      kernel_size=(self.szFiltL1, self.szFiltL1),
                                                      stride=1,
                                                      padding=0,
                                                      bias=True))

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=self.nFiltL1,
                                                      out_channels=self.nFiltL2,
                                                      kernel_size=(self.szFiltL2, self.szFiltL2),
                                                      stride=1,
                                                      padding=0,
                                                      bias=True))

        in_shp, conv2outShape = self._compute_fc_in()
        self.l3_filt_shape = (self.nCell, *conv2outShape)
        self.fc1 = nn.utils.spectral_norm(nn.Linear(in_features=in_shp, out_features=self.nCell, bias=True))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(in_features=self.nCell, out_features=64, bias=True))
        self.fc3 = nn.utils.spectral_norm(nn.Linear(in_features=64, out_features=1, bias=True))

        self.fcSpike1 = nn.utils.spectral_norm(nn.Linear(in_features=self.nCell, out_features=32, bias=True))
        self.fcSpike2 = nn.utils.spectral_norm(nn.Linear(in_features=32, out_features=64, bias=True))
        self.fcSpike3 = nn.utils.spectral_norm(nn.Linear(in_features=64, out_features=self.nCell, bias=True))

    def forward(self, spike, stim):
        x_conv = F.relu(self.conv1(stim))
        x_conv = F.relu(self.conv2(x_conv))
        x_fc1 = F.relu(self.fc1(x_conv.view([x_conv.shape[0], -1])))
        x_fc2 = F.relu(self.fc2(x_fc1))

        x_sp = F.relu(self.fcSpike1(spike))
        x_sp = F.relu(self.fcSpike2(x_sp))
        x_sp = F.relu(self.fcSpike3(x_sp))

        x = F.relu(self.fc3(x_fc2)) + torch.bmm(x_sp, x_fc1.unsqueeze(2)).squeeze(2)
        return x

    def _compute_fc_in(self):
        x = np.random.random([1, self.nL, self.nW, self.nH])
        x = torch.from_numpy(x)
        x = self.conv1(x.float())
        x = self.conv2(x)
        conv2shape = x.size()[1:]
        x = x.view(x.size(0), -1)
        return x.size(1), conv2shape


class DiscriminatorCNN(nn.Module):
    def __init__(self, nw=40, nh=40, nl=5,
                 n_filters=(15, 9),
                 kernel_size=(15, 11),
                 n_cell=1, spectral_norm=True,
                 mid_act_func=nn.LeakyReLU(0.2, inplace=True),
                 n_units=[128, 256, 512, 256, 128],
                 p_drop=None):
        super(DiscriminatorCNN, self).__init__()
        self.nW = nw
        self.nH = nh
        self.nL = nl
        self.nFiltL1 = n_filters[0]
        self.nFiltL2 = n_filters[1]
        self.szFiltL1 = kernel_size[0]
        self.szFiltL2 = kernel_size[1]
        self.nCell = n_cell
        self.l3_filt_shape = None

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=self.nL,
                                                      out_channels=self.nFiltL1,
                                                      kernel_size=(self.szFiltL1, self.szFiltL1),
                                                      stride=1,
                                                      padding=0,
                                                      bias=True))

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=self.nFiltL1,
                                                      out_channels=self.nFiltL2,
                                                      kernel_size=(self.szFiltL2, self.szFiltL2),
                                                      stride=1,
                                                      padding=0,
                                                      bias=True))

        in_shp, conv2outShape = self._compute_fc_in()
        self.l3_filt_shape = (self.nCell, *conv2outShape)

        dense_layers = OrderedDict()
        if spectral_norm:
            dense_layers['fcin'] = nn.utils.spectral_norm(nn.Linear(in_shp+self.nCell, n_units[0]))
        else:
            dense_layers['fcin'] = nn.Linear(in_shp+self.nCell, n_units[0])

        dense_layers['act0'] = mid_act_func
        for i in range(1, len(n_units)):
            if spectral_norm:
                dense_layers['fc' + str(i)] = nn.utils.spectral_norm(nn.Linear(n_units[i - 1], n_units[i]))
            else:
                dense_layers['fc' + str(i)] = nn.Linear(n_units[i - 1], n_units[i])

            if p_drop is not None:
                dense_layers['dropout' + str(i)] = nn.Dropout(p_drop)

            dense_layers['act' + str(i)] = mid_act_func

        if spectral_norm:
            dense_layers['fcout'] = nn.utils.spectral_norm(nn.Linear(n_units[-1], 1))
        else:
            dense_layers['fcout'] = nn.Linear(n_units[-1], 1)

        self.dense_layers = nn.Sequential(dense_layers)

    def forward(self, spike, stim):
        x_conv = F.relu(self.conv1(stim))
        x_conv = F.relu(self.conv2(x_conv))
        x = self.dense_layers(torch.cat((x_conv.view([x_conv.shape[0], -1]), spike.view(spike.size(0), -1)),dim=1))
        return x

    def _compute_fc_in(self):
        x = np.random.random([1, self.nL, self.nW, self.nH])
        x = torch.from_numpy(x)
        x = self.conv1(x.float())
        x = self.conv2(x)
        conv2shape = x.size()[1:]
        x = x.view(x.size(0), -1)
        return x.size(1), conv2shape


if __name__ == '__main__':
    d = DiscriminatorCNN(n_cell=8)
    stim = torch.rand(10, 5, 40, 40)
    spike = torch.randn(10, 8)
    out = d(spike, stim)
    assert out.shape == (10, 1), "Error in the output shape"
