import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GumbelSoftmaxBinary(nn.Module):
    def __init__(self, n_unit, gs_temp, learnable_temperature=False, eps=1e-12):
        """Binary Gumbel_Softmax relaxation module

        Reference:  Maddison et. al. (2016) "The concrete distribution: A continuous relaxation of discrete random
        variables", arXiv preprint arXiv:1611.00712.

        Args:
            n_unit (int): Number of neurons
            learnable_temperature (bool): To learn temperature
            gs_temp: Temperature value
            eps: Epsilon value for numerical stability
        """
        super(GumbelSoftmaxBinary, self).__init__()
        self.n_unit = n_unit
        self.eps = eps
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            self.temperature = Parameter(torch.Tensor(self.n_unit).fill_(gs_temp))
        else:
            self.temperature = gs_temp

    def forward(self, logits):
        L = self.sample_logistic(logits.shape)
        return torch.sigmoid((logits + L)/self.temperature)

    def sample_logistic(self, shape):
        U = torch.rand(shape)
        if torch.cuda.is_available():
            U.cuda()
        return torch.log(U + self.eps) - torch.log(1 - U)


if __name__ == '__main__':
    temperature = .5
    x = torch.ones((10, 10))*-1
    gumbelsoftmax = GumbelSoftmaxBinary(n_unit=10, gs_temp=temperature)
    y = gumbelsoftmax(x)
    print(torch.mean(y, dim=0))
    print(y)
