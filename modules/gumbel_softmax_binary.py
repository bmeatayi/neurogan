import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
# from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.uniform import Uniform
# from torch.distributions.transforms import SigmoidTransform, AffineTransform


class GumbelSoftmaxBinary(nn.Module):
    def __init__(self, n_unit, gs_temp, learnable_temperature=False, eps=1e-20):
        r"""Binary Gumbel_Softmax relaxation module

        Reference:  Maddison et. al. (2016) "The concrete distribution: A continuous relaxation of discrete random
        variables", arXiv preprint arXiv:1611.00712.

        Args:
            n_unit (int): Number of neurons
            learnable_temperature (bool): To learn temperature
            gs_temp (float): Temperature value
            eps (float): Epsilon value for numerical stability
        """
        super(GumbelSoftmaxBinary, self).__init__()
        self.n_unit = n_unit
        self.eps = eps

        # base_distribution = Uniform(0, 1)
        # transforms = [SigmoidTransform().inv, AffineTransform(loc=0, scale=1)]
        # self.logistic_distribution = TransformedDistribution(base_distribution, transforms)

        if learnable_temperature:
            self.temperature = Parameter(torch.Tensor(self.n_unit).fill_(gs_temp))
        else:
            self.temperature = gs_temp

    def forward(self, logits):
        L = self.sample_logistic(logits.shape)
        return torch.sigmoid((logits + L)/self.temperature)

    def sample_logistic(self, shape):
        U = torch.rand(shape)
        # l = self.logistic_distribution.sample(shape)
        return torch.log(U + self.eps) - torch.log(1 - U)


if __name__ == '__main__':
    temperature = .1
    x = torch.ones((1000, 10))*.1
    x = torch.log(x/(1-x))

    gumbelsoftmax = GumbelSoftmaxBinary(n_unit=10, gs_temp=temperature)
    y = gumbelsoftmax(x)
    y[y >= .5] = 1
    y[y <= .5] = 0
    print(torch.mean(y, dim=0))
    # print(y)
