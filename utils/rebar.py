import torch
import numpy as np


class Rebar:
    def __init__(self, hp_lr=1e-5):
        self.temp = torch.tensor([1.], requires_grad=True)
        self.eta = torch.tensor([1.], requires_grad=True)
        self.hp_optim = torch.optim.Adam([self.temp, self.eta], lr=hp_lr, betas=(.9, 0.99999), eps=1e-04)

    def step(self, logits, discriminator, stim):
        u = torch.rand_like(logits)
        v = torch.rand_like(logits)
        z = logits.detach() + torch.log(u) - torch.log1p(-u)
        z.requires_grad_(True)
        b = z.gt(0.).type_as(z)
        f_b = -discriminator(b, stim)
        d_logits = self.estimate(discriminator=discriminator,
                                 f_b=f_b, b=b, u=u, v=v, z=z,
                                 logits=logits, stim=stim)
        logits.backward(d_logits)  # mean of batch
        d_logits = d_logits.numpy()
        self.hp_optim.step()

        print(f"Temperature: {self.temp}, Eta: {self.eta}, TempGrad:{self.temp.grad}, EtaGrad:{self.eta.grad}")
        # self.temp.grad.zero_()
        self.eta.grad.zero_()
        return d_logits.mean()

    def estimate(self, discriminator, f_b, b, u, v, z, logits, stim):
        z_tilde = self._get_z_tilde(logits, b, v)
        sig_z = torch.sigmoid(z / self.temp)
        assert torch.sum(torch.isnan(sig_z)) == 0, "sig_z has a nan!"
        f_z = -discriminator(sig_z, stim)
        assert torch.sum(torch.isnan(f_z)) == 0, "f_z has a nan!"
        f_z_tilde = -discriminator(z_tilde, stim)
        if torch.sum(torch.isnan(f_z_tilde)) != 0:
            print(z_tilde, f_z_tilde)
        assert torch.sum(torch.isnan(f_z_tilde)) == 0, "f_z_tilde has a nan!"
        log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
        assert torch.sum(torch.isnan(log_prob)) == 0, "log_prob has a nan!"
        d_log_prob = torch.autograd.grad(
            [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
        assert torch.sum(torch.isnan(d_log_prob)) == 0, "d_log_prob has a nan!"
        d_f_z = torch.autograd.grad(
            [f_z], [z], grad_outputs=torch.ones_like(f_z),
            create_graph=True, retain_graph=True)[0]
        assert torch.sum(torch.isnan(d_f_z)) == 0, "d_f_z has a nan!"
        d_f_z_tilde = torch.autograd.grad(
            [f_z_tilde], [z_tilde], grad_outputs=torch.ones_like(f_z_tilde),
            create_graph=True, retain_graph=True)[0]
        assert torch.sum(torch.isnan(d_f_z_tilde)) == 0, "d_f_z_tilde has a nan!"
        diff = f_b.unsqueeze(1) - self.eta * f_z_tilde.unsqueeze(1)
        d_logits = diff * d_log_prob + self.eta * (d_f_z - d_f_z_tilde)
        var_loss = (d_logits ** 2).mean()
        var_loss.backward()
        torch.nn.utils.clip_grad_value_([self.eta, self.temp], 1.0)
        return d_logits.detach()

    def _get_z_tilde(self, logits, b, v):
        theta = torch.sigmoid(logits.detach())
        assert torch.sum(torch.isnan(theta)) == 0, "theta has a nan!"
        v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
        z_tilde = logits.detach() + torch.log(v_prime+1e-20) - torch.log(1-v_prime+1e-20)
        assert torch.sum(torch.isnan(z_tilde)) == 0, "z_tilde has a nan!"
        z_tilde.requires_grad_(True)
        return z_tilde
