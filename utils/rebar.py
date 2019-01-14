import torch
import numpy as np


class Rebar:
    def __init__(self, hp_lr=1e-5):
        self.temp = torch.tensor([1.], requires_grad=True)
        self.eta = torch.tensor([1.], requires_grad=True)
        self.hp_optim = torch.optim.Adam([self.temp, self.eta], lr=hp_lr, betas=(.9, 0.99999), eps=1e-04)

    def step(self, logits, discriminator, stim):
        relaxed_spikes, spikes = self._sample_spikes(logits)
        grad_estimation, g_loss = self._estimate_grad(discriminator=discriminator,
                                                      spikes=spikes, relaxed_spikes=relaxed_spikes,
                                                      logits=logits, stim=stim)
        logits.backward(grad_estimation)
        self.hp_optim.step()

        print(f"Temperature: {self.temp}, Eta: {self.eta}, TempGrad:{self.temp.grad}, EtaGrad:{self.eta.grad}")
        # self.temp.grad.zero_()
        self.eta.grad.zero_()
        return g_loss

    def _estimate_grad(self, discriminator, spikes, relaxed_spikes_logit, logits, stim):
        loss_spikes = -discriminator(spikes, stim)
        relaxed_reparam_spikes = self._reparametrize(logits, spikes)

        relaxed_spikes = torch.sigmoid(relaxed_spikes_logit / self.temp)

        assert torch.sum(torch.isnan(relaxed_spikes)) == 0, "sig_z has a nan!"

        loss_relaxed_spikes = -discriminator(relaxed_spikes, stim)
        assert torch.sum(torch.isnan(loss_relaxed_spikes)) == 0, "loss_relaxed_spikes has a nan!"

        loss_relaxed_reparam_spikes = -discriminator(relaxed_reparam_spikes, stim)
        if torch.sum(torch.isnan(loss_relaxed_reparam_spikes)) != 0:
            print(relaxed_reparam_spikes, loss_relaxed_reparam_spikes)
        assert torch.sum(torch.isnan(loss_relaxed_reparam_spikes)) == 0, "loss_relaxed_reparam_spikes has a nan!"

        log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(spikes)
        assert torch.sum(torch.isnan(log_prob)) == 0, "log_prob has a nan!"

        d_log_prob = torch.autograd.grad([log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
        assert torch.sum(torch.isnan(d_log_prob)) == 0, "d_log_prob has a nan!"

        d_loss_relaxed = torch.autograd.grad(
            [loss_relaxed_spikes], [z], grad_outputs=torch.ones_like(loss_relaxed_spikes),
            create_graph=True, retain_graph=True)[0]
        assert torch.sum(torch.isnan(d_loss_relaxed)) == 0, "d_loss_relaxed has a nan!"

        d_loss_relaxed_relaxed = torch.autograd.grad(
            [loss_relaxed_reparam_spikes], [relaxed_reparam_spikes],
            grad_outputs=torch.ones_like(loss_relaxed_reparam_spikes),
            create_graph=True, retain_graph=True)[0]
        assert torch.sum(torch.isnan(d_loss_relaxed_relaxed)) == 0, "d_loss_relaxed_relaxed has a nan!"

        diff = loss_spikes.unsqueeze(1) - self.eta * loss_relaxed_reparam_spikes.unsqueeze(1)

        d_logits = diff * d_log_prob + self.eta * (d_loss_relaxed - d_loss_relaxed_relaxed)

        var_loss = (d_logits ** 2).mean()
        var_loss.backward()
        torch.nn.utils.clip_grad_value_([self.eta, self.temp], 1.0)

        g_loss = loss_spikes.numpy().mean()
        return d_logits.detach(), g_loss

    def _sample_spikes(self, logits):
        u = torch.rand_like(logits)
        relaxed_spikes = logits.detach() + torch.log(u + 1e-20) - torch.log(1 - u)
        relaxed_spikes.requires_grad_(True)
        spikes = relaxed_spikes.detach().gt(0.).type_as(relaxed_spikes)
        return relaxed_spikes, spikes

    def _reparametrize(self, logits, spikes):
        v = torch.rand_like(logits)
        theta = torch.sigmoid(logits.detach())
        assert torch.sum(torch.isnan(theta)) == 0, "theta has a nan!"
        v_prime = v * (spikes - 1.) * (theta - 1.) + spikes * (v * theta + 1. - theta)
        z_tilde = logits.detach() + torch.log(v_prime+1e-20) - torch.log(1-v_prime+1e-20)
        assert torch.sum(torch.isnan(z_tilde)) == 0, "z_tilde has a nan!"
        z_tilde.requires_grad_(True)
        return z_tilde
