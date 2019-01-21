import torch
from torch.nn import functional as F


class Rebar:
    def __init__(self):
        self.log_temp = torch.tensor([.5], requires_grad=True)
        self.eta = torch.tensor([1.], requires_grad=True)
        self.temp_optim = torch.optim.Adam([self.log_temp], lr=1e-3, betas=(.9, 0.999), eps=1e-06)
        self.eta_optim = torch.optim.Adam([self.eta], lr=1e-3, betas=(.9, 0.999), eps=1e-06)

        self.discriminator = None
        self.compute_loss = F.binary_cross_entropy_with_logits
        self.bernulli_sampler = torch.distributions.bernoulli.Bernoulli

    def f(self, sp, stim):
        logits = self.discriminator(sp, stim)
        return F.binary_cross_entropy_with_logits(input=logits,
                                                  target=torch.ones_like(logits),
                                                  reduction='none')

    def concrete(self, logits, u):
        return logits + torch.log(u) - torch.log(1 - u)

    def sigmoid_with_temp(self, z):
        return torch.sigmoid(z / torch.exp(self.log_temp))

    def get_z_cond(self, logits, spikes):
        v = torch.rand_like(logits)
        uprime = 1. - torch.sigmoid(logits)
        noise_cond = spikes * (v * (1. - uprime) + uprime) + (1. - spikes) * v * uprime
        z_cond = self.concrete(logits=logits, u=noise_cond)
        z_cond.requires_grad_(True)
        return z_cond

    def step(self, logits, discriminator, stim):
        self.temp_optim.zero_grad()
        self.eta_optim.zero_grad()

        logits_detached = logits.detach()
        logits_detached.requires_grad_(True)
        self.discriminator = discriminator
        u = torch.rand_like(logits_detached)
        z = self.concrete(logits=logits.detach(), u=u)
        spikes = z.gt(0.).type_as(z)
        z.requires_grad_(True)

        log_probs = self.bernulli_sampler(logits=logits_detached).log_prob(spikes)
        d_log_probs = torch.autograd.grad(log_probs, logits_detached,
                                          grad_outputs=torch.ones_like(log_probs),
                                          retain_graph=False)[0]
        with torch.no_grad():
            loss_spikes = self.f(sp=spikes, stim=stim)

        loss_relaxed = self.f(self.sigmoid_with_temp(z), stim)
        d_loss_relaxed = torch.autograd.grad(outputs=loss_relaxed,
                                             inputs=z,
                                             grad_outputs=torch.ones_like(loss_relaxed),
                                             create_graph=True,
                                             retain_graph=True)[0]

        z_cond = self.get_z_cond(logits=logits.detach(), spikes=spikes)
        loss_cond = self.f(self.sigmoid_with_temp(z_cond), stim=stim)
        d_loss_cond = torch.autograd.grad(outputs=loss_cond,
                                          inputs=z_cond,
                                          grad_outputs=torch.ones_like(loss_cond),
                                          create_graph=True,
                                          retain_graph=True)[0]

        grads = (loss_spikes - self.eta * loss_cond) * d_log_probs.squeeze(2) + self.eta * (d_loss_relaxed - d_loss_cond).squeeze(2)
        var_loss = (grads ** 2).mean()  # CHECK the dimensions
        var_loss.backward()
        print(
            f"Temperature: {torch.exp(self.log_temp).data},"
            f" Eta: {self.eta.data}, TempGrad:{self.log_temp.grad.data},"
            f" EtaGrad:{self.eta.grad.data}, grad_est:{grads.mean().data}")

        self.eta_optim.step()
        self.temp_optim.step()

        logits.backward(grads.unsqueeze(1).detach())

        return loss_spikes.mean().cpu().numpy()
