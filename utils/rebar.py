import torch
from torch.nn import functional as F


class Rebar:
    def __init__(self, hp_lr=5e-4):
        self.temp = torch.tensor([.5], requires_grad=True)
        self.eta = torch.tensor([1.], requires_grad=True)
        self.hp_optim = torch.optim.Adam([self.temp, self.eta], lr=hp_lr, betas=(.9, 0.999), eps=1e-06)

    def step(self, logits, discriminator, stim):
        discriminator.train()
        relaxed_spikes_logit, spikes = self._sample_spikes(logits)
        grad_estimation, g_loss = self._estimate_grad(discriminator=discriminator,
                                                      spikes=spikes,
                                                      relaxed_spikes_logit=relaxed_spikes_logit,
                                                      logits=logits,
                                                      stim=stim)
        logits.backward(grad_estimation)
        self.hp_optim.step()

        print(
            f"Temperature: {self.temp}, Eta: {self.eta}, TempGrad:{self.temp.grad}, EtaGrad:{self.eta.grad}, grad_est{grad_estimation.mean()}")
        self.temp.grad.zero_()
        self.eta.grad.zero_()
        return g_loss

    def _estimate_grad(self, discriminator, spikes, relaxed_spikes_logit, logits, stim):
        discriminator.zero_grad()
        spikes_logit_discriminator = discriminator(spikes, stim)
        # expected_labels = torch.ones_like(spikes_logit_discriminator)
        # loss_spikes = F.binary_cross_entropy_with_logits(input=spikes_logit_discriminator,
        #                                                  target=expected_labels)
        loss_spikes = -spikes_logit_discriminator#.mean()

        discriminator.zero_grad()
        relaxed_spikes = torch.sigmoid(relaxed_spikes_logit / self.temp)
        assert torch.sum(torch.isnan(relaxed_spikes)) == 0, "relaxed_spikes has a nan!"
        # loss_relaxed_spikes = F.binary_cross_entropy_with_logits(input=discriminator(relaxed_spikes, stim),
        #                                                          target=expected_labels)
        loss_relaxed_spikes = -discriminator(relaxed_spikes, stim)#.mean()

        assert torch.sum(torch.isnan(loss_relaxed_spikes)) == 0, "loss_relaxed_spikes has a nan!"
        d_loss_relaxed = torch.autograd.grad(
            loss_relaxed_spikes, relaxed_spikes_logit, grad_outputs=torch.ones_like(loss_relaxed_spikes),
            create_graph=True, retain_graph=True)[0]
        assert torch.sum(torch.isnan(d_loss_relaxed)) == 0, "d_loss_relaxed has a nan!"

        discriminator.zero_grad()
        relaxed_reparam_spikes = self._reparametrize(logits, spikes)
        # loss_relaxed_reparam_spikes = F.binary_cross_entropy_with_logits(
        #     input=discriminator(relaxed_reparam_spikes, stim),
        #     target=expected_labels)
        loss_relaxed_reparam_spikes = -discriminator(relaxed_reparam_spikes, stim)#.mean()
        if torch.sum(torch.isnan(loss_relaxed_reparam_spikes)) != 0:
            print(relaxed_reparam_spikes, loss_relaxed_reparam_spikes)
        assert torch.sum(torch.isnan(loss_relaxed_reparam_spikes)) == 0, "loss_relaxed_reparam_spikes has a nan!"
        d_loss_relaxed_reparam = torch.autograd.grad(
            loss_relaxed_reparam_spikes, relaxed_reparam_spikes,
            grad_outputs=torch.ones_like(loss_relaxed_reparam_spikes),
            create_graph=True, retain_graph=True)[0]
        assert torch.sum(torch.isnan(d_loss_relaxed_reparam)) == 0, "d_loss_relaxed_reparam has a nan!"

        log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(spikes)
        assert torch.sum(torch.isnan(log_prob)) == 0, "log_prob has a nan!"

        d_log_prob = torch.autograd.grad(log_prob, logits, grad_outputs=torch.ones_like(log_prob))[0]
        assert torch.sum(torch.isnan(d_log_prob)) == 0, "d_log_prob has a nan!"

        # diff = loss_spikes.unsqueeze(1) - self.eta * loss_relaxed_reparam_spikes.unsqueeze(1)
        diff = loss_spikes - self.eta * loss_relaxed_reparam_spikes
        d_logits = diff * d_log_prob + self.eta * (d_loss_relaxed - d_loss_relaxed_reparam)

        var_loss = (d_logits ** 2).mean()
        var_loss.backward()
        # torch.nn.utils.clip_grad_value_([self.eta, self.temp], 1.0)

        g_loss = loss_spikes.detach().cpu().numpy().mean()
        return d_logits.detach(), g_loss

    @staticmethod
    def _sample_spikes(logits):
        u = torch.rand_like(logits)
        relaxed_spikes_logits = logits.detach() + torch.log(u + 1e-20) - torch.log(1 - u)
        relaxed_spikes_logits.requires_grad_(True)
        spikes = relaxed_spikes_logits.detach().gt(0.).type_as(relaxed_spikes_logits)
        return relaxed_spikes_logits, spikes

    @staticmethod
    def _reparametrize(logits, spikes):
        v = torch.rand_like(logits)
        theta = torch.sigmoid(logits.detach())
        assert torch.sum(torch.isnan(theta)) == 0, "theta has a nan!"
        reparam_logits = spikes * torch.log(v / ((1 - v) * (1 - theta)) + 1) + (spikes - 1) * torch.log(
            v / ((1 - v) * theta) + 1)
        reparam_logits.requires_grad_(True)
        return reparam_logits
