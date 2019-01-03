import torch
from torch.nn import functional as F
import torch.autograd as autograd
import os
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.evaluation import Visualize
from modules.gumbel_softmax_binary import GumbelSoftmaxBinary
from utils.rebar import Rebar
from utils.plot_props import PlotProps

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
plt.ioff()  # Deactivate interactive mode to avoid error on cluster run


class TrainerCGAN(object):
    def __init__(self, optimizer_g=torch.optim.Adam,
                 optimizer_d=torch.optim.Adam,
                 log_folder='results',
                 gan_mode='js',
                 lambda_gp=None,
                 grad_mode='gs',
                 gs_temp=None,
                 n_neuron=None,
                 gen_loss_mode='bce'
                 ):
        r"""
        Trainer class for conditional GAN
        Args:
            optimizer_g (torch.optim.Optimizer): optimizer of generator
            optimizer_d (torch.optim.Optimizer): optimizer of discriminator
            log_folder (str): path to the log folder
            gan_mode (str): mode of training
                            'js': Jensen-Shannon divergence
                            'wgan-gp': Wasserstein GAN with Gradient Penalty
                            'sn': Spectral Normalization (NOT IMPLEMENTED YET)
            lambda_gp (float): Gradient penalty scale factor (only for 'wgan-gp' gan_mode)
            grad_mode (str): Gradient estimator method
                                'gs': binary Gumbel-Softmax relaxation
                                'rebar': REBAR method
                                'reinforce': REINFORCE method
            gs_temp (float): Gumbel-Softmax temperature
            n_neuron (int): number of neurons
            gen_loss_mode (str): Generator loss function (options: 'bce' for binary cross entropy, 'hinge')
        """
        self.log_folder = log_folder
        self.optimizer_G = optimizer_g
        self.optimizer_D = optimizer_d

        assert gan_mode in ['js', 'wgan-gp', 'sn'], gan_mode + ' is not supported!'
        assert grad_mode in ['gs', 'rebar', 'reinforce'], grad_mode + ' is not supported!'
        assert gen_loss_mode in ['bce', 'hinge'], gen_loss_mode + 'is not supported!'

        if gan_mode == 'wgan-gp':
            assert lambda_gp is not None, "lambda_gp is not given!"
            self.lambda_gp = lambda_gp

        if grad_mode == 'gs':
            assert gs_temp is not None, 'gs_temp is not given!'
            assert n_neuron is not None, 'n_unit is not given!'
            self.gumbel_softmax = GumbelSoftmaxBinary(n_unit=n_neuron, gs_temp=gs_temp)
        elif grad_mode == 'reinforce':
            self.bernoulli_func = torch.distributions.bernoulli.Bernoulli
        elif grad_mode == 'rebar':
            self.rebar_estimator = Rebar()
            self.bernoulli_func = torch.distributions.bernoulli.Bernoulli

        self.gan_mode = gan_mode
        self.grad_mode = grad_mode
        self.gen_loss_mode = gen_loss_mode

        self.d_loss_history = []
        self.g_loss_history = []

        os.makedirs(log_folder, exist_ok=True)
        self.logger = SummaryWriter(log_folder)

    def _reset_loss_history(self):
        self.d_loss_history = []
        self.g_loss_history = []

    def train(self, generator, discriminator, train_loader, val_loader,
              lr=0.0002, b1=0.5, b2=0.999,
              log_interval=400, n_epochs=200,
              n_disc_train=5,
              temp_anneal=1):
        r"""
        train conditional GAN

        Args:
            generator (nn.module): Generator
            discriminator (nn.module): Discriminator
            train_loader (dataloader): train dataloader
            val_loader (dataloader): validation dataloader
            lr (float): Adam optimizer learning rate
            b1 (float): Adam optimizer beta1 parameter
            b2 (float): Adam optimizer beta2 parameter
            log_interval (int): iteration intervals for logging results
            n_epochs  (int): number of total epochs
            n_disc_train (int): train discriminator n_disc_train times vs. 1 train step of generator
            temp_anneal (float): annealing factor of Gumbel-Softmax temperature

        Returns:
            void
        """

        self.logger.add_text('G-Architecture', repr(generator))
        self.logger.add_text('D-Architecture', repr(discriminator))
        self.logger.add_text('GAN-mode', self.gan_mode)
        self.logger.add_text('Grad-mode', self.grad_mode)
        self._reset_loss_history()

        if torch.cuda.is_available():
            generator.cuda()
            discriminator.cuda()

        optim_g = self.optimizer_G(generator.parameters(), lr=lr*10, betas=(b1, b2))
        optim_d = self.optimizer_D(discriminator.parameters(), lr=lr, betas=(b1, b2))

        self.logger.add_text('G-optim', repr(optim_g))
        self.logger.add_text('D-optim', repr(optim_d))

        for epoch in range(n_epochs):
            for i, inputs in enumerate(train_loader):
                spike, stim = inputs
                batch_size = spike.shape[0]

                real_sample = spike.type(FloatTensor)
                stim = stim.type(FloatTensor)

                real_label = FloatTensor(batch_size, 1).fill_(1.0)
                fake_label = FloatTensor(batch_size, 1).fill_(0.0)

                if i % n_disc_train == 0:
                    # Train Generator
                    optim_g.zero_grad()
                    discriminator.eval()
                    z = FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dim)))
                    fake_logits = generator(z, stim)
                    if self.grad_mode == 'rebar':
                        g_loss = self.rebar_estimator.step(logits=fake_logits,
                                                           discriminator=discriminator,
                                                           stim=stim)
                    else:
                        fake_samples = self._logit2sample(fake_logits)
                        pred_fake = discriminator(fake_samples, stim)

                        g_loss = self._compute_g_loss(fake_logits=fake_logits,
                                                      pred_fake=pred_fake,
                                                      fake_samples=fake_samples)
                        if self.grad_mode == 'reinforce':
                            fake_logits.backward(g_loss)
                            g_loss = g_loss.mean()
                        else:
                            g_loss.backward()
                        g_loss = g_loss.data.cpu().numpy()

                    optim_g.step()

                    self.g_loss_history.append(g_loss)

                # Train discriminator
                discriminator.train()
                optim_d.zero_grad()
                optim_g.zero_grad()

                z = FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dim)))
                fake_logits = generator(z, stim)
                pred_real = discriminator(real_sample, stim)

                if self.gan_mode == 'wgan-gp':
                    pred_fake = discriminator(self._logit2sample(fake_logits), stim)
                    grad_penalty = self.compute_gp(discriminator, real_sample, fake_logits, stim)
                    d_loss = torch.mean(pred_fake) - torch.mean(pred_real) + self.lambda_gp * grad_penalty
                elif self.gan_mode == 'js' or self.gan_mode == 'sn':
                    pred_fake = discriminator(self._logit2sample(fake_logits), stim)
                    d_real_loss = F.binary_cross_entropy_with_logits(pred_real, real_label)
                    d_fake_loss = F.binary_cross_entropy_with_logits(pred_fake, fake_label)
                    d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optim_d.step()
                d_loss = d_loss.data.cpu().numpy()
                self.d_loss_history.append(d_loss)

                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss}] [G loss: {g_loss}]")

                batches_done = epoch * len(train_loader) + i
                if batches_done % log_interval == 0:
                    self.log_result(generator, discriminator,
                                    batches_done,
                                    val_loader=val_loader)

            # Temperature annealing
            if self.grad_mode == 'gs':
                self.gumbel_softmax.temperature *= temp_anneal

        self.log_result(generator, discriminator, batches_done, val_loader=val_loader)

        self.plot_loss_history()
        self.logger.export_scalars_to_json(self.log_folder + "./all_scalars.json")
        self.logger.close()
        torch.save(generator, self.log_folder + 'generator.pt')
        torch.save(discriminator, self.log_folder + 'discriminator.pt')

    def _logit2sample(self, fake_logits):
        r"""
        Converts logits to samples based on the gradient estimator method
        Args:
            fake_logits (torch.tensor): logits generated by generator

        Returns:
            sample (torch.tensor): binary or relaxed samples
        """
        if self.grad_mode == 'gs':
            return self.gumbel_softmax(fake_logits)
        elif self.grad_mode == 'reinforce' or self.grad_mode == 'rebar':
            self.sampler = self.bernoulli_func(logits=fake_logits)
            return self.sampler.sample()

    def compute_gp(self, discriminator, real_sample, fake_sample, stim):
        r"""
        Computes gradient penatly in WGAN-GP method
        Reference: Gulrajani et. al. (2017). Improved training of Wasserstein GANs.

        Args:
            discriminator (torch.nn.Module): Discriminator
            real_sample (torch.nn.Module): Real samples
            fake_sample (torch.tensor: Fake samples
            stim (torch.tensor): Stimulation

        Returns:
            gradient penalty (torch.tensor)
        """
        alpha = FloatTensor(np.random.rand(real_sample.size(0), 1, 1))
        ip = autograd.Variable(alpha * real_sample - (1 - alpha) * fake_sample, requires_grad=True)
        disc_ip = discriminator(ip, stim)
        pre_grads = FloatTensor(real_sample.size(0), 1).fill_(1.0)

        grads = torch.autograd.grad(outputs=disc_ip, inputs=ip,
                                    grad_outputs=pre_grads, retain_graph=True,
                                    create_graph=True, only_inputs=True)[0]
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def _compute_g_loss(self, fake_logits, pred_fake, fake_samples):
        r"""
        Computes loss for the generator
        Args:
            pred_fake (torch.tensor): output of the discriminator (logit)
            fake_samples (torch.tensor): generated samples by the generator (discretized or relaxed version)

        Returns:
            g_loss (torch.tensor): loss value
        """
        if self.gen_loss_mode == 'bce':
            g_loss = -pred_fake.mean()
        elif self.gen_loss_mode == 'hinge':
            pass
            # TODO: Implement hinge loss

        if self.grad_mode == 'reinforce':
            log_probability = self.sampler.log_prob(fake_samples)
            d_log_probability = autograd.grad([log_probability], [fake_logits],
                                              grad_outputs=torch.ones_like(log_probability))[0]
            g_loss = -pred_fake.detach() * d_log_probability.detach()

            # g_loss = (g_loss.detach() * log_probability).mean()
        elif self.grad_mode is 'rebar':
            pass
            # TODO: Implement REBAR
        return g_loss

    def log_result(self, generator, discriminator, batches_done, val_loader, n_sample=200):

        generator.eval()
        discriminator.eval()

        fake_data = torch.zeros([0, 970, generator.n_t, generator.n_cell])
        real_data = torch.zeros([0, 970, generator.n_t, generator.n_cell])

        for j in range(n_sample):
            temp_gen = torch.zeros([0, generator.n_t, generator.n_cell])
            temp_real = torch.zeros([0, generator.n_t, generator.n_cell])
            for i, inputs in enumerate(val_loader):
                cnt, stim = inputs
                batch_size = cnt.shape[0]
                stim = stim.type(FloatTensor)
                z = FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dim)))
                fake_sample = self._logit2sample(generator(z, stim))
                if self.grad_mode == 'gs':
                    fake_sample[fake_sample >= .5] = 1
                    fake_sample[fake_sample < .5] = 0
                temp_gen = torch.cat((temp_gen, fake_sample.detach().cpu()))
                temp_real = torch.cat((temp_real, cnt.type(FloatTensor)))
            fake_data = torch.cat((fake_data, temp_gen.unsqueeze(0)))
            real_data = torch.cat((real_data, temp_real.unsqueeze(0)))

        fake_data = np.squeeze(fake_data.numpy())
        real_data = np.squeeze(real_data.detach().cpu().numpy())

        pdf = PdfPages(self.log_folder + 'iter_' + str(batches_done) + '.pdf')
        if fake_data.ndim == 2:
            fake_data = fake_data[:, :, np.newaxis]
            real_data = real_data[:, :, np.newaxis]

        # Evaluation metrics
        viz = Visualize(real_data, time=1)  # specify time for correct firing rates
        fig, ax = plt.subplots(figsize=(5, 5))
        viz.mean(fake_data, '')
        pdf.savefig(bbox_inches='tight')

        viz.std(fake_data, '')
        pdf.savefig(bbox_inches='tight')

        viz.corr(fake_data, model='')
        pdf.savefig(bbox_inches='tight')

        viz.noise_corr(fake_data, model='')
        pdf.savefig(bbox_inches='tight')

        real_glm_filters = np.load('..//dataset//GLM_2D_30n_shared_noise//W.npy')
        real_glm_biases = np.load('..//dataset//GLM_2D_30n_shared_noise//bias.npy')
        real_w_shared_noise = -.5

        gen_glm_filters = generator.GLM.weight.detach().cpu().numpy().reshape(real_glm_filters.shape)
        gen_glm_biases = generator.GLM.bias.detach().cpu().numpy()
        gen_w_shared_noise = generator.shn_layer.weight.detach().cpu().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title('GLM filter parameters')
        ax[0].plot([-1, 2], [-1, 2], 'black')
        ax[0].plot(real_glm_filters.flatten(), np.flip(gen_glm_filters, axis=(1, 2)).flatten(), '.')
        ax[0].plot(real_w_shared_noise, gen_w_shared_noise, '*', markersize=5, label='Shared noise scale')

        ax[1].set_title('GLM biases')
        ax[1].plot([-6, -3], [-6, -3], 'black')
        ax[1].plot(real_glm_biases, gen_glm_biases, '.')
        pdf.savefig(bbox_inches='tight')

        viz.mean_per_bin(fake_data, 'GAN 1', neurons=[], label='Neuron ', figsize=[15, 10])
        pdf.savefig(bbox_inches='tight')

        # for i, spikes in enumerate(fake_data.transpose(2, 0, 1)):
        #     fig, ax = plt.subplots(2, 2, figsize=(20, 5))
        #     fig.suptitle('Neuron %i, iteration %i' % (i, batches_done))
        #     ax[0, 0].imshow(spikes)
        #     ax[0, 0].set_xlabel('time')
        #     ax[0, 0].set_ylabel('repetitions')
        #     ax[0, 0].set_xticks([])
        #     ax[0, 0].set_yticks([])
        #     ax[0, 0].set_title('GAN data')
        #
        #     ax[1, 0].imshow(real_data[:, :, i])
        #     ax[1, 0].set_xlabel('time')
        #     ax[1, 0].set_ylabel('repetitions')
        #     ax[1, 0].set_xticks([])
        #     ax[1, 0].set_yticks([])
        #     ax[1, 0].set_title('Real data')
        #
        #     ax[0, 1].plot(real_data[:, :, i].mean(axis=0), label='Real data')
        #     ax[0, 1].plot(spikes.mean(axis=0), label='GAN data')
        #     ax[0, 1].set_ylim(0, 1)
        #     ax[0, 1].set_xlabel('time')
        #     ax[0, 1].set_title('Mean firing rate')
        #     ax[0, 1].legend(loc=1)
        #
        #     ax[1, 1].plot(real_data[:, :, i].std(axis=0), label='Real data')
        #     ax[1, 1].plot(spikes.std(axis=0), label='GAN data')
        #     ax[1, 1].set_xlabel('time')
        #     ax[1, 1].set_title('Std of spike data')
        #     ax[1, 1].legend(loc=1)
        #     ax[1, 1].set_ylim(0, 1)
        #     plt.subplots_adjust(top=0.9, bottom=0.1, hspace=.8, wspace=0.2)
        #     pdf.savefig(fig)

        pdf.close()
        plt.close()

        # GLM_filters = generator.GLM.weight.detach().cpu().numpy()
        # N = GLM_filters.shape[0]
        # fig, ax = plt.subplots(1, N, figsize=(40, 5))
        # for i, f in enumerate(GLM_filters):
        #     ax[i].imshow(np.flip(f.reshape((30, 40)), axis=(0, 1)))
        #     ax[i].set_xlabel('x')
        #     ax[i].set_ylabel('t')
        #     ax[i].set_xticks([])
        #     ax[i].set_yticks([])
        #     ax[i].set_title('Neuron' + str(i))
        #
        # plt.savefig(self.log_folder + 'filt %i.jpg' % batches_done, dpi=120)
        plt.close()
        generator.train()
        discriminator.train()

    def plot_loss_history(self):
        plotprop = PlotProps()
        fig = plotprop.init_figure(figsize=(14, 7))
        ax = plotprop.init_subplot(title='Loss history',
                                   tot_tup=(1, 1), sp_tup=(0, 0),
                                   xlabel='Iteration', ylabel='Value')

        ax.plot(np.arange(0, len(self.d_loss_history)), self.d_loss_history,
                linewidth=2.5, label='Discriminator loss')
        ax.plot(np.linspace(0, len(self.d_loss_history), len(self.g_loss_history)),
                self.g_loss_history, linewidth=2.5, label='Generator loss')
        plotprop.legend()
        plt.savefig(self.log_folder + 'loss_history.jpg', dpi=200)
        plt.close()