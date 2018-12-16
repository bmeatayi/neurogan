import torch
from torch.nn import functional as F
import torch.autograd as autograd
import os
import numpy as np
import numpy.matlib
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from utils.evaluation import Visualize
import seaborn as sns
from modules.gumbel_softmax_binary import GumbelSoftmaxBinary

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
                 gs_temp = None
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
        """
        self.log_folder = log_folder
        self.optimizer_G = optimizer_g
        self.optimizer_D = optimizer_d

        assert gan_mode in ['js', 'wgan-gp', 'sn'], gan_mode + ' is not supported!'
        assert grad_mode in ['gs', 'rebar', 'reinforce'], grad_mode + ' is not supported!'
        assert gan_mode == 'wgan-gp' and lambda_gp is not None, "lambda_gp is not given!"

        if grad_mode is 'gs':
            assert gs_temp == None, 'gs_temp is not given!'
            self.gumbel_softmax = GumbelSoftmaxBinary(gs_temp=gs_temp)
        elif grad_mode is 'reinforce':
            self.bernoulli_sampler = torch.distributions.bernoulli.Bernoulli
        self.lambda_gp = lambda_gp
        self.gan_mode = gan_mode
        self.grad_mode = grad_mode

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
              n_disc_train=5
              ):

        self.logger.add_text('G-Architecture', repr(generator))
        self.logger.add_text('D-Architecture', repr(discriminator))
        self._reset_loss_history()

        if torch.cuda.is_available():
            generator.cuda()
            discriminator.cuda()

        optim_g = self.optimizer_G(generator.parameters(), lr=lr, betas=(b1, b2))
        optim_d = self.optimizer_G(discriminator.parameters(), lr=lr, betas=(b1, b2))

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
                    # Train generator
                    optim_g.zero_grad()
                    z = FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dim)))
                    fake_logits = generator(z, stim)
                    pred_fake = discriminator(self._logit2sample(fake_logits), stim)

                    if self.gan_mode == 'wgan-gp':
                        g_loss = -pred_fake.mean()
                    elif self.gan_mode == 'js':
                        g_loss = F.binary_cross_entropy_with_logits(inputs=pred_fake, target=real_label)
                    elif self.gan_mode == 'sn':
                        pass
                        # TODO: Implement Spectral Normalization

                    g_loss.backward()
                    optim_g.step()
                    g_loss = g_loss.data.cpu().numpy()
                    self.g_loss_history.append(g_loss)

                # Train discriminator
                optim_d.zero_grad()
                optim_g.zero_grad()

                z = FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dim)))
                fake_logits = generator(z, stim)
                pred_real = discriminator(real_sample, stim)


                if self.gan_mode == 'wgan-gp':
                    pred_fake = discriminator(self._logit2sample(fake_logits), stim)
                    grad_penalty = self.compute_gp(discriminator, real_sample, fake_logits, stim)
                    d_loss = torch.mean(pred_fake) - torch.mean(pred_real) + self.lambda_gp * grad_penalty
                elif self.gan_mode == 'js':
                    pred_fake = discriminator(self._logit2sample(fake_logits), stim)
                    d_real_loss = F.binary_cross_entropy_with_logits(pred_real, real_label)
                    d_fake_loss = F.binary_cross_entropy_with_logits(pred_fake, fake_label)
                    d_loss = (d_real_loss + d_fake_loss) / 2
                elif self.gan_mode == 'sn':
                    pass
                    # TODO: Implement spectral normalization objective function

                d_loss.backward()
                optim_d.step()
                d_loss = d_loss.data.cpu().numpy()
                self.d_loss_history.append(d_loss)

                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss}] [G loss: {g_loss}]")

                batches_done = epoch * len(train_loader) + i
                if batches_done % log_interval == 0:
                    self.log_result(generator, discriminator,
                                    batches_done, tr_loader=train_loader,
                                    val_loader=val_loader)
        self.log_result(generator, discriminator,
                        batches_done, tr_loader=train_loader,
                        val_loader=val_loader)

        self.plot_loss_history()
        self.logger.export_scalars_to_json(self.log_folder + "./all_scalars.json")
        self.logger.close()
        torch.save(generator, self.log_folder + 'generator.pt')

    def _logit2sample(self, fake_logits):
        r"""
        Converts logits to samples based on the gradient estimator method
        Args:
            fake_logits (torch.tensor): logits generated by generator

        Returns:
            sample (torch.tensor): binary or relaxed samples
        """
        if self.grad_mode is 'gs':
            return self.gumbel_softmax(fake_logits)
        elif self.grad_mode is 'reinforce':
            sampler = self.bernoulli_sampler(logits=fake_logits)
            return sampler.sample()
        elif self.grad_mode is 'rebar':
            return None
            # TODO: IMPLEMENT REBAR

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

    def log_result(self, generator, discriminator, batches_done, val_loader, n_sample=200):

        generator.eval()
        discriminator.eval()

        fake_data = torch.zeros([0, 0, 0, 0])
        real_data = torch.zeros([0, 0, 0, 0])

        for j in range(n_sample):
            temp_gen = torch.zeros([0, 0, 0])
            temp_real = torch.zeros([0, 0, 0])
            for i, inputs in enumerate(val_loader):
                cnt, stim = inputs
                batch_size = cnt.shape[0]
                stim = stim.type(FloatTensor)
                z = FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dim)))
                fake_sample = generator.generate(z, stim)
                temp_gen = torch.cat((temp_gen, fake_sample.detach().cpu()))
                temp_real = torch.cat((temp_real, cnt.type(FloatTensor)))
            fake_data = torch.cat((fake_data, temp_gen.unsqueeze(0)))
            real_data = torch.cat((real_data, temp_real.unsqueeze(0)))

        fake_data = np.squeeze(fake_data.numpy())
        real_data = np.squeeze(real_data.detach().cpu().numpy())

        pdf = PdfPages(self.out_folder + 'iter_' + str(batches_done) + '.pdf')
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
        viz.mean_per_bin(fake_data, 'GAN 1', neurons=[], label='Neuron ', figsize=[15, 10])
        pdf.savefig(bbox_inches='tight')
        viz.corr(fake_data, model='')
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

        GLM_filters = generator.GLM.weight.detach().cpu().numpy()
        N = GLM_filters.shape[0]
        fig, ax = plt.subplots(1, N, figsize=(40, 5))
        for i, f in enumerate(GLM_filters):
            ax[i].imshow(np.flip(f.reshape((30, 40)), axis=(0, 1)))
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('t')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title('Neuron' + str(i))

        # ax[0].imshow(np.flip(GLM_filters[0, :].reshape((20, 10)), axis=(0, 1)))
        # ax[1].imshow(np.flip(GLM_filters[1, :].reshape((20, 10)), axis=(0, 1)))
        plt.savefig(self.out_folder + 'filt %i.jpg' % batches_done, dpi=120)
        plt.close()
        generator.train()
        discriminator.train()
