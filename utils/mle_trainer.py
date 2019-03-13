import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as tvutil

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class MLE_trainer(object):
    default_adam_args = {"lr": 1e-3,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim=torch.optim.Adam,
                 optim_args={},
                 loss_func=torch.nn.BCEWithLogitsLoss(),
                 log_folder='log'):
        """

        Args:
            optim:
            optim_args:
            loss_func:
            log_folder:
        """
        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok=True)
        self.logger = SummaryWriter(log_folder)

        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

    def _reset_histories(self):
        """
        Resets the loss history of the model.
        """
        self.train_loss_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=100, log_nth=1, mdl_file='model'):
        """

        Args:
            model:
            train_loader:
            val_loader:
            num_epochs:
            log_nth:
            mdl_file:

        Returns:

        """

        optim = self.optim(model.parameters(), **self.optim_args)
        self.logger.add_text('Model architecture', repr(model))
        self.logger.add_text('Optimizer', repr(optim))
        self._reset_histories()

        if torch.cuda.is_available():
            model.cuda()

        print('Start training...')

        for epoch in range(num_epochs):
            for i, stim in enumerate(train_loader):
                spike, stim = stim
                batch_size = spike.shape[0]

                if torch.cuda.is_available():
                    stim, spike = stim.type(FloatTensor), spike.type(FloatTensor)

                optim.zero_grad()

                z = FloatTensor(np.random.normal(0, 1, (batch_size, 1)))
                outputs = model.forward(z, stim)

                loss = self.loss_func(outputs, spike)

                loss.backward()

                optim.step()

                print('[Epoch %d/%d] [Batch %d/%d] [Train loss: %f]' % (epoch, num_epochs, i, len(train_loader),
                                                                        loss.data.cpu().numpy()))

                self.train_loss_history.append(loss.data.cpu().numpy())
                batches_done = epoch * len(train_loader) + i
                self.logger.add_scalar('train_loss', self.train_loss_history[-1], batches_done)
                if batches_done % log_nth == 0:
                    val_loss = self.test(model, val_loader)
                    self.val_loss_history.append(val_loss)
                    print('[Epoch %d/%d] [Batch %d/%d] [Val loss: %f]' % (
                    epoch, num_epochs, i, len(train_loader), val_loss))
                    self.logger.add_scalar('val_loss', self.train_loss_history[-1], batches_done)
            print(model.shn_layer.weight.data.cpu().numpy().T)
            torch.save(model, self.log_folder + '//' + mdl_file + '.mdl')
        self.generate_data(generator=model, val_loader=val_loader, is_save=True)
        self.logger.close()
        print('FINISH.')

    def test(self, model, val_loader):
        model.eval()  # Set model state to evaluation
        val_losses = []
        for j, (spike, stim) in enumerate(val_loader, 1):
            batch_size = spike.shape[0]
            if torch.cuda.is_available():
                stim, spike = stim.type(FloatTensor), spike.type(FloatTensor)

            z = FloatTensor(np.random.normal(0, 1, (batch_size, 1)))
            outputs = model.forward(z, stim)

            loss = self.loss_func(outputs, spike)
            val_losses.append(loss.data.cpu().numpy())
        model.train()
        return np.mean(val_losses)

    def generate_data(self, generator, val_loader, n_sample=200, is_save=False):
        generator.eval()

        fake_data = None  # torch.zeros([0, 995, generator.n_t, generator.n_cell])
        real_data = None  # torch.zeros([0, 995, generator.n_t, generator.n_cell])

        for j in range(n_sample):
            temp_gen = torch.zeros([0, generator.n_t, generator.n_cell]).type(FloatTensor)
            temp_real = torch.zeros([0, generator.n_t, generator.n_cell]).type(FloatTensor)
            for i, inputs in enumerate(val_loader):
                cnt, stim = inputs
                batch_size = cnt.shape[0]
                stim = stim.type(FloatTensor)
                z = FloatTensor(np.random.normal(0, 1, (batch_size, generator.latent_dim)))
                fake_sample = generator.generate(z, stim)

                temp_gen = torch.cat((temp_gen, fake_sample.detach()))
                temp_real = torch.cat((temp_real, cnt.type(FloatTensor)))
            if fake_data is None:
                fake_data = torch.zeros([0, temp_gen.size(0), generator.n_t, generator.n_cell]).type(FloatTensor)
                real_data = torch.zeros([0, temp_gen.size(0), generator.n_t, generator.n_cell]).type(FloatTensor)

            fake_data = torch.cat((fake_data, temp_gen.unsqueeze(0)))
            real_data = torch.cat((real_data, temp_real.unsqueeze(0)))
            del temp_gen, temp_real

        fake_data = np.squeeze(fake_data.cpu().numpy())
        real_data = np.squeeze(real_data.cpu().numpy())
        if is_save:
            np.save(self.log_folder + 'fake_data.npy', fake_data)
            np.save(self.log_folder + 'real_data.npy', real_data)
        generator.train()
        return fake_data, real_data
