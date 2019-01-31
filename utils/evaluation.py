#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:42:19 2018
@author: j.lappalainen
Plots inspired by:
Molano-Mazon, Manuel, Arno Onken, Eugenio Piasini, and Stefano Panzeri. "Synthesizing realistic neural population
activity patterns using Generative Adversarial Networks." arXiv preprint arXiv:1803.00338 (2018).
"""

import numpy as np
from utils.plot_props import PlotProps


class Evaluate:
    """This class implements several evaluation metrics.

      Args:
          groundtruth (array): Groundtruth spike data
              (#repeats, #bins, #neurons)
          time (float): Total duration of the spike train in s.
      Attributes:
          groundtruth (array): Groundtruth spike data
              (#repeats, #bins, #neurons)
          n_repeats (int): # repetitions
          n_bins (int): # bins
          n_neurons (int): # neurons
          time (float): Total duration of the spike train in s.
      """

    def __init__(self, groundtruth, time=1):
        self.groundtruth = np.array(groundtruth)
        self.n_repeats, self.n_bins, self.n_neurons = self.groundtruth.shape
        self.time = time

    def spike_count_average(self, generated):
        """Computes the average number of spikes per neuron.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
        Returns:
            ndarray: Spike count average of generated data.
            ndarray: Spike count average of groundtruth data.
        """
        generated = self._check(generated)
        return generated.mean(axis=(0, 1)) / self.time, \
               self.groundtruth.mean(axis=(0, 1)) / self.time

    def spike_count_std(self, generated):
        """Computes the standard deviation of number of spikes per neuron.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
        Returns:
            ndarray: Spike count std of generated data.
            ndarray: Spike count std of groundtruth data.
        """
        generated = self._check(generated)
        return generated.std(axis=(0, 1)) / self.time, \
               self.groundtruth.std(axis=(0, 1)) / self.time

    def spikes_per_bin(self, generated):
        """"""
        generated = self._check(generated)
        return generated.mean(axis=0), \
               self.groundtruth.mean(axis=0)

    def correlation(self, generated):
        """Computes the normalized covariance between pairs of neurons.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
        Note: The resulting square matrix contains both, pairwise
        covariance of neurons within and across generated and grountruth
        data. For indexing,the first n_neurons correspond to generated
        neurons and the next n_neurons correspond to groundtruth neurons.
        Returns:
            ndarray: Square matrix of size (2 * n_neurons, 2 * n_neurons).
        """
        generated = self._check(generated)
        return np.corrcoef(generated.reshape(-1, self.n_neurons).T,
                           self.groundtruth.reshape(-1, self.n_neurons).T)

    def signal_correlation(self, generated):
        r"""
        Computes signal correlation between pairs of neurons
        Reference:      Lyamzin, D. R., Macke, J. H., & Lesica, N. A. (2010). Modeling population spike trains with
                        specified time-varying spike rates, trial-to-trial variability, and pairwise signal and noise
                        correlations
        Args:
            generated (ndarray): generated spike data
                    (#repeats, #bins, #neurons)

        Returns:
            ndarray: Square matrix of size (2 * n_neurons, 2 * n_neurons).
        """
        generated = self._check(generated)
        groundtruth_shuffled = np.copy(self.groundtruth)
        np.random.shuffle(groundtruth_shuffled)
        corr_shuffled = np.corrcoef(generated.reshape(-1, self.n_neurons).T,
                                    groundtruth_shuffled.reshape(-1, self.n_neurons).T).T
        return np.rot90(corr_shuffled)

    def noise_correlation(self, generated):
        r"""
        Computes noise correlation between pairs of neurons
        Args:
            generated (ndarray): generated spike data
                    (#repeats, #bins, #neurons)

        Returns:
            ndarray: Square matrix of size (2 * n_neurons, 2 * n_neurons).
        """
        generated = self._check(generated)
        return self.correlation(generated) - self.signal_correlation(generated)

    def lag_correlation(self, generated):
        """Lag-covariance between pairs of neurons.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
        Note: for each pair of neurons, we shift the activity of one of the
        neurons by one bin and compute the covariance between the resulting
        activities. This quantity thus indicates how strongly the activity
        of one of the neurons is related to the future activity of the
        other neuron.
        Returns:
            ndarray: Square matrix with lag covariance for generated
                neurons (n_neurons, n_neurons)
            ndarray: Square matrix with lag covariance for groundtruth
                neurons (n_neurons, n_neurons)

        """
        generated = self._check(generated)
        return np.corrcoef(np.roll(generated.reshape(-1, self.n_neurons),
                                   -1, axis=0),
                           generated.reshape(-1, self.n_neurons)), \
               nnp.corrcoef(np.roll(self.groundtruth.reshape(-1, self.n_neurons),
                                    -1, axis=0),
                            self.groundtruth.reshape(-1, self.n_neurons))

    def average_time_course(self):
        return NotImplemented

    def synchrony(self, generated):
        return NotImplemented

    def autocorrelogram(self, generated):
        return NotImplemented

    def _check(self, generated):
        generated = np.array(generated)
        assert generated.shape == self.groundtruth.shape, "Size mismatch!"
        return generated

    def _scramble(a, axis=-1):
        """
        Return an array with the values of `a` independently shuffled along the
        given axis
        Source:https://stackoverflow.com/questions/36272992/numpy-random-shuffle-by-row-independently
        """
        b = a.swapaxes(axis, -1)
        n = a.shape[axis]
        idx = np.random.choice(n, n, replace=False)
        b = b[..., idx]
        return b.swapaxes(axis, -1)


class Visualize(Evaluate):
    """Visualization class that leverages the evaluation metrics.

      Args:
          groundtruth (array): Groundtruth spike data
              (#repeats, #bins, #neurons)
          time (float): Total duration of the spike train in s.
      Attributes:
          plot (object): A plot properties instance, specifying
              some layout choices.
          Please read the Evaluation docstring for more infos.
      """
    plot = PlotProps()

    def __init__(self, groundtruth, time=None):
        super(Visualize, self).__init__(groundtruth, time)

    def mean(self, generated, model, ax=None, marker='.'):
        """Plots the spike-count-average for each neuron of
        groundtruth vs. the generated data.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
            model (str): The name of the model.
            ax (object, optional): An existing axis object.
                Defaults to None which creates a new axis.
            marker (str): Defaults to '.'.
        Returns:
            object: Axis object.
        """
        if not ax:
            ax = self.plot.init_subplot('Mean Firing Rates')
        model_mean, gt_mean = self.spike_count_average(generated)
        vmax = np.max([model_mean.max(), gt_mean.max()])
        ax.plot([0, vmax + .2], [0, vmax + .2], 'black')
        ax.plot(gt_mean, model_mean, marker, label=model, markersize=6)
        ax.set_xlabel('Real Mean Firing Rate')
        ax.set_ylabel('Generated Mean Firing Rate')
        ax.legend()
        return ax

    def std(self, generated, model, ax=None, marker='.'):
        """Plots the spike-count-std for each neuron of
        groundtruth vs. the generated data.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
            model (str): The name of the model.
            ax (object, optional): An existing axis object.
                Defaults to None which creates a new axis.
            marker (str): Defaults to '.'.
        Returns:
            object: Axis object.
        """
        if not ax:
            ax = self.plot.init_subplot('Std')
        model_std, gt_std = self.spike_count_std(generated)
        vmax = np.max([model_std.max(), gt_std.max()])
        ax.plot([0, vmax + .2], [0, vmax + .2], 'black')
        ax.plot(gt_std, model_std, marker, label=model, markersize=6)
        ax.set_xlabel('Real Std')
        ax.set_ylabel('Generated Std')
        ax.legend()
        return ax

    def mean_per_bin(self, generated, model, neurons=None, marker='.', label=None,
                     figsize=[5, 5]):
        """Plots a grid of the generated mean activity in timebins vs. the expected mean
           activity in timebins.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
            model (str): The name of the model.
            neurons (list, optional): List of neurons that are supposed to be plotted.
                Defaults to None, i.e. all neurons are plottet.
            marker (str): Defaults to '.'.
            label (str, optional): If specified, the subplots will be labeled in the top
                right corner with 'label%s'%neuron.
            figsize (list): Size of the figure.
        Returns:
            object: Axis object.
        """
        model, gt = self.spikes_per_bin(generated)
        if not neurons:
            neurons = np.arange(0, self.n_neurons, 1)
        gridwidth = int(np.ceil(np.sqrt(len(neurons))))
        gridheight = gridwidth if gridwidth * (gridwidth - 1) < len(neurons) else (gridwidth - 1)
        fig = self.plot.init_figure(figsize=figsize)
        for i, neuron in enumerate(neurons):
            ax = self.plot.init_subplot('',
                                        tot_tup=(gridheight, gridwidth),
                                        sp_tup=(int(i // gridwidth), int(i % gridwidth)))
            ax.plot(gt[:, neuron], model[:, neuron], '.', alpha=0.8)
            ax.plot([0, 1], [0, 1], 'black')
            if isinstance(label, str):
                ax.text(0.65, 0.9, '%s%s' % (label, neuron), transform=ax.transAxes, ha='right',
                        fontsize='small')
        fig.suptitle('Mean per Bin', y=1.02)
        fig.tight_layout()
        fig.text(0.5, 0.001, 'Expected Mean (a.u.)', ha='center')
        fig.text(0.001, 0.5, 'Generated Mean (a.u.)', va='center', rotation='vertical')

    def corr(self, generated, model, ax=None, marker='.'):
        """Plots the intrinsic correlation between neurons of
         the generated data vs. the groundtruth data.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
            model (str): The name of the model.
            ax (object, optional): An existing axis object.
                Defaults to None which creates a new axis.
            marker (str): Defaults to '.'.
        Returns:
            object: Axis object.
        """
        if not ax:
            ax = self.plot.init_subplot('Pairwise Total Correlation')
        corr = self.correlation(generated)
        triu_idx = np.triu_indices(n=self.n_neurons, k=1)
        within_gen = corr[:self.n_neurons, :self.n_neurons][triu_idx]
        within_gt = corr[self.n_neurons::, self.n_neurons::][triu_idx]
        vmax = np.max([within_gen.max(), within_gt.max()])
        vmin = np.max([within_gen.min(), within_gt.min()])
        ax.plot([vmin - .1, vmax + .1], [vmin - .1, vmax + .1], 'black')
        ax.plot(within_gt, within_gen, marker, label=model, markersize=8)
        ax.set_xlabel('Real Correlation')
        ax.set_ylabel('Generated Correlation')
        ax.legend()
        return ax

    def noise_corr(self, generated, model, ax=None, marker='.'):
        """Plots the noise correlation between neurons of
         the generated data vs. the groundtruth data.

        Args:
            generated (array): Generated spike data
                (#repeats, #bins, #neurons)
            model (str): The name of the model.
            ax (object, optional): An existing axis object.
                Defaults to None which creates a new axis.
            marker (str): Defaults to '.'.
        Returns:
            object: Axis object.
        """
        if not ax:
            ax = self.plot.init_subplot('Pairwise Noise Correlation')
        corr = self.noise_correlation(generated)
        triu_idx = np.triu_indices(n=self.n_neurons, k=1)
        within_gen = corr[:self.n_neurons, :self.n_neurons][triu_idx]
        within_gt = corr[self.n_neurons::, self.n_neurons::][triu_idx]
        vmax = np.max([within_gen.max(), within_gt.max()])
        vmin = np.max([within_gen.min(), within_gt.min()])
        ax.plot([vmin - .1, vmax + .1], [vmin - .1, vmax + .1], 'black')
        ax.plot(within_gt, within_gen, marker, label=model, markersize=8)
        ax.set_xlabel('Real Noise Correlation')
        ax.set_ylabel('Generated Noise Correlation')
        ax.legend()
        return ax

    def spiketrains(self, generated, neurons=[0, 1], trial_avg=False, figsize=[12, 3], aspect='auto', labels=False):
        """Plots an overview of the spiketrains.
    Args:
        generated (array): Generated spike data
            (#repeats, #bins, #neurons)
        neurons (list, optional): List of neurons that are supposed to be plotted.
            Defaults to [0, 1]. Leave empty to plot all neurons with trial averages.
        trial_avg (bool): Useful for an overview over all neurons! Whether to average over trials.
        figsize (list): Size of the figure.
        labels (bool): Label y axis with neuron numbers.
    Returns:
        tuple of objects: (ax1, ax2)
    """
        if not neurons:
            neurons = np.arange(0, self.n_neurons, 1)
            trial_avg = True
            labels = False
            print("No neurons selected. Averaging over trials to fit all neurons into the plot. Neuron labels were set"
                  " off.")
        if trial_avg:
            generated, groundtruth = self.spikes_per_bin(generated)
            generated = generated[None, ...]
            groundtruth = groundtruth[None, ...]
            n_repeats = 1
        else:
            generated = self._check(generated)
            groundtruth = self.groundtruth
            n_repeats = self.n_repeats

        fig = self.plot.init_figure(figsize)
        ax1 = self.plot.init_subplot('Groundtruth',
                                     tot_tup=(2, 1),
                                     sp_tup=(0, 0))
        ax2 = self.plot.init_subplot('Generated',
                                     tot_tup=(2, 1),
                                     sp_tup=(1, 0),
                                     sharex=ax1)
        ax2.set_xlabel('Timebins')
        ax1.set_ylabel('Neurons', labelpad=20)
        ax2.set_ylabel('Neurons', labelpad=20)
        if labels:
            ypad = 1 - 1 / (2 * (len(neurons) + 1))
            for i, neuron in enumerate(neurons):
                ax1.text(-0.01, ypad - i / len(neurons), str(neuron), transform=ax1.transAxes, ha='right', va='center')
                ax2.text(-0.01, ypad - i / len(neurons), str(neuron), transform=ax2.transAxes, ha='right', va='center')
        indices = np.tile(np.arange(0, n_repeats), len(neurons)) \
                  + np.array(neurons).repeat(n_repeats) * n_repeats
        ax1.imshow(groundtruth.transpose((2, 0, 1)).reshape(-1, self.n_bins)[indices], aspect=aspect)
        ax2.imshow(generated.transpose((2, 0, 1)).reshape(-1, self.n_bins)[indices], aspect=aspect)
        ax1 = self._rm_spines(ax1)
        ax2 = self._rm_spines(ax2)
        return ax1, ax2

    def _rm_spines(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
