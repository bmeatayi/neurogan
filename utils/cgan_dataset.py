"""
Dataset classes for conditional GAN

Author: Mohamad Atayi
"""
import numpy as np
import torch.utils.data as data
import math


class GanDataset(data.Dataset):
    def __init__(self, spike_file, stim_file=None, cell_idx=None,
                 stim_win_len=1, cnt_win_len=0, n_split=1, st_end=None):
        r"""
        Spike count dataset for GAN

        Args:
            spike_file (str): path to the spike file
            stim_file (str): path to the stimulus file
            cell_idx (list): list of neurons' ID to be included in the dataset
            stim_win_len (int): length of returned stimulus chunk
            cnt_win_len (int): length of returned spike chunk
            n_split (int): number of time bins in each stimulus frame
            st_end (tuple): start and end frame of dataset
        """
        super(GanDataset, self).__init__()

        self.cell_idx = cell_idx
        self.stim_win_len = stim_win_len
        self.cnt_win_len = cnt_win_len
        self.n_split = n_split
        self.stim = None
        self.spike_count = np.load(spike_file)
        self.stim = np.load(stim_file)

        if st_end is not None:
            self.stim = self.stim[st_end[0]: st_end[1], :, :]
            self.spike_count = self.spike_count[:, st_end[0] * self.n_split:st_end[1] * self.n_split, :]

        if cell_idx is not None:
            self.spike_count = self.spike_count[:, :, cell_idx]

        self.cnt_mu = self.spike_count.mean(axis=(0, 1))
        self.cnt_std = self.spike_count.std(axis=(0, 1))
        self.spike_count = self.spike_count  # (self.spike_count - self.cnt_mu)/self.cnt_std
        self.n_repeats = self.spike_count.shape[0]
        self.n_bins = self.spike_count.shape[1]
        self.n_frames = self.stim.shape[0]
        self.length = int((self.n_frames - self.stim_win_len) * self.n_split) + 1

        print("Shape of spike count array:", self.spike_count.shape)

    def __len__(self):
        """Returns length of dataset"""
        return self.length

    def __getitem__(self, idx):
        r"""
        Returns idx_th chunk of stimulus and corresponding response

        Args:
            idx (int): index of data chunk

        Returns:
                stim: Stimuli matrix
                cnt: Spike counts
        """
        repeat_idx = np.random.randint(low=0, high=self.n_repeats)
        cnt_idx = idx + self.stim_win_len * self.n_split - 1
        cnt = self.spike_count[repeat_idx, (cnt_idx - self.cnt_win_len):cnt_idx + 1, :]
        rem = (idx % self.n_split)
        lf = math.floor(idx / self.n_split)
        stim = self.stim[lf:(lf + self.stim_win_len + 1), :, :].repeat(self.n_split, axis=0)
        stim = stim[rem:(self.n_split * self.stim_win_len) + rem, :, :]
        return cnt, stim