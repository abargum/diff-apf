import torch
import torchaudio
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
import numpy as np

"""STFT and Multi Resolution STFT Loss Modules.
    See [Steinmetz & Reiss, 2020]('auraloss: Audio-focused loss functions in PyTorch')"""
class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        eps=1e-8,
    ):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.eps = eps
        self.loss = torch.nn.MSELoss(reduction='mean')

    def stft(self, x):
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=self.eps)
        )
        
        return x_mag[:, :(self.fft_size//8), :]
    
    def forward(self, x, y):
        self.window = self.window.to(x.device)
        
        x_mag = self.stft(x.view(-1, x.size(-1)))
        y_mag = self.stft(y.view(-1, y.size(-1)))

        mag_1 = self.stft(x.view(-1, x.size(-1)) + y.view(-1, y.size(-1)))
        mag_2 = x_mag + y_mag
        
        stft_loss = self.loss(mag_2, mag_1)
        loss = stft_loss
        return loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 512, 2048],
        hop_sizes=[120, 50, 240],
        win_lengths=[600, 240, 1200],
        #fft_sizes=[128, 512, 256],
        #hop_sizes=[13, 50, 25],
        #win_lengths=[60, 240, 120],
        window="hann_window",
        **kwargs,
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        mrstft_loss = 0.0

        for f in self.stft_losses:
                mrstft_loss += f(x, y)
        mrstft_loss /= len(self.stft_losses)
        
        return mrstft_loss
    
"""Cross Correlation Loss.
    See [Hiasa, Otake, Takao etc., 2018](https://arxiv.org/abs/1803.06629)"""
def normalized_cross_correlation(x, y, reduction='mean', eps=1e-8):
    shape = x.shape
    b = shape[0]
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)
    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)
    return ncc

class NormalizedCrossCorrelation(torch.nn.Module):
    def __init__(self, eps=1e-8, reduction='mean'):
        super(NormalizedCrossCorrelation, self).__init__()
        self._eps = eps
        self._reduction = reduction

    def forward(self, x, y):
        return normalized_cross_correlation(x, y, self._reduction, self._eps)

class NormalizedCrossCorrelationLoss(NormalizedCrossCorrelation):
    def forward(self, x, y):
        gc = super().forward(x, y)
        #Maximize the loss function to take account of minimization-nature
        return (1.0 - gc) * -1.0
    
"""Phase-Loss (not working as intended)"""
class PhaseLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=64,
        win_length=1024,
        window="hann_window",
    ):
        super(PhaseLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.loss = torch.nn.L1Loss(reduction='mean')
        
    def unwrap(self, phi, dim=1):
        dphi = torch.diff(phi, dim=dim)
        dphi = F.pad(dphi, (0, 0, 1, 0))
        
        dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi
        dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi
        phi_adj = dphi_m - dphi
        phi_adj[dphi.abs() < np.pi] = 0
        
        return phi + torch.cumsum(phi_adj, dim=dim)

    def stft(self, x):
        x_stft = torch.stft(
                 x,
                 self.fft_size,
                 self.hop_size,
                 self.win_length,
                 self.window,
                 return_complex=True)
        
        x_phs = x_stft[:, :(self.fft_size//8), :]
        x_phs = self.unwrap(torch.angle(x_phs))
        return x_phs
    
    def forward(self, x, y):
        self.window = self.window.to(x.device)
        x_phs = self.stft(x.view(-1, x.size(-1)))
        y_phs = self.stft(y.view(-1, y.size(-1)))
        phase_loss = self.loss(x_phs, y_phs)
        loss = phase_loss
        return loss
    
"""Final loss module for loss definition and training forward call"""
class LossModule(torch.nn.Module):
    def __init__(self, loss_term='MSE') -> None:
        super(LossModule, self).__init__()
        
        self.loss_term = loss_term
        
        self.loss_function_mse = torch.nn.MSELoss(reduction='mean')
        self.loss_function_stft = STFTLoss()
        self.loss_function_mstft = MultiResolutionSTFTLoss()
        self.loss_function_crosscorr = NormalizedCrossCorrelationLoss()
        self.loss_function_phase = PhaseLoss()

    def forward(self, true : Tensor, pred : Tensor) -> Tensor:
        if self.loss_term == 'STFT':
            loss = self.loss_function_stft(true.squeeze(-1), pred.squeeze(-1))
        elif self.loss_term == 'M-STFT':
            loss = self.loss_function_mstft(true.squeeze(-1), pred.squeeze(-1))
        elif self.loss_term == 'CROSSCORR':
            loss = self.loss_function_crosscorr(true.squeeze(-1), pred.squeeze(-1))
        elif self.loss_term == 'PHASE':
            loss = self.loss_function_phase(true.squeeze(-1), pred.squeeze(-1))
        elif self.loss_term == 'MSE':
            loss = self.loss_function_mse(true.squeeze(-1), pred.squeeze(-1))
        else:
            print("Please choose a valid loss function")
        
        return loss