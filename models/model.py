import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from timm.models.layers import to_2tuple
from models.transformer import Transformer
from models.convtran import ConvTran, CasualConvTran
from models.pssa_block import PSSABlock
from models.Attention import SimpleAttention

from models.visual import visualize_fused_features_3d, visualize_conv_features_3d, save_random_class_samples, visualize_pssa_heatmap, visualize_comatt_heatmap, vision_coma_heatmap, vision_pssa_heatmap, vision_combined_heatmap, visualize_temporal_sparsity
from pytorch_wavelets import DWT1D

from models.visual_tsne import plot_dwt_tsne, plot_ori_tsne, compute_average_std, compute_feature_std_over_time, visualize_std_comparison

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)

def model_factory(config):
    if config['Net_Type'][0] == 'MT':
        model = WaveState(config, c_in=8, d_model=8, window_size=to_2tuple(1), num_heads=8,
                            num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'CC-T':
        model = CasualConvTran(config, num_classes=config['num_labels'])
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    return model

class AdaptiveWaveletTransform(nn.Module):
    def __init__(self, wavelet='db4', max_level=3, energy_threshold=0.95):
        super(AdaptiveWaveletTransform, self).__init__()
        self.max_level = max_level
        self.energy_threshold = energy_threshold

        self.dwt = DWT1D(wave=wavelet, mode='zero')
        self.min_len = getattr(pywt.Wavelet(wavelet), 'dec_len', 4)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape

        if len(x.shape) != 3:
            raise ValueError(f"Input tensor x should be 3D, but got {x.shape} instead.")

        approximations = [x]
        retained = [torch.sum(x.detach() ** 2, dim=-1)]
        approx = x

        for level in range(self.max_level):
            if approx.shape[-1] < self.min_len * 2:
                break
            cA, cD = self.dwt(approx)
            approximations.append(cA)
            detail_energy = sum(torch.sum(d.detach() ** 2, dim=-1) for d in cD)
            retained.append(retained[-1] - detail_energy)
            approx = cA

        if len(approximations) == 1:
            return x

        ratios = torch.stack([r / (retained[0] + 1e-8) for r in retained[1:]], dim=-1)
        active = ratios >= self.energy_threshold
        first_valid = torch.argmax(active.to(torch.uint8), dim=-1)
        has_valid = active.any(dim=-1)
        level_choice = torch.where(has_valid, first_valid + 1,
                                   torch.full_like(first_valid, len(approximations) - 1)).cpu()

        rows = []
        for b in range(batch_size):
            row = []
            for c in range(channels):
                row.append(approximations[level_choice[b, c].item()][b, c].reshape(-1))
            rows.append(row)

        out_len = max(t.shape[0] for row in rows for t in row)
        transformed = torch.stack([
            torch.stack([F.pad(t.float(), (0, out_len - t.shape[0])) for t in row])
            for row in rows
        ])

        return transformed

class SparseFFN(nn.Module):
    def __init__(self, c_in, drop=0.1, sparsity=0.5):
        super(SparseFFN, self).__init__()
        self.c_in = c_in
        self.drop = drop
        self.sparsity = sparsity

        self.linear1 = nn.Linear(c_in, 1024)
        self.linear2 = nn.Linear(1024, c_in)

        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

        self.register_buffer('original_weight1', self.linear1.weight.data.clone())
        self.register_buffer('original_weight2', self.linear2.weight.data.clone())
        
        self.register_buffer('sparsity_mask1', None)
        self.register_buffer('sparsity_mask2', None)
        
        self.apply_sparsity()

    def apply_sparsity(self):
        weight_mask1 = torch.bernoulli(torch.full(self.linear1.weight.shape, 1 - self.sparsity)).to(self.linear1.weight.device)
        weight_mask2 = torch.bernoulli(torch.full(self.linear2.weight.shape, 1 - self.sparsity)).to(self.linear2.weight.device)

        self.sparsity_mask1 = weight_mask1
        self.sparsity_mask2 = weight_mask2

        self.linear1.weight.data *= weight_mask1
        self.linear2.weight.data *= weight_mask2
        
        actual_sparsity1 = 1 - (weight_mask1.sum().float() / weight_mask1.numel())
        actual_sparsity2 = 1 - (weight_mask2.sum().float() / weight_mask2.numel())
        
        print(f"Layer 1 - Target sparsity: {self.sparsity:.2%}, Actual sparsity: {actual_sparsity1:.2%}")
        print(f"Layer 2 - Target sparsity: {self.sparsity:.2%}, Actual sparsity: {actual_sparsity2:.2%}")

    def get_sparsity_info(self):
        total_params = self.linear1.weight.numel() + self.linear2.weight.numel()
        zero_params1 = (self.linear1.weight.abs() < 1e-6).sum().item()
        zero_params2 = (self.linear2.weight.abs() < 1e-6).sum().item()
        total_zero = zero_params1 + zero_params2
        
        return {
            'total_params': total_params,
            'zero_params': total_zero,
            'active_params': total_params - total_zero,
            'sparsity_ratio': total_zero / total_params,
            'layer1_sparsity': zero_params1 / self.linear1.weight.numel(),
            'layer2_sparsity': zero_params2 / self.linear2.weight.numel(),
        }

    def forward(self, x):
        x = F.relu(self.linear1(self.dropout1(x)))
        x = self.linear2(self.dropout2(x))
        return x

class FFN(nn.Module):
    def __init__(self, c_in, drop=0.1):
        super(FFN, self).__init__()
        self.c_in = c_in
        self.drop = drop

        self.linear1 = nn.Linear(c_in, 1024)
        self.linear2 = nn.Linear(1024, c_in)

        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):
        x = F.relu(self.linear1(self.dropout1(x)))
        x = self.linear2(self.dropout2(x))
        return x

class WaveState(nn.Module):
    def __init__(self, config, c_in, num_heads, num_classes, dropout=0.1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        channel_size = config['Data_shape'][1]

        self.adaptive_wavelet_transform = AdaptiveWaveletTransform(
            wavelet='db4',
            max_level=3,
            energy_threshold=0.95
        )

        self.feature1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, c_in * 4, kernel_size=[1, 3], padding='same'),
                nn.BatchNorm2d(c_in * 4),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(1, c_in * 4, kernel_size=[1, 5], padding='same'),
                nn.BatchNorm2d(c_in * 4),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(1, c_in * 4, kernel_size=[1, 7], padding='same'),
                nn.BatchNorm2d(c_in * 4),
                nn.GELU()
            )
        ])

        self.reduce = nn.Conv2d(c_in * 12, c_in * 4, kernel_size=1)

        self.feature2 = nn.Sequential(
            nn.Conv2d(c_in * 4, c_in, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(c_in),
            nn.GELU()
        )

        self.LayerNorm = nn.LayerNorm(c_in, eps=1e-5)

        self.tmamba = PSSABlock(dim=8, num_heads=2)

        self.LayerNorm2 = nn.LayerNorm(c_in, eps=1e-5)

        self.FeedForward = SparseFFN(c_in=c_in, sparsity=0.5, drop=dropout)
        # self.FeedForward = FFN(c_in=c_in, drop=dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(c_in, num_classes)

    def forward(self, x, labels=None, mask=None):
        # plot_ori_tsne(x.view(x.shape[0], -1), labels=labels)
        x = x.unsqueeze(1)
        original_features = x.reshape(x.shape[0], -1).detach()

        feature1_outs = [branch(x) for branch in self.feature1]
        time_domain_features = torch.cat(feature1_outs, dim=1)
        time_domain_features = self.reduce(time_domain_features)

        freq_domain_features = self.adaptive_wavelet_transform(x.squeeze(1))

        freq_domain_features = freq_domain_features.squeeze(1)

        freq_domain_features = freq_domain_features.squeeze(2).squeeze(2)

        # plot_dwt_tsne(freq_domain_features.view(x.shape[0], -1), labels=labels)

        freq_domain_features_resized = F.interpolate(
            freq_domain_features, size=time_domain_features.shape[-1], mode='linear'
        )

        freq_domain_features_expanded = freq_domain_features_resized.unsqueeze(1)
        freq_domain_features_expanded = freq_domain_features_expanded.expand(-1, time_domain_features.size(1), -1,
                                                                             -1)

        combined_features = time_domain_features + freq_domain_features_expanded

        x = self.feature2(combined_features)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        x_pssa = self.tmamba(x)

        x = x + x_pssa
        x = self.LayerNorm(x)

        x = x + self.FeedForward(x)
        x = self.LayerNorm2(x)
        x = x.permute(0, 2, 1)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.out(x)
        return x