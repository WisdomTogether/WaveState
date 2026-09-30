import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

try:
    from pytorch_wavelets import DWT1D
except ImportError:
    try:
        import pywt
        DWT1D = None
    except ImportError:
        raise ImportError("Please install pytorch_wavelets or pywt: pip install pytorch_wavelets pywt")

sys.path.append('.')
from Dataset import load_UEA_data

class AdaptiveWaveletVisualization:
    def __init__(self, wavelet='db4', max_level=3, num_classes=10, class_to_levels=None):
        self.max_level = max_level
        self.num_classes = num_classes
        self.wavelet = wavelet
        
        if DWT1D is not None:
            self.dwt = DWT1D(wave=wavelet, mode='zero')
            self.use_pytorch_wavelets = True
        else:
            import pywt
            self.pywt = pywt
            self.use_pytorch_wavelets = False
        
        if class_to_levels is None:
            class_to_levels = torch.randint(1, max_level + 1, (num_classes,))
        else:
            class_to_levels = torch.tensor(class_to_levels, dtype=torch.long)
        
        self.class_to_levels = class_to_levels

    def decompose_signal(self, x, label):
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        x = x.float()
        
        channels, seq_len = x.shape[1], x.shape[2]
        decomposition_results = []
        
        for c in range(channels):
            predicted_level = self.class_to_levels[label].item()
            
            coeffs_low = []
            coeffs_high = []
            
            if self.use_pytorch_wavelets:
                data = x[0, c, :].unsqueeze(0).unsqueeze(0).float()
                current_data = data
                
                for level in range(predicted_level):
                    try:
                        cA, cD = self.dwt(current_data)
                        if isinstance(cA, torch.Tensor):
                            coeffs_low.append(cA.squeeze())
                        else:
                            coeffs_low.append(torch.tensor(cA).squeeze())
                        
                        if isinstance(cD, torch.Tensor):
                            coeffs_high.append(cD.squeeze())
                        else:
                            coeffs_high.append(torch.tensor(cD).squeeze())
                        
                        current_data = cA if isinstance(cA, torch.Tensor) else torch.tensor(cA).unsqueeze(0).unsqueeze(0)
                    except Exception as e:
                        print(f"pytorch_wavelets error: {e}, falling back to pywt")
                        self.use_pytorch_wavelets = False
                        break
                
                if not self.use_pytorch_wavelets:
                    import pywt
                    self.pywt = pywt
                    coeffs_low = []
                    coeffs_high = []
            
            if not self.use_pytorch_wavelets:
                data = x[0, c, :].numpy()
                current_data = data
                
                for level in range(predicted_level):
                    try:
                        cA, cD = self.pywt.dwt(current_data, self.wavelet, mode='zero')
                        coeffs_low.append(torch.tensor(cA, dtype=torch.float32))
                        coeffs_high.append(torch.tensor(cD, dtype=torch.float32))
                        current_data = cA
                    except Exception as e:
                        print(f"pywt error: {e}")
                        coeffs_low.append(torch.tensor(current_data[:len(current_data)//2], dtype=torch.float32))
                        coeffs_high.append(torch.tensor(current_data[len(current_data)//2:], dtype=torch.float32))
                        current_data = current_data[:len(current_data)//2]
            
            decomposition_results.append({
                'low_freq': coeffs_low,
                'high_freq': coeffs_high,
                'original': x[0, c, :].squeeze()
            })
        
        return decomposition_results

    def plot_single_sample_decomposition(self, x, label, sample_idx=0, save_path=None):
        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy()
        
        if len(x.shape) == 3:
            signal = torch.tensor(x[sample_idx], dtype=torch.float32)
            sample_label = label[sample_idx]
        else:
            signal = torch.tensor(x, dtype=torch.float32)
            sample_label = label
        
        decomp_results = self.decompose_signal(signal, sample_label)
        
        n_channels = len(decomp_results)
        n_levels = len(decomp_results[0]['low_freq']) if decomp_results[0]['low_freq'] else 0
        
        if n_levels == 0:
            print("No decomposition levels found")
            return
        
        fig, axes = plt.subplots(3, n_channels, figsize=(5 * n_channels, 12))
        
        if n_channels == 1:
            axes = axes.reshape(-1, 1)
        
        colors_low = plt.cm.Blues(np.linspace(0.5, 0.9, n_levels))
        colors_high = plt.cm.Reds(np.linspace(0.5, 0.9, n_levels))
        
        for ch in range(n_channels):
            result = decomp_results[ch]
            
            original = result['original']
            if isinstance(original, torch.Tensor):
                original = original.numpy()
            
            axes[0, ch].plot(original, 'k-', linewidth=2)
            axes[0, ch].set_title(f'Original Time Series - Channel {ch+1}', fontsize=14, fontweight='bold')
            axes[0, ch].grid(True, alpha=0.3)
            axes[0, ch].set_ylabel('Amplitude', fontsize=12)
            axes[0, ch].set_xlabel('Time Steps', fontsize=12)
            
            for level in range(n_levels):
                low_freq = result['low_freq'][level]
                high_freq = result['high_freq'][level]
                
                if isinstance(low_freq, torch.Tensor):
                    low_freq = low_freq.numpy()
                if isinstance(high_freq, torch.Tensor):
                    high_freq = high_freq.numpy()
                
                time_steps_low = np.linspace(0, len(original), len(low_freq))
                axes[1, ch].plot(time_steps_low, low_freq, 
                               color=colors_low[level], linewidth=1.8, 
                               label=f'Level {level+1}', alpha=0.8)
                
                time_steps_high = np.linspace(0, len(original), len(high_freq))
                axes[2, ch].plot(time_steps_high, high_freq, 
                               color=colors_high[level], linewidth=1.8, 
                               label=f'Level {level+1}', alpha=0.8)
            
            axes[1, ch].set_title(f'Low Frequency Components - Channel {ch+1}', fontsize=14, fontweight='bold')
            axes[1, ch].grid(True, alpha=0.3)
            axes[1, ch].set_ylabel('Amplitude', fontsize=12)
            axes[1, ch].set_xlabel('Time Steps', fontsize=12)
            axes[1, ch].legend(fontsize=10)
            
            axes[2, ch].set_title(f'High Frequency Components - Channel {ch+1}', fontsize=14, fontweight='bold')
            axes[2, ch].grid(True, alpha=0.3)
            axes[2, ch].set_ylabel('Amplitude', fontsize=12)
            axes[2, ch].set_xlabel('Time Steps', fontsize=12)
            axes[2, ch].legend(fontsize=10)
        
        plt.suptitle(f'Class-Adaptive Wavelet Transform - Sample {sample_idx} (Class {sample_label})\n'
                    f'Decomposition Levels: {self.class_to_levels[sample_label].item()}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def load_basicaction_data():
    config = {
        'data_dir': 'Dataset/UEA/Multivariate_ts/BasicMotions',
        'Norm': False,
        'val_ratio': 0.2
    }
    
    if not os.path.exists(config['data_dir']):
        config['data_dir'] = 'Dataset/UEA/Multivariate_ts/BasicAction'
    
    if not os.path.exists(config['data_dir']):
        print("BasicAction dataset not found. Creating sample data...")
        return generate_sample_data()
    
    try:
        data = load_UEA_data.load(config)
        train_data = data['train_data']
        train_labels = data['train_label']
        
        print(f"Loaded BasicAction dataset:")
        print(f"Train data shape: {train_data.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Number of classes: {len(np.unique(train_labels))}")
        print(f"Classes: {np.unique(train_labels)}")
        
        return train_data, train_labels
    except Exception as e:
        print(f"Error loading BasicAction dataset: {e}")
        print("Using sample data instead...")
        return generate_sample_data()

def generate_sample_data(n_samples=20, n_channels=6, seq_length=100, n_classes=4):
    np.random.seed(42)
    torch.manual_seed(42)
    
    data = []
    labels = []
    
    for i in range(n_samples):
        label = i % n_classes
        
        if label == 0:
            signal = np.sin(2 * np.pi * 0.1 * np.arange(seq_length)) + \
                    0.5 * np.random.randn(seq_length)
        elif label == 1:
            signal = np.sin(2 * np.pi * 0.3 * np.arange(seq_length)) + \
                    0.3 * np.sin(2 * np.pi * 0.05 * np.arange(seq_length)) + \
                    0.3 * np.random.randn(seq_length)
        elif label == 2:
            t = np.arange(seq_length)
            signal = np.exp(-t/50) * np.sin(2 * np.pi * 0.2 * t) + \
                    0.2 * np.random.randn(seq_length)
        else:
            signal = np.sin(2 * np.pi * 0.05 * np.arange(seq_length)) + \
                    np.sin(2 * np.pi * 0.4 * np.arange(seq_length)) + \
                    0.4 * np.random.randn(seq_length)
        
        multi_channel_signal = np.tile(signal, (n_channels, 1))
        for ch in range(n_channels):
            multi_channel_signal[ch] += 0.1 * np.random.randn(seq_length)
        
        data.append(multi_channel_signal)
        labels.append(label)
    
    return np.array(data, dtype=np.float32), np.array(labels)

if __name__ == "__main__":
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("加载BasicAction数据集...")
    data, labels = load_basicaction_data()
    
    n_classes = len(np.unique(labels))
    print(f"数据集信息: {data.shape}, 类别数: {n_classes}")
    
    print("初始化类别自适应小波变换可视化器...")
    if n_classes <= 3:
        class_to_levels = [1, 2, 3][:n_classes]
    else:
        class_to_levels = [1, 2, 3, 2, 3][:n_classes]
    
    visualizer = AdaptiveWaveletVisualization(
        wavelet='db4', 
        max_level=3, 
        num_classes=n_classes, 
        class_to_levels=class_to_levels
    )
    
    print(f"使用的小波变换库: {'pytorch_wavelets' if visualizer.use_pytorch_wavelets else 'pywt'}")
    
    sample_idx = 0
    print(f"绘制样本 {sample_idx} (类别 {labels[sample_idx]}) 的小波分解图...")
    
    visualizer.plot_single_sample_decomposition(
        data, labels, sample_idx=sample_idx,
        save_path=f'cawt_decomposition_sample_{sample_idx}.svg'
    )
    
    print("可视化完成！")