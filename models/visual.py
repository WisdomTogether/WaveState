import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import pandas as pd

def visualize_fused_features_3d(combined_features, sample_idx=0, channel_idx=0, label=None):
    sample = combined_features[sample_idx].cpu().detach().numpy()
    label = label[sample_idx].cpu().detach().numpy()
    num_channels, H, W = sample.shape

    if channel_idx >= num_channels:
        raise ValueError(f"Invalid channel_idx {channel_idx}, should be less than {num_channels}.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(W)
    Y = np.arange(H)
    X, Y = np.meshgrid(X, Y)
    Z = sample[channel_idx]

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f'Fused Feature Map (Sample {sample_idx}, Channel {channel_idx+1})')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Feature Dimension')
    ax.set_zlabel('Magnitude')

    if label is not None:
        ax.text2D(0.05, 0.95, f'Label: {label}', transform=ax.transAxes, fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig(f'combined_single_channel_label_{sample_idx}.png')
    plt.close(fig)

def visualize_conv_features_3d(combined_features, sample_idx=0, channel_idx=0, label=None):
    sample = combined_features[sample_idx].cpu().detach().numpy()
    label = label[sample_idx].cpu().detach().numpy()
    num_channels, H, W = sample.shape

    if channel_idx >= num_channels:
        raise ValueError(f"Invalid channel_idx {channel_idx}, should be less than {num_channels}.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X = np.arange(W)
    Y = np.arange(H)
    X, Y = np.meshgrid(X, Y)
    Z = sample[channel_idx]

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f'Conv Feature Map (Sample {sample_idx}, Channel {channel_idx+1})')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Feature Dimension')
    ax.set_zlabel('Magnitude')

    if label is not None:
        ax.text2D(0.05, 0.95, f'Label: {label}', transform=ax.transAxes, fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig(f'conv_single_channel_label_{sample_idx}.png')
    plt.close(fig)

def visualize_pssa_heatmap(pssa_features, sample_idx=0, title='PSSA Features Heatmap for All Channels'):
    feature_map = pssa_features[sample_idx].cpu().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(feature_map, aspect='auto', cmap='hot')
    
    plt.title(title)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Channels')
    
    plt.colorbar(label='Magnitude')
    
    plt.tight_layout()
    plt.savefig(f'pssa_features_all_channels_{sample_idx}.png')
    # plt.show()

def visualize_comatt_heatmap(pssa_features, sample_idx=0, title='ComAtt Features Heatmap for All Channels'):
    feature_map = pssa_features[sample_idx].cpu().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(feature_map, aspect='auto', cmap='hot')
    
    plt.title(title)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Channels')
    
    plt.colorbar(label='Magnitude')
    
    plt.tight_layout()
    plt.savefig(f'comatt_features_all_channels_{sample_idx}.png')
    # # plt.show()

def save_random_class_samples(conv_features, combined_features, labels, num_classes=4):
    unique_labels = np.unique(labels.cpu().numpy())
    
    selected_classes = random.sample(list(unique_labels), num_classes)

    for class_label in selected_classes:
        class_indices = np.where(labels.cpu().numpy() == class_label)[0]
        if class_indices.size == 0:
            continue

        sample_idx = random.choice(class_indices)

        visualize_fused_features_3d(combined_features, sample_idx=sample_idx, channel_idx=0, label=labels)
        
        visualize_conv_features_3d(conv_features, sample_idx=sample_idx, channel_idx=0, label=labels)

def vision_pssa_heatmap(input_sample, output_sample, sample_idx=0):
    input_sample = input_sample[sample_idx].cpu().detach().numpy()
    output_sample = output_sample[sample_idx].cpu().detach().numpy()

    time_steps = input_sample.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(min(1.5 * time_steps, 18), 12), gridspec_kw={'height_ratios': [1, 1]})

    sns.heatmap(input_sample.T, cmap='Blues', cbar=True, square=False, ax=ax1, cbar_kws={"orientation": "horizontal"})
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Feature Channels')

    sns.heatmap(output_sample.T, cmap='Greens', cbar=True, square=False, ax=ax2, cbar_kws={"orientation": "horizontal"})
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Feature Channels')

    plt.tight_layout()

    plt.savefig('PSSA_Heatmap_Long.svg', format='svg', bbox_inches='tight', dpi=300)
    # # plt.show()

def vision_coma_heatmap(input_sample, output_sample, sample_idx=0):
    input_sample = input_sample[sample_idx].cpu().detach().numpy()
    output_sample = output_sample[sample_idx].cpu().detach().numpy()

    time_steps = input_sample.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(min(1.5 * time_steps, 18), 12), gridspec_kw={'height_ratios': [1, 1]})

    sns.heatmap(input_sample.T, cmap='Blues', cbar=True, square=False, ax=ax1, cbar_kws={"orientation": "horizontal"})
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Feature Channels')

    sns.heatmap(output_sample.T, cmap='Greens', cbar=True, square=False, ax=ax2, cbar_kws={"orientation": "horizontal"})
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Feature Channels')

    plt.tight_layout()

    plt.savefig('CommonAtt_Heatmap_Long.svg', format='svg', bbox_inches='tight', dpi=300)
    # plt.show()

def vision_combined_heatmap(input_sample, pssa_sample, attn_sample, sample_idx=0):
    input_sample = input_sample[sample_idx].cpu().detach().numpy()
    pssa_sample = pssa_sample[sample_idx].cpu().detach().numpy()
    attn_sample = attn_sample[sample_idx].cpu().detach().numpy()

    time_steps = input_sample.shape[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(min(1.6 * time_steps, 20), 4), gridspec_kw={'width_ratios': [1, 1, 1]})

    sns.heatmap(input_sample.T, cmap='Blues', cbar=True, square=False, ax=ax1)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Feature Channels')

    sns.heatmap(pssa_sample.T, cmap='Greens', cbar=True, square=False, ax=ax2)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Feature Channels')

    sns.heatmap(attn_sample.T, cmap='Reds', cbar=True, square=False, ax=ax3)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Feature Channels')

    plt.tight_layout()

    plt.savefig('Attention_PSSA_Comparison_Heatmap_Eth.svg', format='svg', bbox_inches='tight', dpi=96)
    # plt.show()

def visualize_temporal_sparsity(input_tensor, output_tensor, ffn_tensor, batch_idx=0, title='Input vs Output Features'):
    input_tensor = input_tensor[batch_idx].cpu().detach().numpy()
    output_tensor = output_tensor[batch_idx].cpu().detach().numpy()
    ffn_tensor = ffn_tensor[batch_idx].cpu().detach().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    sns.heatmap(input_tensor.T, cmap='Blues', ax=ax1, cbar=True)
    ax1.set_title('Input Features Over Time')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Feature Channels')

    sns.heatmap(output_tensor.T, cmap='Reds', ax=ax2, cbar=True)
    ax2.set_title('Sparse FFN Features Over Time')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Feature Channels')

    sns.heatmap(ffn_tensor.T, cmap='Greens', ax=ax3, cbar=True)
    ax3.set_title('Common FFN Features Over Time')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Feature Channels')

    plt.tight_layout()
    plt.savefig('visualize_temporal_sparsity.svg', format='svg', bbox_inches='tight', dpi=96)
    # plt.show()

def visualize_sparse_weights(model, save_path='sparse_weights_analysis.png'):
    sparse_ffn = None
    for name, module in model.named_modules():
        if hasattr(module, 'FeedForward') and hasattr(module.FeedForward, 'linear1'):
            sparse_ffn = module.FeedForward
            break
    
    if sparse_ffn is None:
        print("No sparse FFN layer found")
        return None, None, None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    w1 = sparse_ffn.linear1.weight.data.cpu().numpy()
    w2 = sparse_ffn.linear2.weight.data.cpu().numpy()
    
    axes[0, 0].hist(w1.flatten(), bins=50, alpha=0.7, label='Layer 1')
    axes[0, 0].hist(w2.flatten(), bins=50, alpha=0.7, label='Layer 2')
    axes[0, 0].set_title('Original Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    mask1 = np.abs(w1) > 1e-6
    sparse_w1 = w1 * mask1
    im1 = axes[0, 1].imshow(sparse_w1, cmap='RdBu', aspect='auto')
    axes[0, 1].set_title(f'Layer 1 Sparse Weights (Sparsity: {(1-np.mean(mask1)):.2%})')
    plt.colorbar(im1, ax=axes[0, 1])
    
    mask2 = np.abs(w2) > 1e-6
    sparse_w2 = w2 * mask2
    im2 = axes[0, 2].imshow(sparse_w2, cmap='RdBu', aspect='auto')
    axes[0, 2].set_title(f'Layer 2 Sparse Weights (Sparsity: {(1-np.mean(mask2)):.2%})')
    plt.colorbar(im2, ax=axes[0, 2])
    
    axes[1, 0].imshow(mask1.astype(int), cmap='binary', aspect='auto')
    axes[1, 0].set_title('Layer 1 Sparsity Pattern (White=Active)')
    
    axes[1, 1].imshow(mask2.astype(int), cmap='binary', aspect='auto')
    axes[1, 1].set_title('Layer 2 Sparsity Pattern (White=Active)')
    
    total_params = w1.size + w2.size
    active_params = np.sum(mask1) + np.sum(mask2)
    sparsity_ratio = 1 - (active_params / total_params)
    
    stats_text = f"""
    Total Parameters: {total_params:,}
    Active Parameters: {int(active_params):,}
    Pruned Parameters: {int(total_params - active_params):,}
    Sparsity Ratio: {sparsity_ratio:.2%}
    
    Layer 1 Sparsity: {(1-np.mean(mask1)):.2%}
    Layer 2 Sparsity: {(1-np.mean(mask2)):.2%}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Parameter Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    return sparsity_ratio, active_params, total_params

def visualize_threshold_sensitivity(model, dataloader, device, tau_range=[0.1, 0.2, 0.3, 0.4, 0.5]):
    results = []
    
    for tau in tau_range:
        for module in model.modules():
            if hasattr(module, 'FeedForward'):
                apply_sparsity(module.FeedForward, tau)
        
        sparsity_ratio = calculate_sparsity(model)
        
        accuracy = evaluate_model_accuracy(model, dataloader, device)
        
        results.append({
            'tau': tau,
            'sparsity': sparsity_ratio,
            'accuracy': accuracy
        })
    
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(df['tau'], df['sparsity'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Threshold τ')
    ax1.set_ylabel('Sparsity Ratio')
    ax1.set_title('Sparsity vs Threshold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['tau'], df['accuracy'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Threshold τ')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return df

def apply_sparsity(ffn_module, tau):
    with torch.no_grad():
        for name, param in ffn_module.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param) >= tau
                param.data = param.data * mask.float()

def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'FeedForward' in name and 'weight' in name:
            total_params += param.numel()
            zero_params += (param.abs() < 1e-6).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0

def evaluate_model_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels, _ in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def visualize_training_sparsification(sparsity_history, accuracy_history, save_path='training_sparsification.png'):
    epochs = range(len(sparsity_history))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(epochs, sparsity_history, 'b-', linewidth=2, label='Sparsity Ratio')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sparsity Ratio')
    ax1.set_title('Sparsification Evolution During Training')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, accuracy_history, 'r-', linewidth=2, label='Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Evolution During Training')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def compare_dense_vs_sparse_ffn(dense_weights, sparse_weights, save_path='dense_vs_sparse_comparison.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(dense_weights, cmap='RdBu', aspect='auto')
    axes[0, 0].set_title('Dense FFN Weights')
    
    axes[0, 1].imshow(sparse_weights, cmap='RdBu', aspect='auto')
    axes[0, 1].set_title('Sparse FFN Weights')
    
    diff = np.abs(dense_weights - sparse_weights)
    axes[0, 2].imshow(diff, cmap='Reds', aspect='auto')
    axes[0, 2].set_title('Absolute Difference')
    
    axes[1, 0].hist(dense_weights.flatten(), bins=50, alpha=0.7, label='Dense')
    axes[1, 0].hist(sparse_weights.flatten(), bins=50, alpha=0.7, label='Sparse')
    axes[1, 0].set_title('Weight Distribution Comparison')
    axes[1, 0].legend()
    
    sparse_mask = np.abs(sparse_weights) > 1e-6
    axes[1, 1].imshow(sparse_mask.astype(int), cmap='binary', aspect='auto')
    axes[1, 1].set_title('Sparsity Pattern (White=Active)')
    
    dense_params = np.prod(dense_weights.shape)
    sparse_params = np.sum(sparse_mask)
    reduction = (dense_params - sparse_params) / dense_params
    
    stats_text = f"""
    Dense Parameters: {dense_params:,}
    Sparse Parameters: {sparse_params:,}
    Parameter Reduction: {reduction:.2%}
    Sparsity Ratio: {1-sparse_params/dense_params:.2%}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def generate_sparsity_summary_report(visualization_results, output_dir):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    summary_data = []
    for dataset, results in visualization_results.items():
        stats = results['param_stats']
        summary_data.append(stats)
    
    df = pd.DataFrame(summary_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    axes[0, 0].hist(df['parameter_reduction'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Parameter Reduction Distribution')
    axes[0, 0].set_xlabel('Reduction Ratio')
    axes[0, 0].set_ylabel('Number of Datasets')
    
    axes[0, 1].scatter(df['sparsity_ratio'], df['final_accuracy'], alpha=0.7, s=50)
    axes[0, 1].set_xlabel('Sparsity Ratio')
    axes[0, 1].set_ylabel('Final Accuracy')
    axes[0, 1].set_title('Sparsity vs Accuracy')
    
    axes[0, 2].bar(range(len(df)), df['active_params'], alpha=0.7, color='lightgreen')
    axes[0, 2].set_xlabel('Dataset Index')
    axes[0, 2].set_ylabel('Active Parameters')
    axes[0, 2].set_title('Active Parameters per Dataset')
    
    top_sparse = df.nlargest(10, 'sparsity_ratio')
    axes[1, 0].barh(range(len(top_sparse)), top_sparse['sparsity_ratio'])
    axes[1, 0].set_yticks(range(len(top_sparse)))
    axes[1, 0].set_yticklabels(top_sparse['dataset'], fontsize=8)
    axes[1, 0].set_xlabel('Sparsity Ratio')
    axes[1, 0].set_title('Top 10 Most Sparse Datasets')
    
    df['param_efficiency'] = df['final_accuracy'] / (df['active_params'] / 1000)
    axes[1, 1].scatter(df['active_params'], df['final_accuracy'], 
                      c=df['param_efficiency'], cmap='viridis', alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Active Parameters')
    axes[1, 1].set_ylabel('Final Accuracy')
    axes[1, 1].set_title('Parameter Efficiency (Color = Acc/1K Params)')
    
    summary_stats = f"""
    Dataset Count: {len(df)}
    Avg Sparsity: {df['sparsity_ratio'].mean():.2%}
    Avg Param Reduction: {df['parameter_reduction'].mean():.2%}
    Avg Accuracy: {df['final_accuracy'].mean():.4f}
    
    Min Sparsity: {df['sparsity_ratio'].min():.2%}
    Max Sparsity: {df['sparsity_ratio'].max():.2%}
    
    Total Params Saved: {(df['total_params'] - df['active_params']).sum():,}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_stats, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Summary Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparsity_analysis_summary.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    df.to_csv(f'{output_dir}/sparsity_analysis_detailed.csv', index=False)
    
    print(f"Sparsity analysis report saved to: {output_dir}")