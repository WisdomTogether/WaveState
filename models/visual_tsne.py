from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F


def preprocess_features(original_features):
    """
    预处理特征数据
    - 标准化
    - 可选的维度归一化
    """
    if not torch.is_tensor(original_features):
        original_features = torch.tensor(original_features, dtype=torch.float32)

    # 标准化
    features_mean = torch.mean(original_features, dim=0)
    features_std = torch.std(original_features, dim=0)
    normalized_features = (original_features - features_mean) / (features_std + 1e-8)

    # 维度归一化 (L2 norm)
    normalized_features = F.normalize(normalized_features, p=2, dim=1)

    return normalized_features


def compute_improved_intra_class_compactness(tsne_result, labels):
    """
    计算改进的类内紧密度
    使用归一化的欧氏距离
    """
    unique_labels = np.unique(labels)
    compactness_per_class = []

    for label in unique_labels:
        class_points = tsne_result[labels == label]
        if len(class_points) > 1:  # 确保类别中至少有两个点
            class_centroid = np.mean(class_points, axis=0)
            # 计算到中心点的欧氏距离
            distances = np.linalg.norm(class_points - class_centroid, axis=1)
            # 归一化距离到 [0,1] 范围
            if np.max(distances) - np.min(distances) > 0:
                distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            compactness_per_class.append(np.mean(distances))

    return np.mean(compactness_per_class) if compactness_per_class else 0.0


def compute_improved_inter_class_separation(tsne_result, labels):
    """
    计算改进的类间分离度
    使用归一化的欧氏距离
    """
    unique_labels = np.unique(labels)
    centroids = []

    # 计算每个类的中心点
    for label in unique_labels:
        class_points = tsne_result[labels == label]
        class_centroid = np.mean(class_points, axis=0)
        centroids.append(class_centroid)

    centroids = np.array(centroids)
    # 计算类别中心之间的欧氏距离
    inter_class_distances = pairwise_distances(centroids, metric='euclidean')

    # 归一化距离到 [0,1] 范围
    if np.max(inter_class_distances) - np.min(inter_class_distances) > 0:
        inter_class_distances = (inter_class_distances - np.min(inter_class_distances)) / (
                np.max(inter_class_distances) - np.min(inter_class_distances))

    # 排除对角线上的自身距离
    mask = ~np.eye(len(centroids), dtype=bool)
    mean_distance = np.mean(inter_class_distances[mask])
    min_distance = np.min(inter_class_distances[mask]) if len(centroids) > 1 else 0.0

    return mean_distance, min_distance


def compute_fisher_score(tsne_result, labels):
    """
    计算Fisher判别指数
    Fisher判别指数 = 类间散度 / 类内散度
    较大的Fisher判别指数表示更好的类别可分性
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # 计算总体均值
    global_mean = np.mean(tsne_result, axis=0)

    # 初始化类间散度和类内散度
    between_class_scatter = 0
    within_class_scatter = 0

    # 计算每个类的统计量
    for label in unique_labels:
        class_points = tsne_result[labels == label]
        n_samples = len(class_points)

        # 计算类均值
        class_mean = np.mean(class_points, axis=0)

        # 更新类间散度
        mean_diff = class_mean - global_mean
        between_class_scatter += n_samples * np.sum(mean_diff ** 2)

        # 更新类内散度
        class_scatter = np.sum((class_points - class_mean) ** 2)
        within_class_scatter += class_scatter

    # 避免除零
    if within_class_scatter < 1e-10:
        within_class_scatter = 1e-10

    fisher_score = between_class_scatter / within_class_scatter

    # 归一化Fisher分数到[0,1]范围
    fisher_score = 1 / (1 + np.exp(-fisher_score))

    return fisher_score


def compute_cluster_separation_index(tsne_result, labels):
    """
    计算聚类分离指数
    综合考虑类内紧密度和类间分离度
    """
    intra_class = compute_improved_intra_class_compactness(tsne_result, labels)
    inter_class_mean, _ = compute_improved_inter_class_separation(tsne_result, labels)

    # 计算分离指数 (较大值表示更好的聚类效果)
    separation_index = inter_class_mean / (intra_class + 1e-10)

    # 归一化到[0,1]范围
    separation_index = 1 / (1 + np.exp(-separation_index))

    return separation_index


def plot_enhanced_tsne(original_features, labels, title, save_name, perplexity=None):
    """
    增强的t-SNE可视化函数
    支持自动寻找最佳困惑度参数
    """
    # 数据预处理
    if torch.is_tensor(original_features):
        features = preprocess_features(original_features)
        features = features.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
    else:
        features = preprocess_features(torch.tensor(original_features))
        features = features.numpy()

    # 如果没有指定困惑度，使用默认值
    if perplexity is None:
        perplexity = 30

    # 使用改进的t-SNE参数
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12,
        learning_rate='auto',
        n_iter=1000,
        random_state=42
    )
    tsne_result = tsne.fit_transform(features)

    # 计算所有评估指标
    intra_class_compactness = compute_improved_intra_class_compactness(tsne_result, labels)
    inter_class_mean, inter_class_min = compute_improved_inter_class_separation(tsne_result, labels)
    fisher_score = compute_fisher_score(tsne_result, labels)
    separation_index = compute_cluster_separation_index(tsne_result, labels)

    # 创建增强的可视化
    plt.figure(figsize=(6, 4))

    # 使用更好的配色方案并添加透明度效果
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                          c=labels, cmap='tab20', s=50, alpha=0.6)

    # 添加点的轮廓效果
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                c=labels, cmap='tab20', s=50, alpha=0.1)

    # 将图例放在右下角，调整字体和大小
    legend = plt.legend(*scatter.legend_elements(),
                        title="Classes",
                        loc="lower left",
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        bbox_to_anchor=(0.01, 0.01),
                        prop={'family': 'serif', 'size': 9})

    # 设置图例标题的字体和大小
    legend.get_title().set_fontsize(9)
    legend.get_title().set_family('serif')

    # 添加网格背景
    plt.grid(True, linestyle='--', alpha=0.3)

    # 设置全局字体为 Serif
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9  # 增加默认字号

    # 更新度量值显示
    metrics_text = (
        f'Metrics:\n'
        f'Intra-Class Compactness: {intra_class_compactness:.3f}\n'
        f'Inter-Class Mean Distance: {inter_class_mean:.3f}\n'
        f'Inter-Class Min Distance: {inter_class_min:.3f}\n'
        f'Fisher Score: {fisher_score:.3f}\n'
        f'Separation Index: {separation_index:.3f}\n'
        f'Perplexity: {perplexity}'
    )

    # 调整文本框位置和样式
    plt.text(0.98, 0.98,
             metrics_text,
             transform=plt.gca().transAxes,
             fontsize=9,  # 增加字号
             family='serif',
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.5",
                       edgecolor="gray",
                       facecolor="white",
                       alpha=0.3))

    # 设置标题
    # plt.title(title, pad=20, size=16, weight='bold', family='serif')

    # 设置坐标轴标签字体和大小
    plt.xticks(fontsize=9, family='serif')
    plt.yticks(fontsize=9, family='serif')

    plt.tight_layout()
    plt.savefig(save_name, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

    return {
        'intra_class_compactness': intra_class_compactness,
        'inter_class_mean': inter_class_mean,
        'inter_class_min': inter_class_min,
        'fisher_score': fisher_score,
        'separation_index': separation_index
    }


# 使用示例：
def plot_ori_tsne(original_features, labels):
    return plot_enhanced_tsne(
        original_features,
        labels,
        title='Original Features t-SNE',
        save_name='visual_ori_tsne.pdf'
    )


def plot_dwt_tsne(original_features, labels):
    return plot_enhanced_tsne(
        original_features,
        labels,
        title='DWT Features t-SNE',
        save_name='visual_dwt_tsne.pdf'
    )


def validate_cawt_effectiveness(original_features, cawt_features, labels):
    """
    验证CAWT特征的有效性，通过对比原始特征和CAWT特征的t-SNE可视化及相关指标

    Args:
        original_features: 原始特征，numpy数组或PyTorch张量
        cawt_features: CAWT转换后的特征，numpy数组或PyTorch张量
        labels: 类别标签，numpy数组或PyTorch张量
    """
    # 1. 可视化原始特征的t-SNE
    print("正在进行原始特征的t-SNE可视化...")
    ori_metrics = plot_ori_tsne(original_features, labels)

    # 2. 可视化CAWT特征的t-SNE
    print("\n正在进行CAWT特征的t-SNE可视化...")
    cawt_metrics = plot_dwt_tsne(cawt_features, labels)

    # 3. 打印并对比指标
    print("\n=== 特征表示能力对比分析 ===")
    print("\n类内紧密度分析(越小越好):")
    print(f"原始特征: {ori_metrics['intra_class_compactness']:.3f}")
    print(f"CAWT特征: {cawt_metrics['intra_class_compactness']:.3f}")
    improvement = ((ori_metrics['intra_class_compactness'] - cawt_metrics['intra_class_compactness'])
                   / ori_metrics['intra_class_compactness'] * 100)
    print(f"改善程度: {abs(improvement):.1f}%")

    print("\n类间平均距离分析(越大越好):")
    print(f"原始特征: {ori_metrics['inter_class_mean']:.3f}")
    print(f"CAWT特征: {cawt_metrics['inter_class_mean']:.3f}")
    improvement = ((cawt_metrics['inter_class_mean'] - ori_metrics['inter_class_mean'])
                   / ori_metrics['inter_class_mean'] * 100)
    print(f"改善程度: {improvement:.1f}%")

    print("\n类间最小距离分析(越大越好):")
    print(f"原始特征: {ori_metrics['inter_class_min']:.3f}")
    print(f"CAWT特征: {cawt_metrics['inter_class_min']:.3f}")
    improvement = ((cawt_metrics['inter_class_min'] - ori_metrics['inter_class_min'])
                   / ori_metrics['inter_class_min'] * 100)
    print(f"改善程度: {improvement:.1f}%")

    print("\n轮廓系数分析(越大越好):")
    print(f"原始特征: {ori_metrics['silhouette_score']:.3f}")
    print(f"CAWT特征: {cawt_metrics['silhouette_score']:.3f}")
    improvement = ((cawt_metrics['silhouette_score'] - ori_metrics['silhouette_score'])
                   / ori_metrics['silhouette_score'] * 100)
    print(f"改善程度: {improvement:.1f}%")

    return {
        'original_metrics': ori_metrics,
        'cawt_metrics': cawt_metrics
    }

def compute_feature_std_over_time(input_tensor, output_tensor, com_tensor, batch_idx=0):
    # 提取当前 batch 的数据
    input_tensor = input_tensor[batch_idx].cpu().detach().numpy()
    output_tensor = output_tensor[batch_idx].cpu().detach().numpy()
    com_tensor = com_tensor[batch_idx].cpu().detach().numpy()

    # 对每个特征通道，计算其在时间步上的标准差
    input_std = np.std(input_tensor, axis=0)  # 输入特征的标准差
    output_std = np.std(output_tensor, axis=0)  # 输出特征的标准差
    com_std = np.std(com_tensor, axis=0)  # 输出特征的标准差

    # 返回标准差结果
    return input_std, output_std, com_std

def visualize_std_comparison(input_std, output_std, com_std, title='Standard Deviation Comparison'):

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制输入和输出的标准差对比
    plt.plot(input_std, label='Input Features', color='blue', marker='o')
    plt.plot(output_std, label='Output Features (After PSSA Block)', color='green', marker='x')
    plt.plot(com_std, label='Output Features (After Common Attention)', color='red', marker='v')

    plt.title(title)
    plt.xlabel('Feature Channels')
    plt.ylabel('Standard Deviation')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.savefig('std_comparison.svg', format='svg', bbox_inches='tight', dpi=96)
    plt.show()

def compute_average_std(input_std, output_std, com_std):
    # 如果 input_std 和 output_std 是 torch 张量，将其转换为 numpy 数组
    if isinstance(input_std, torch.Tensor):
        input_std = input_std.cpu().detach().numpy()
    if isinstance(output_std, torch.Tensor):
        output_std = output_std.cpu().detach().numpy()
    if isinstance(com_std, torch.Tensor):
        output_std = com_std.cpu().detach().numpy()

    # 计算输入和输出特征的平均标准差
    input_avg_std = np.mean(input_std)
    output_avg_std = np.mean(output_std)
    com_avg_std = np.mean(com_std)

    return input_avg_std, output_avg_std, com_avg_std
