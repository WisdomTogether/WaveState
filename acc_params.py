import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体为 Times New Roman 并调整字体大小
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 16

# 数据
params = [21.315, 44.022, 245.76, 831.488, 227.328, 222.208]
models = ['SegMamaba', 'Segformer', 'Deeplabv3', 'UNet', 'FCN', 'PSPNet']
accuracy = [97.77, 96.23, 97.12, 97.01, 95.33, 90.62]
colors = ['pink', 'red', 'green', 'blue', 'yellow', 'purple']
bubble_sizes = [200, 400, 1000, 1500, 900, 903]

# 创建图形
plt.figure(figsize=(9, 6))

# 绘制散点图
for i in range(len(models)):
    plt.scatter(params[i], accuracy[i], s=bubble_sizes[i], c=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    
    # 显示模型名称和参数量，参数量用逗号分隔更易读
    plt.text(params[i], accuracy[i] - 0.02, f'{models[i]}\n({params[i]:,.3f} M)', 
             fontsize=12.5, ha='center', va='top', color='black')

# 添加轴标签和标题
plt.xlabel('GFlops (G)', fontweight='bold', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
# plt.title('Model Accuracy vs Parameters', fontsize=18, fontweight='bold')

# 设置y轴范围
plt.ylim(90, 98)

# 添加网格并设置透明度
plt.grid(True, alpha=0.3)

# 调整图表布局
plt.tight_layout()

# 保存为SVG格式图片
plt.savefig('miou_para_optimized.svg', bbox_inches='tight', dpi=300, format='svg')

# 显示图形
plt.show()
