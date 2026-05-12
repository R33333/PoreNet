import matplotlib.pyplot as plt

# ==========================================
# 1. 全局字体与排版设置 (学术期刊标准)
# ==========================================
# 强制使用纯白背景，避免 LaTeX 插入时出现灰底色差
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# 设置学术标准字体 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

# ==========================================
# 2. 数据定义 (严格对应论文 Table 1)
# ==========================================
# 基线 TinyBERT (Int8)
tinybert_sram = [840]
tinybert_f1 = [82.4]

# Vanilla QAT (2-bit, 4-bit)
qat_sram = [220, 420]
qat_f1 = [72.1, 75.6]

# Ours: LSMA + GCRQ (2-bit, 4-bit)
ours_sram = [220, 440]
ours_f1 = [78.5, 81.8]

# ==========================================
# 3. 绘图与样式调整
# ==========================================
# 创建画布，设置合适的长宽比
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制 Vanilla QAT 曲线 (红色虚线 + 方块)
ax.plot(qat_sram, qat_f1, marker='s', linestyle='--', color='#d62728',
        linewidth=2, markersize=8, label='Vanilla QAT Baseline')

# 绘制 Ours 曲线 (蓝色实线 + 大五角星，突出核心方法)
ax.plot(ours_sram, ours_f1, marker='*', linestyle='-', color='#1f77b4',
        linewidth=2.5, markersize=14, label='Ours (LSMA + GCRQ)')

# 绘制 TinyBERT 基准点 (绿色圆点)
ax.scatter(tinybert_sram, tinybert_f1, color='#2ca02c', marker='o',
           s=120, zorder=5, label='TinyBERT Baseline (8-bit)')

# 添加数据点文本标注 (微调了坐标位置以防遮挡曲线)
# 注：这里的偏移量 (+15, -0.6 等) 是为了视觉美观，不会改变数据点本身的真实位置
ax.text(qat_sram[0]+15, qat_f1[0]-0.6, '2-bit', color='#d62728', fontsize=11)
ax.text(qat_sram[1]+15, qat_f1[1]-0.6, '4-bit', color='#d62728', fontsize=11)
ax.text(ours_sram[0]-65, ours_f1[0]+0.3, '2-bit', color='#1f77b4', fontsize=11, fontweight='bold')
ax.text(ours_sram[1]-65, ours_f1[1]+0.3, '4-bit', color='#1f77b4', fontsize=11, fontweight='bold')

# ==========================================
# 4. 辅助线、坐标轴与图例
# ==========================================
# 添加 512KB 硬件约束线 (核心卖点)
ax.axvline(x=512, color='gray', linestyle=':', linewidth=2, label='Hardware Limit (512 KB)')
# 可以在 0 到 512KB 之间加一个非常淡的阴影区域，表示“安全部署区” (可选)
ax.axvspan(0, 512, color='#f0f0f0', alpha=0.5, zorder=0)

# 设置 X/Y 轴标签及加粗
ax.set_xlabel('Peak SRAM Occupancy (KB)', fontweight='bold')
ax.set_ylabel('F1-Score (%)', fontweight='bold')

# 限制坐标轴范围，使曲线处于视觉中心
ax.set_xlim(150, 900)
ax.set_ylim(70, 85)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=0)

# 设置图例位置在右下角，添加边框阴影
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)

# ==========================================
# 5. 输出与保存
# ==========================================
# 紧凑布局，防止标签被截断
plt.tight_layout()

# 保存为 PDF 矢量图 (分辨率 dpi=300，完美适配 LaTeX)
output_filename = 'fig3_tradeoff_curve.pdf'
plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

print(f"图表已成功生成并保存为: {output_filename}")