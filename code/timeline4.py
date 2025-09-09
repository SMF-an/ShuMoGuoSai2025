import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 数据
uav_data = [
    {"name": "FY 1", "release": 0.461, "delay": 0.793, "burst": 1.254, "cover_start": 1.540, "cover_end": 6.110},
    {"name": "FY 2", "release": 11.067, "delay": 2.980, "burst": 14.047, "cover_start": 18.000, "cover_end": 21.860},
    {"name": "FY 3", "release": 23.833, "delay": 4.268, "burst": 28.101, "cover_start": 28.520, "cover_end": 31.470}
]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 设置y轴位置
y_positions = [3, 2, 1]  # 从顶部开始

# 绘制每个无人机的时间线
for i, uav in enumerate(uav_data):
    y = y_positions[i]
    
    # 投放时刻
    ax.plot(uav["release"], y, 'o', markersize=10, color='blue', label='drop time' if i == 0 else "")
    ax.text(uav["release"], y + 0.1, f't_drop={uav["release"]:.3f}s', 
            ha='center', va='bottom', fontsize=9)
    
    # 起爆时刻
    ax.plot(uav["burst"], y, 's', markersize=10, color='red', label='burst time' if i == 0 else "")
    ax.text(uav["burst"], y - 0.1, f't_burst={uav["burst"]:.3f}s', 
            ha='center', va='bottom', fontsize=9)
    
    # 投放与起爆之间的连线
    ax.plot([uav["release"], uav["burst"]], [y, y], 'k--', linewidth=1.5)
    
    # 遮蔽区间（如果有）
    if uav["cover_start"] is not None and uav["cover_end"] is not None:
        width = uav["cover_end"] - uav["cover_start"]
        rect = patches.Rectangle((uav["cover_start"], y-0.2), width, 0.4, 
                                linewidth=1, edgecolor='green', facecolor='green', alpha=0.3)
        ax.add_patch(rect)
        ax.text(uav["cover_start"] + width/2, y-0.25, 
                f'cover interval: [{uav["cover_start"]:.2f}, {uav["cover_end"]:.2f}]s', 
                ha='center', va='top', fontsize=9, color='green')
    
    # 弹名称
    ax.text(-1.5, y, uav["name"], ha='right', va='center', fontsize=12, fontweight='bold')

# 设置图表属性
ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.set_title('uav Release and Burst Timeline')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right')

# 设置x轴范围
ax.set_xlim(-2, max(12, max(uav["burst"] for uav in uav_data) + 2))

plt.tight_layout()
plt.savefig('./figures/uav_timeline.png', dpi=300, bbox_inches='tight')
plt.show()