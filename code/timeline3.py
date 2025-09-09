import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 数据
bomb_data = [
    {"name": "Bomb 1", "release": 0, "delay": 0, "burst": 0., "cover_start": 4.85, "cover_end": 7.4},
    {"name": "Bomb 2", "release": 1, "delay": 0, "burst": 1, "cover_start": 1.0, "cover_end": 5.37},
    {"name": "Bomb 3", "release": 10.1, "delay": 1.21, "burst": 11.31, "cover_start": None, "cover_end": None}
]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 设置y轴位置
y_positions = [3, 2, 1]  # 从顶部开始

# 绘制每个弹的时间线
for i, bomb in enumerate(bomb_data):
    y = y_positions[i]
    
    # 投放时刻
    ax.plot(bomb["release"], y, 'o', markersize=10, color='blue', label='drop time' if i == 0 else "")
    ax.text(bomb["release"], y + 0.1, f't_drop={bomb["release"]:.3f}s', 
            ha='center', va='bottom', fontsize=9)
    
    # 起爆时刻
    ax.plot(bomb["burst"], y, 's', markersize=10, color='red', label='burst time' if i == 0 else "")
    ax.text(bomb["burst"], y - 0.1, f't_burst={bomb["burst"]:.3f}s', 
            ha='center', va='bottom', fontsize=9)
    
    # 投放与起爆之间的连线
    ax.plot([bomb["release"], bomb["burst"]], [y, y], 'k--', linewidth=1.5)
    
    # 遮蔽区间（如果有）
    if bomb["cover_start"] is not None and bomb["cover_end"] is not None:
        width = bomb["cover_end"] - bomb["cover_start"]
        rect = patches.Rectangle((bomb["cover_start"], y-0.2), width, 0.4, 
                                linewidth=1, edgecolor='green', facecolor='green', alpha=0.3)
        ax.add_patch(rect)
        ax.text(bomb["cover_start"] + width/2, y-0.25, 
                f'cover interval: [{bomb["cover_start"]:.2f}, {bomb["cover_end"]:.2f}]s', 
                ha='center', va='top', fontsize=9, color='green')
    
    # 弹名称
    ax.text(-1.5, y, bomb["name"], ha='right', va='center', fontsize=12, fontweight='bold')

# 设置图表属性
ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.set_title('Bomb Release and Burst Timeline')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right')

# 设置x轴范围
ax.set_xlim(-2, max(12, max(bomb["burst"] for bomb in bomb_data) + 2))

plt.tight_layout()
plt.savefig('./figures/bomb_timeline_3.png', dpi=300, bbox_inches='tight')
plt.show()