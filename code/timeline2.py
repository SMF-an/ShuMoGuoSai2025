import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 数据
bomb = {
    "name": "Bomb", "release": 0.461, "delay": 0.793, "burst": 1.254, "cover_start": 1.54, "cover_end": 6.11
}


# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

y = 1  # y轴位置
    
# 投放时刻
ax.plot(bomb["release"], y, 'o', markersize=10, color='blue', label='drop time')
ax.text(bomb["release"], y + 0.1, f't_drop={bomb["release"]:.3f}s', 
            ha='center', va='bottom', fontsize=9)
    
# 起爆时刻
ax.plot(bomb["burst"], y, 's', markersize=10, color='red', label='burst time')
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
ax.set_xlim(-2, max(12, bomb["burst"]) + 2)

plt.tight_layout()
plt.savefig('./figures/bomb_timeline_2.png', dpi=300, bbox_inches='tight')
plt.show()