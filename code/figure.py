import matplotlib.pyplot as plt
import params as p
from funcs import unit


def plot_init_pos():
    """
    绘制导弹和无人机的初始位置和飞行方向
    """
    plt.figure(1, figsize=(12, 8))

    # 绘制真假目标位置
    plt.plot(p.real[0], p.real[1], 'g*', markersize=6)
    plt.plot(p.fake[0], p.fake[1], 'm*', markersize=6)
    plt.text(p.real[0]+50, p.real[1]+50, 'Real Target', fontsize=6)
    plt.text(p.fake[0]+50, p.fake[1]+50, 'Fake Target', fontsize=6)

    # 绘制导弹位置
    plt.plot(p.M1_0[0], p.M1_0[1], 'ro', markersize=4)
    plt.plot(p.M2_0[0], p.M2_0[1], 'ro', markersize=4)
    plt.plot(p.M3_0[0], p.M3_0[1], 'ro', markersize=4)
    plt.text(p.M1_0[0]+50, p.M1_0[1]+50, 'M1', fontsize=6)
    plt.text(p.M2_0[0]+50, p.M2_0[1]+50, 'M2', fontsize=6)
    plt.text(p.M3_0[0]+50, p.M3_0[1]+50, 'M3', fontsize=6)

    # 绘制无人机位置
    plt.plot(p.FY1_0[0], p.FY1_0[1], 'bo', markersize=4)
    plt.plot(p.FY2_0[0], p.FY2_0[1], 'bo', markersize=4)
    plt.plot(p.FY3_0[0], p.FY3_0[1], 'bo', markersize=4)
    plt.plot(p.FY4_0[0], p.FY4_0[1], 'bo', markersize=4)
    plt.plot(p.FY5_0[0], p.FY5_0[1], 'bo', markersize=4)
    plt.text(p.FY1_0[0]+50, p.FY1_0[1]+50, 'FY1', fontsize=6)
    plt.text(p.FY2_0[0]+50, p.FY2_0[1]+50, 'FY2', fontsize=6)
    plt.text(p.FY3_0[0]+50, p.FY3_0[1]+50, 'FY3', fontsize=6)
    plt.text(p.FY4_0[0]+50, p.FY4_0[1]+50, 'FY4', fontsize=6)
    plt.text(p.FY5_0[0]+50, p.FY5_0[1]+50, 'FY5', fontsize=6)

    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制飞行方向箭头
    arrow_length = 250  # 箭头长度
    for i, m in enumerate([p.M1_0, p.M2_0, p.M3_0]):
        direction = unit(p.fake - m)  # 飞行方向单位向量
        dx, dy = direction[0] * arrow_length, direction[1] * arrow_length
        plt.arrow(m[0], m[1], dx, dy, head_width=6, head_length=10, fc='red', ec='red', linewidth=3)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title('Initial Positions of Missiles and Drones', fontsize=14)
    plt.savefig('./figures/init_pos.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_init_pos()