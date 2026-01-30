import matplotlib.pyplot as plt
import numpy as np


def visualize_trajectory(traj_data, sample_step=None, arrow_length=0.05):
    """
    可视化 6D 轨迹。

    Args:
        traj_data: 包含 {'pos', 'rot'} 的字典
        sample_step: 每隔多少帧画一个坐标轴 (None则自动计算，约画20个)
        arrow_length: 坐标轴箭头的长度 (米)
    """
    if traj_data is None:
        print("没有轨迹数据可供可视化。")
        return

    positions = traj_data['pos']
    rotations = traj_data['rot']
    num_points = len(positions)

    # 创建 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 画质心轨迹线 (黑色虚线)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            label='CoM Path', color='k', linestyle='--', linewidth=1, alpha=0.5)

    t = np.linspace(0, 1, num_points)
    # 计算公式：Start + t * (End - Start)
    backbone_points = positions[0] + np.outer(t, (positions[-1] - positions[0]))

    ax.plot(backbone_points[:, 0], backbone_points[:, 1], backbone_points[:, 2],
            label='Linear Backbone', color='blue', linestyle=':', linewidth=2, alpha=0.8)

    # 1. 画质心轨迹线 (黑色虚线) - 这里的 positions 是实际带偏差的路径


    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
        label='Actual Path (with Deviation)', color='k', linestyle='--', linewidth=1, alpha=0.5)

    # 2. 标记起点 (绿色圆点) 和 终点 (红色星号)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='g', s=50, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='r', marker='*', s=100, label='End')

    # 3. 抽样绘制坐标轴 (显示姿态)
    # 如果没指定步长，就自动计算，保证大概画 15-20 个坐标系，避免太密
    if sample_step is None:
        sample_step = max(1, num_points // 20)

    for i in range(0, num_points, sample_step):
        pos = positions[i]
        rot = rotations[i]

        # 旋转矩阵的列向量就是局部坐标轴在世界坐标系下的方向
        # rot[:, 0] -> 局部 X 轴
        # rot[:, 1] -> 局部 Y 轴
        # rot[:, 2] -> 局部 Z 轴

        # 画 X 轴 (红色)
        ax.quiver(pos[0], pos[1], pos[2],
                  rot[0, 0], rot[1, 0], rot[2, 0],
                  length=arrow_length, color='r', linewidth=1)

        # 画 Y 轴 (绿色)
        ax.quiver(pos[0], pos[1], pos[2],
                  rot[0, 1], rot[1, 1], rot[2, 1],
                  length=arrow_length, color='g', linewidth=1)

        # 画 Z 轴 (蓝色)
        ax.quiver(pos[0], pos[1], pos[2],
                  rot[0, 2], rot[1, 2], rot[2, 2],
                  length=arrow_length, color='b', linewidth=1)

    # 设置轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Object Trajectory ({num_points} frames)')
    ax.legend()

    # 强制坐标轴比例一致 (Matplotlib 3D 默认比例通常是歪的，这个函数很重要)
    set_axes_equal(ax)

    plt.show()


def set_axes_equal(ax):
    """
    辅助函数：强制设置 3D 图形的 X, Y, Z 轴比例一致。
    这样轨迹才不会变形。
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # 找出最大的范围，作为所有轴的范围
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    small_data = np.load("../data/small_box_trajectory_data.npz")
    big_data = np.load("../data/big_box_trajectory_data.npz")
    # small_trajectory = {'pos': small_data['pos'], 'rot': small_data['rot']}
    big_traje = {'pos': big_data['pos'], 'rot': big_data['rot']}

    # 可视化 (这时候用普通 python 运行，plt.show() 是完全正常的)
    # visualize_trajectory(small_trajectory, arrow_length=0.05)
    visualize_trajectory(big_traje, arrow_length=0.05)


if __name__ == '__main__':
    main()