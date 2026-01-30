import numpy as np


class SimpleProMP:
    def __init__(self, n_basis=20, n_dims=3):
        """
        n_basis: 基函数的数量 (控制曲线的平滑度和复杂度)
        n_dims: 数据的维度 (这里是3，对应 x, y, z)
        """
        self.n_basis = n_basis
        self.n_dims = n_dims
        self.mu_w = None  # 权重的均值
        self.cov_w = None  # 权重的协方差

    def _get_basis(self, t, duration=1.0):
        """生成高斯基函数矩阵 Phi"""
        # t 归一化到 [0, 1]
        phase = t / duration

        # 均匀分布的中心点
        centers = np.linspace(0, 1, self.n_basis)
        # 宽度 (根据中心点间距设定)
        h = 1.0 / (self.n_basis * 0.6) ** 2

        # 计算高斯基函数
        # Phi shape: (N_samples, n_basis)
        Phi = np.exp(-0.5 * (phase[:, None] - centers[None, :]) ** 2 / h)

        # 归一化基函数 (使其和为1)
        Phi = Phi / np.sum(Phi, axis=1, keepdims=True)
        return Phi

    def train(self, trajectories):
        """
        输入: trajectories list of (N, n_dims)
        即使只有一条轨迹也可以训练，虽然方差会很小。
        """
        # 1. 准备数据
        # 假设所有轨迹长度归一化，或者取重采样后的点
        # 这里简化处理：我们对每条轨迹计算权重 w

        weights = []

        for traj in trajectories:
            N = len(traj)
            t = np.linspace(0, 1, N)
            Phi = self._get_basis(t)  # (N, n_basis)

            # 使用岭回归 (Ridge Regression) 计算该轨迹对应的权重 w
            # Y = Phi * w => w = (Phi.T * Phi + lam * I)^-1 * Phi.T * Y
            # 我们需要对每个维度(x,y,z)分别算，或者堆叠算
            # w shape should be (n_dims * n_basis)

            # 这里构建块对角矩阵会比较大，我们简化为逐维度计算然后拼接
            w_sample = []
            lam = 1e-12

            for d in range(self.n_dims):
                y_d = traj[:, d]  # (N,)
                # w_d = inv(Phi.T @ Phi + lam*I) @ Phi.T @ y_d
                matrix = Phi.T @ Phi + np.eye(self.n_basis) * lam
                w_d = np.linalg.solve(matrix, Phi.T @ y_d)
                w_sample.append(w_d)

            # 展平: [w_x1...w_xn, w_y1...w_yn, ...]
            weights.append(np.concatenate(w_sample))

        weights = np.array(weights)  # (n_demos, n_basis * n_dims)

        # 2. 估计参数分布 (Mean and Covariance)
        self.mu_w = np.mean(weights, axis=0)
        # 加上一点正则化防止奇异矩阵
        if len(weights) == 1:
            # 如果只有一条轨迹，无法计算统计协方差。
            # 我们手动构建一个对角矩阵，表示"默认的不确定性"。
            # 这里的 1e-6 决定了模型对初始轨迹的"信任程度"：
            # 值越小，生成的轨迹越死板地贴合原轨迹；值越大，越容易被新起点拉弯。
            self.cov_w = np.eye(len(self.mu_w)) * 1e-6
        else:
            # 如果有多条轨迹，正常计算
            self.cov_w = np.cov(weights, rowvar=False) + np.eye(len(self.mu_w)) * 1e-6

    def generate_trajectory(self, duration=1.0, num_steps=100,
                            start_pos=None, end_pos=None):
        """
        条件推理 (Conditioning)：给定起点和终点，推导最可能的轨迹
        """
        t = np.linspace(0, duration, num_steps)
        Phi = self._get_basis(t, duration)  # (N, n_basis)

        # 构建大 Phi 矩阵 (N*dims, n_basis*dims)
        # 这是一个块对角矩阵
        Phi_block = np.zeros((num_steps * self.n_dims, self.n_basis * self.n_dims))
        for d in range(self.n_dims):
            row_start = d * num_steps
            col_start = d * self.n_basis
            Phi_block[row_start: row_start + num_steps, col_start: col_start + self.n_basis] = Phi

        # --- Conditioning (核心) ---
        new_mu_w = self.mu_w.copy()
        new_cov_w = self.cov_w.copy()

        observations = []  # list of (matrix_idx, value)

        # 1. 构造观测矩阵 H 和观测值 y
        # 我们要约束 t=0 时位置为 start_pos，t=end 时位置为 end_pos

        H_list = []
        y_list = []

        # 基函数在 t=0 和 t=T 的值
        phi_0 = self._get_basis(np.array([0.0]), duration)[0]  # (n_basis,)
        phi_T = self._get_basis(np.array([duration]), duration)[0]

        if start_pos is not None:
            for d in range(self.n_dims):
                # 构造一行：对应维度 d 的基函数系数
                h_row = np.zeros(self.n_basis * self.n_dims)
                h_row[d * self.n_basis: (d + 1) * self.n_basis] = phi_0
                H_list.append(h_row)
                y_list.append(start_pos[d])

        if end_pos is not None:
            for d in range(self.n_dims):
                h_row = np.zeros(self.n_basis * self.n_dims)
                h_row[d * self.n_basis: (d + 1) * self.n_basis] = phi_T
                H_list.append(h_row)
                y_list.append(end_pos[d])

        if len(H_list) > 0:
            H = np.array(H_list)  # (num_obs, n_params)
            y_obs = np.array(y_list)

            # 观测噪声 (极小，表示强制通过)
            Sigma_y = np.eye(len(y_obs)) * 1e-6

            # 卡尔曼更新公式 / 贝叶斯推理
            # K = Cov * H.T * inv(H * Cov * H.T + Sigma_y)
            temp = H @ new_cov_w @ H.T + Sigma_y
            K = new_cov_w @ H.T @ np.linalg.inv(temp)

            # mu_new = mu + K * (y - H * mu)
            new_mu_w = new_mu_w + K @ (y_obs - H @ new_mu_w)

            # Cov_new = Cov - K * H * Cov
            new_cov_w = new_cov_w - K @ H @ new_cov_w

        # --- 生成最终轨迹 ---
        # Traj = Phi_block * w_mean
        # 我们为了方便，还是拆开维度算
        generated_pos = np.zeros((num_steps, self.n_dims))

        for d in range(self.n_dims):
            w_d = new_mu_w[d * self.n_basis: (d + 1) * self.n_basis]
            generated_pos[:, d] = Phi @ w_d

        return generated_pos