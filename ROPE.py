import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        max_seq_len: 最大序列长度
        """
        super().__init__()
        self.d_k = d_k

        # 1. 计算频率频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2], 对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers) # 形状: (d_k/2,)

        # 2. 创建位置序列 [0, 1, ..., max_seq_len - 1]
        t = torch.arange(max_seq_len, device=device).float() # 形状: (max_seq_len,)

        # 3. 计算所有位置的所有角度 (外积)
        # freqs_matrix 形状: (max_seq_len, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 提取 cos/sin (..., Seq, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时，才需要手动插入 Head 维。
        # 对于 test_rope 这种 3D x vs 2D cos 的情况，PyTorch 会自动左侧补 1，无需操作。
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分并旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        # 旋转公式:
        # x_new = x * cos - y * sin
        # y_new = x * sin + y * cos
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output