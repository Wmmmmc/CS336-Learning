import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MyLinear(nn.Module):
    """
    手写 Linear 层 (全连接层)
    数学公式: y = x @ W.T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. 定义权重 (Weight)
        # 形状是 [out_features, in_features]，这是 PyTorch 的惯例 (为了方便计算 x @ W.T)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # 2. 定义偏置 (Bias)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # 3. 初始化参数 (这一步非常关键！)
        self.reset_parameters()

    def reset_parameters(self):
        """
        使用 Kaiming Uniform (He 初始化) 初始化权重，这对 ReLU/GELU 网络收敛很重要。
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            # 偏置通常初始化为一个很小的均匀分布
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, ..., in_features]
        # output: [Batch, ..., out_features]
        
        # 这里的 F.linear 等价于: x @ self.weight.T + self.bias
        return F.linear(x, self.weight, self.bias)

# ==========================================
# 进阶：如何与激活函数组合 (Transformer 的 FFN)
# ==========================================

class FeedForward(nn.Module):
    """
    标准的 MLP 块：Linear -> Activation -> Linear
    """
    def __init__(self, dim: int, hidden_dim: int, act_layer: nn.Module = nn.ReLU()):
        super().__init__()
        # 第一层：放大维度 (例如 4倍)
        self.w1 = MyLinear(dim, hidden_dim)
        # 激活函数 (你可以传入 ReLU, GELU, SiLU 等)
        self.act = act_layer
        # 第二层：还原维度
        self.w2 = MyLinear(hidden_dim, dim)
        # Dropout (防止过拟合，可选)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x -> [Linear] -> [Act] -> [Linear] -> [Dropout] -> out
        return self.dropout(self.w2(self.act(self.w1(x))))

class LlamaMLP(nn.Module):
    """
    Llama 风格的 MLP (使用 SwiGLU)
    公式: down( silu(gate(x)) * up(x) )
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # SwiGLU 需要三个 Linear 层
        self.gate_proj = MyLinear(dim, hidden_dim, bias=False) # 门
        self.up_proj   = MyLinear(dim, hidden_dim, bias=False) # 值
        self.down_proj = MyLinear(hidden_dim, dim, bias=False) # 输出

    def forward(self, x):
        # 1. 门控分支: Linear -> SiLU
        gate = F.silu(self.gate_proj(x))
        # 2. 数值分支: Linear
        up = self.up_proj(x)
        # 3. 相乘 (SwiGLU 核心)
        fused = gate * up
        # 4. 输出投影
        return self.down_proj(fused)

# ==========================================
# 运行测试
# ==========================================
if __name__ == "__main__":
    # 1. 基础 Linear 测试
    print("--- 1. 测试 MyLinear ---")
    linear = MyLinear(in_features=4, out_features=2)
    x = torch.randn(3, 4) # Batch=3, Dim=4
    out = linear(x)
    print(f"输入: {x.shape} -> Linear -> 输出: {out.shape}")
    print(f"权重形状: {linear.weight.shape}") # 应该是 [2, 4]

    # 2. 传统 MLP 测试 (ReLU)
    print("\n--- 2. 测试 FeedForward (with ReLU) ---")
    ffn = FeedForward(dim=64, hidden_dim=256, act_layer=nn.ReLU())
    x_ffn = torch.randn(2, 10, 64) # Seq_len=10
    out_ffn = ffn(x_ffn)
    print(f"Standard FFN 输出: {out_ffn.shape}")

    # 3. Llama MLP 测试 (with SwiGLU)
    print("\n--- 3. 测试 LlamaMLP (SwiGLU 结构) ---")
    # 注意：Llama 中 hidden_dim 通常是 4d 的 2/3 左右，且是 256 的倍数
    llama_mlp = LlamaMLP(dim=64, hidden_dim=172) 
    out_llama = llama_mlp(x_ffn)
    print(f"Llama MLP 输出: {out_llama.shape}")
