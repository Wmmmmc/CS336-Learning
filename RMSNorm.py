import torch
import torch.nn as nn

# --- 传统 Layer Normalization (BERT, GPT-2 等使用) ---
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 区别 1: LayerNorm 通常包含两个可学习参数：缩放 (weight) 和 偏移 (bias)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, d_model]
        
        # 区别 2: 先计算均值 (Mean)，目的是为了“去中心化”
        mean = x.mean(dim=-1, keepdim=True)
        
        # 区别 3: 计算方差 (Variance)
        # unbiased=False 对应公式中的 1/N，而不是 1/(N-1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 区别 4: 归一化公式：(x - mean) / sqrt(var + eps)
        # 核心在于它要把数据平移到 0 附近 (减去 mean)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 区别 5: 最后加上偏置 bias
        return self.weight * x_norm + self.bias


# --- RMS Normalization (Llama, PaLM 等现代大模型使用) ---
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 区别 1: RMSNorm 通常只有一个可学习参数：缩放 (weight)，没有 bias
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 区别 2: 不计算均值 (Mean)，不做“去中心化”操作
        
        # 区别 3: 计算均方根 (Root Mean Square)
        # 也就是：平方 -> 求平均 -> 开根号
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + self.eps) # 1 / sqrt(...)
        
        # 区别 4: 归一化公式：x * rsqrt
        # 核心在于它只缩放数据的幅度，不改变数据的中心位置
        x_norm = x * rsqrt
        
        # 区别 5: 最后没有加偏置 bias
        return self.weight * x_norm