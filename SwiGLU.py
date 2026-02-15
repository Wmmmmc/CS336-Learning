import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 第一部分：经典基础类 (教科书常客)
# ==========================================

class ClassicActivations(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        print(f"输入数据范围: {x.min():.2f} ~ {x.max():.2f}")
        
        # 1. Sigmoid
        # 公式: 1 / (1 + exp(-x))
        # 优点: 将输出压缩到 (0, 1)，适合二分类概率输出。
        # 缺点: 容易出现“梯度消失” (Gradient Vanishing)，输出不是 0 均值。
        sigmoid_out = torch.sigmoid(x)
        
        # 2. Tanh (双曲正切)
        # 公式: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # 优点: 输出范围 (-1, 1)，0 均值，比 Sigmoid 收敛快。
        # 缺点: 依然存在梯度消失问题。
        tanh_out = torch.tanh(x)
        
        # 3. ReLU (Rectified Linear Unit) - 深度学习的里程碑
        # 公式: max(0, x)
        # 优点: 计算极快，解决了正区间的梯度消失问题，稀疏性好。
        # 缺点: Dead ReLU 问题 (负数部分梯度为 0，神经元可能“死掉”不再更新)。
        relu_out = F.relu(x)
        
        # 4. LeakyReLU
        # 公式: max(0.01x, x)
        # 优点: 给负数部分一个很小的斜率，解决 Dead ReLU 问题。
        leaky_out = F.leaky_relu(x, negative_slope=0.01)
        
        return {"Sigmoid": sigmoid_out, "ReLU": relu_out}

# ==========================================
# 第二部分：现代主流类 (Transformer/BERT/CNN 常用)
# ==========================================

class ModernActivations(nn.Module):
    def forward(self, x):
        # 5. GELU (Gaussian Error Linear Unit) - BERT/GPT-2 标配
        # 原理: 在 ReLU 基础上增加了平滑性和概率思想。
        # 优点: 在负数区域更平滑，大模型表现通常优于 ReLU。
        gelu_out = F.gelu(x)
        
        # 6. SiLU (Sigmoid Linear Unit) 也叫 Swish
        # 公式: x * sigmoid(x)
        # 优点: 自门控 (Self-gated)，平滑非单调，YOLOv5 和 LLaMA 都在用。
        silu_out = F.silu(x)
        
        return {"GELU": gelu_out, "SiLU": silu_out}

# ==========================================
# 第三部分：大模型门控机制 (LLaMA/PaLM 核心)
# ==========================================

# 7. GLU (Gated Linear Unit) - 门控线性单元
# 原理: 输入被切成两半，一半做值，一半做门(0~1)，相乘控制信息流。
# 公式: GLU(a, b) = a * sigmoid(b)
class GLU(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # GLU 需要把维度翻倍再切分，或者直接输入就是双倍维度
        self.linear = nn.Linear(in_features, in_features * 2)

    def forward(self, x):
        # x shape: [batch, in_features]
        out = self.linear(x)
        # 将输出切分为两份: a (内容) 和 b (门)
        a, b = out.chunk(2, dim=-1) 
        # 门控机制：内容 * 门的开启程度(Sigmoid)
        return a * torch.sigmoid(b)

# 8. SwiGLU (Swish-Gated Linear Unit) - 目前 LLM 的最强王者
# 原理: 把 GLU 里的 Sigmoid 换成 Swish (SiLU)。
# 公式: SwiGLU(x) = (xW + b) * SiLU(xV + c)
# 优点: LLaMA、PaLM、Baichuan 都在用，相比 ReLU 性能有显著提升。
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # SwiGLU 通常涉及三个矩阵：
        # 1. 门投影 (Gate proj)
        # 2. 值投影 (Value proj)
        # 3. 输出投影 (Output proj) - 这里只展示核心激活部分
        self.w_gate = nn.Linear(dim, dim)
        self.w_val  = nn.Linear(dim, dim)

    def forward(self, x):
        # 1. 计算门控分支 (包含 SiLU 激活)
        gate = F.silu(self.w_gate(x))