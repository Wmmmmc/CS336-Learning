import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GeneralAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, context_dim: int = None):
        """
        通用的注意力机制模块，支持 MHA, GQA, MQA 和 Cross-Attention。
        
        Args:
            dim: 输入向量的维度 (d_model)
            n_heads: Query (Q) 的头数
            n_kv_heads: Key (K) 和 Value (V) 的头数。
                        - 如果 = n_heads: 就是标准的 MHA (多头注意力)
                        - 如果 = 1: 就是 MQA (多查询注意力)
                        - 如果 < n_heads 且 > 1: 就是 GQA (分组查询注意力)
            context_dim: 上下文维度。如果不为 None，则开启 Cross-Attention 模式。
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.dim = dim
        self.head_dim = dim // n_heads # 每个头的维度
        
        # GQA/MQA 的核心：计算每个 KV head 需要重复多少次才能匹配 Q head
        self.n_rep = self.n_heads // self.n_kv_heads 

        # --- 定义投影层 ---
        # Query 永远来自输入 x
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        
        # Key 和 Value 的来源取决于是否是 Cross Attention
        # 如果是 Self-Attention, K/V 输入维度是 dim
        # 如果是 Cross-Attention, K/V 输入维度是 context_dim
        kv_input_dim = context_dim if context_dim is not None else dim
        self.wk = nn.Linear(kv_input_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(kv_input_dim, self.n_kv_heads * self.head_dim, bias=False)
        
        # 输出投影层
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        GQA/MQA 的核心操作：将 K 和 V 的头进行复制扩展，以便与 Q 的头对齐。
        x shape: [Batch, n_kv_heads, Seq_Len, Head_Dim]
        output:  [Batch, n_heads,    Seq_Len, Head_Dim]
        """
        if n_rep == 1:
            return x
        
        batch_size, n_kv_heads, seq_len, head_dim = x.shape
        # 在第 2 个维度 (n_kv_heads) 后增加一个维度用于复制
        x = x[:, :, None, :, :].expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)
        # 展平，使得 n_kv_heads * n_rep = n_heads
        return x.reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None):
        """
        Args:
            x: 输入 Query 源 [Batch, Seq_Len_Q, Dim]
            context: 上下文 (Key/Value 源)。
                     - 如果为 None，则是 Self-Attention (自己查自己)
                     - 如果不为 None，则是 Cross-Attention (x 查 context)
            mask: 掩码 (用于通过 causal masking 屏蔽未来信息)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算 Query, Key, Value
        # Q 始终来自 x
        xq = self.wq(x) # [Batch, Seq_Len, n_heads * head_dim]
        
        # K, V 取决于是自注意力还是交叉注意力
        if context is None:
            # Self-Attention: K, V 也来自 x
            xk = self.wk(x)
            xv = self.wv(x)
        else:
            # Cross-Attention: K, V 来自 context
            xk = self.wk(context)
            xv = self.wv(context)

        # 2. 拆分 Heads (Reshape + Transpose)
        # 变换为: [Batch, n_heads, Seq_Len, Head_Dim]
        xq = xq.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # K, V 的序列长度可能与 Q 不同 (在 Cross Attention 中)
        src_len = xk.shape[1] 
        xk = xk.view(batch_size, src_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, src_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. GQA/MQA 处理: 如果 KV 头数少于 Q 头数，需要复制扩展
        if self.n_kv_heads != self.n_heads:
            xk = self.repeat_kv(xk, self.n_rep)
            xv = self.repeat_kv(xv, self.n_rep)

        # 此刻，Q, K, V 的维度都是: [Batch, n_heads, Seq_Len, Head_Dim]

        # 4. Scaled Dot-Product Attention (核心公式)
        # scores = Q @ K.T / sqrt(d)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores shape: [Batch, n_heads, Seq_Len_Q, Seq_Len_K]

        # 5. 应用 Mask (如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 6. Softmax 归一化
        probs = F.softmax(scores, dim=-1)

        # 7. 加权求和: probs @ V
        output = torch.matmul(probs, xv) # [Batch, n_heads, Seq_Len_Q, Head_Dim]

        # 8. 拼接 Heads 并通过输出层
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

# ==========================================
# 运行测试：包含 MHA, GQA, Cross-Attention
# ==========================================
if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模拟参数
    BATCH = 2
    SEQ_LEN = 10
    DIM = 64     # 模型维度
    N_HEADS = 8  # Q 的头数

    print("--- 1. 测试标准 Multi-Head Attention (MHA) ---")
    # MHA 特点: KV 头数 = Q 头数 (都是 8)
    mha = GeneralAttention(dim=DIM, n_heads=N_HEADS, n_kv_heads=N_HEADS)
    x = torch.randn(BATCH, SEQ_LEN, DIM)
    out = mha(x)
    print(f"MHA 输入: {x.shape} -> 输出: {out.shape}")

    print("\n--- 2. 测试 Grouped-Query Attention (GQA) ---")
    # GQA 特点: KV 头数 < Q 头数 (这里 KV=2, Q=8, 每个 KV 组管 4 个 Q)
    # 这种机制在 Llama 2 (70B) 和 Llama 3 中被广泛使用，能极大减少显存占用
    gqa = GeneralAttention(dim=DIM, n_heads=N_HEADS, n_kv_heads=2)
    out = gqa(x)
    print(f"GQA 输入: {x.shape} -> 输出: {out.shape}")
    print(f"KV 重复倍数: {gqa.n_rep} (这意味着每个 KV 头复制了 4 次)")

    print("\n--- 3. 测试 Cross-Attention (交叉注意力) ---")
    # Cross-Attention 特点: Q 来自文本/解码器，K/V 来自图像/编码器
    # 假设 context (比如图像特征) 维度是 128
    CONTEXT_DIM = 128
    cross_attn = GeneralAttention(dim=DIM, n_heads=N_HEADS, context_dim=CONTEXT_DIM)
    
    query_input = torch.randn(BATCH, SEQ_LEN, DIM)    # 比如文本 Prompt
    context_input = torch.randn(BATCH, 20, CONTEXT_DIM) # 比如 CLIP 提取的图像特征 (序列长20)
    
    out = cross_attn(query_input, context=context_input)
    print(f"Query 输入: {query_input.shape}")
    print(f"Context 输入: {context_input.shape}")
    print(f"Cross 输出: {out.shape} (保持 Query 的序列长度和维度)")
