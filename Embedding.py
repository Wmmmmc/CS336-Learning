import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        
        # 1. 分配内存并包装为参数 (W 维度: vocab_size x d_model)
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        # 2. 按照作业要求执行初始化
        # mean=0, std=1.0, 截断在 [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids 形状: [B, S]
        # 直接通过索引从矩阵中“捞出”对应的向量
        return self.weight[token_ids] # 返回形状: [B, S, D]
    