# model.py

import torch
from torch import nn
import config
from torchinfo import summary

class ReviewAnalyzeModel(nn.Module):
  """
  评论情感分析模型，基于 LSTM。
  """

  def __init__(self, vocab_size, padding_idx):
    """
    初始化模型。

    :param vocab_size: 词表大小。
    :param padding_idx: padding token 的索引。
    """
    super().__init__()
    # 嵌入层：将索引映射为词向量
    self.embedding = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=config.EMBEDDING_DIM,
      padding_idx=padding_idx
    )
    # LSTM 层：提取序列特征
    self.lstm = nn.LSTM(
      input_size=config.EMBEDDING_DIM,
      hidden_size=config.HIDDEN_DIM,
      batch_first=True
    )
    # 线性层：映射到单输出，用于二分类
    self.linear = nn.Linear(in_features=config.HIDDEN_DIM, out_features=1)

  def forward(self, x):
    """
    前向传播。

    :param x: 输入张量，形状 (batch_size, seq_len)。
    :return: 模型输出张量，形状 (batch_size,)。
    """
    # 嵌入层处理
    embed = self.embedding(x) # (batch_size, seq_len, embedding_dim)

    # LSTM 处理序列
    output, _ = self.lstm(embed) # (batch_size, seq_len, hidden_dim)

    # 取最后时间步隐藏状态用于分类
    result = self.linear(4[:, -1, :]).squeeze(dim=1) # (batch_size,)

    return result

if __name__ == '__main__':
  model = ReviewAnalyzeModel(vocab_size=1000, padding_idx=0)

  # 创建 dummy 输入张量用于结构展示
  dummy_input = torch.randint(
    low=0,
    high=1000,
    size=(config.BATCH_SIZE, config.SEQ_LEN),
    dtype=torch.long
  )

  # 打印模型结构信息
  summary(model, input_data=dummy_input)