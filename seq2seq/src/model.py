import torch
from torch import nn
from torchinfo import summary

import config

class TranslationEncoder(nn.Module):
  """
  翻译模型编码器，基于双向 GRU。
  """

  def __init__(self, vocab_size, padding_index):
    """
    初始化编码器。

    :param vocab_size: 词表大小。
    :param padding_index: padding token 的索引。
    """
    super().__init__()
    # 嵌入层：将 token 索引映射为稠密向量
    self.embedding = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=config.EMBEDDING_DIM,
      padding_idx=padding_index
    )
    # 双向 GRU
    self.rnn = nn.GRU(
      input_size=config.EMBEDDING_DIM,
      hidden_size=config.ENCODER_HIDDEN_DIM,
      num_layers=config.ENCODER_LAYERS,
      batch_first=True,
      bidirectional=True
    )

  def forward(self, src):
    """
    前向传播。

    :param src: 输入张量，形状 (batch_size, seq_len)。
    :return: (输出张量, 最终隐藏状态)。
    """
    embedded = self.embedding(src) # (batch_size, seq_len, embedding_dim)
    output, hidden = self.rnn(embedded)
    return output, hidden

class TranslationDecoder(nn.Module):
  """
  翻译模型解码器，基于单向 GRU。
  """

  def __init__(self, vocab_size, padding_index):
    """
    初始化解码器。

    :param vocab_size: 词表大小。
    :param padding_index: padding token 的索引。
    """
    super().__init__()
    # 嵌入层
    self.embedding = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=config.EMBEDDING_DIM,
      padding_idx=padding_index
    )
    # GRU
    self.rnn = nn.GRU(
      input_size=config.EMBEDDING_DIM,
      hidden_size=config.DECODER_HIDDEN_DIM,
      batch_first=True
    )
    # 线性层：映射到词表概率分布
    self.linear = nn.Linear(
      in_features=config.DECODER_HIDDEN_DIM,
      out_features=vocab_size
    )

  def forward(self, tgt, hidden):
    """
    前向传播。

    :param tgt: 输入张量，形状 (batch_size, 1)。
    :param hidden: 隐藏状态张量，形状 (1, batch_size, hidden_dim)。
    :return: (输出张量, 新的隐藏状态)。
    """
    embedded = self.embedding(tgt) # (batch_size, 1, embedding_dim)
    output, hidden = self.rnn(embedded, hidden) # output: (batch_size, 1, hidden_dim)
    output = self.linear(output) # (batch_size, 1, vocab_size)
    return output, hidden

if __name__ == '__main__':
  encoder = TranslationEncoder(vocab_size=10000, padding_index=0)
  dummy_encoder_input = torch.randint(low=0, high=10000, size=(config.BATCH_SIZE, config.SEQ_LEN))
  summary(encoder, input_data=dummy_encoder_input)

  print('-' * 100)

  decoder = TranslationDecoder(vocab_size=10000, padding_index=0)
  dummy_decoder_input = torch.randint(low=0, high=10000, size=(config.BATCH_SIZE, 1))
  dummy_decoder_hidden = torch.randn(size=(1, config.BATCH_SIZE, config.DECODER_HIDDEN_DIM))
  summary(decoder, input_data=[dummy_decoder_input, dummy_decoder_hidden])