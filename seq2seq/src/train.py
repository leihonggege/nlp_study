import time
from itertools import chain

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationEncoder, TranslationDecoder

def train_one_epoch(dataloader, encoder, decoder, loss_function, optimizer, device):
  """
  训练一个 epoch。

  :param dataloader: 数据加载器。
  :param encoder: 编码器。
  :param decoder: 解码器。
  :param loss_function: 损失函数。
  :param optimizer: 优化器。
  :param device: 设备。
  :return: 平均损失。
  """
  encoder.train()
  decoder.train()
  total_loss = 0

  for src, tgt in tqdm(dataloader, desc='训练'):
    src = src.to(device)
    tgt = tgt.to(device)

    optimizer.zero_grad()

    # 编码器处理
    _, encoder_hidden = encoder(src)

    # 拼接前向后向隐藏状态
    forward_hidden = encoder_hidden[-2]
    backward_hidden = encoder_hidden[-1]
    context_vector = torch.cat([forward_hidden, backward_hidden], dim=1)

    # 初始化解码器输入和隐藏状态
    decoder_input = tgt[:, 0:1]
    decoder_hidden = context_vector.unsqueeze(0)

    decoder_outputs = []
    for step in range(1, config.SEQ_LEN):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
      decoder_outputs.append(decoder_output)
      decoder_input = tgt[:, step:step + 1]

    decoder_outputs = torch.cat(decoder_outputs, dim=1)
    decoder_targets = tgt[:, 1:]

    loss = loss_function(
      decoder_outputs.reshape(-1, decoder_outputs.shape[-1]),
      decoder_targets.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(dataloader)

def train():
  """
  模型训练主函数。
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataloader = get_dataloader()

  zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
  en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

  encoder = TranslationEncoder(
    vocab_size=zh_tokenizer.vocab_size,
    padding_index=zh_tokenizer.pad_token_index
  ).to(device)
  decoder = TranslationDecoder(
    vocab_size=en_tokenizer.vocab_size,
    padding_index=en_tokenizer.pad_token_index
  ).to(device)

  loss_function = CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
  optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=config.LEARNING_RATE)

  writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

  best_loss = float('inf')

  for epoch in range(1, config.EPOCHS + 1):
    print(f'========== Epoch {epoch} ==========')

    avg_loss = train_one_epoch(dataloader, encoder, decoder, loss_function, optimizer, device)

    print(f'平均损失: {avg_loss:.4f}')
    writer.add_scalar('Loss', avg_loss, epoch)

    if avg_loss < best_loss:
      best_loss = avg_loss
      torch.save(encoder.state_dict(), config.MODELS_DIR / 'encoder.pt')
      torch.save(decoder.state_dict(), config.MODELS_DIR / 'decoder.pt')
      print('已保存模型')

if __name__ == '__main__':
  train()