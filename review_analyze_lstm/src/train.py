# train.py

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import JiebaTokenizer
import config
from model import ReviewAnalyzeModel

def train_one_epoch(model, dataloader, loss_function, optimizer, device):
  """
  训练一个 epoch。

  :param model: 模型。
  :param dataloader: 数据加载器。
  :param loss_function: 损失函数。
  :param optimizer: 优化器。
  :param device: 设备。
  :return: 平均损失。
  """
  total_loss = 0
  model.train()

  for inputs, targets in tqdm(dataloader, desc='训练'):
    # 移动数据到设备
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = loss_function(outputs, targets)

    # 反向传播
    loss.backward()

    # 参数更新
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(dataloader)

def train():
  """
  模型训练主函数。
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataloader = get_dataloader()

  tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

  model = ReviewAnalyzeModel(
    vocab_size=tokenizer.vocab_size,
    padding_idx=tokenizer.pad_token_index
  ).to(device)

  loss_function = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

  writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

  best_loss = float('inf')

  for epoch in range(1, config.EPOCHS + 1):
    print(f'========== Epoch: {epoch} ==========')

    avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer, device)

    print(f'Loss: {avg_loss:.4f}')

    writer.add_scalar('Loss/Train', avg_loss, epoch)

    if avg_loss < best_loss:
      best_loss = avg_loss
      torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
      print('模型保存成功')

if __name__ == '__main__':
  train()