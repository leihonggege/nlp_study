# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config

class ReviewAnalyzeDataset(Dataset):
  """
  评论情感分析数据集。
  """

  def __init__(self, file_path):
    """
    初始化数据集。

    :param file_path: 数据文件路径（JSONL 格式）。
    """
    # 加载 JSONL 数据到内存
    self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

  def __len__(self):
    """
    获取数据集样本数。

    :return: 样本数量。
    """
    return len(self.data)

  def __getitem__(self, index):
    """
    获取指定索引的样本。

    :param index: 样本索引。
    :return: (input_tensor, target_tensor)
    """
    # 构建输入和目标张量
    input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
    target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float)

    return input_tensor, target_tensor

def get_dataloader(train=True):
  """
  创建数据加载器。

  :param train: 是否加载训练集（True）或测试集（False）。
  :return: DataLoader 实例。
  """
  file_name = 'indexed_train.jsonl' if train else 'indexed_test.jsonl'

  # 创建数据集实例
  dataset = ReviewAnalyzeDataset(config.PROCESSED_DATA_DIR / file_name)

  # 返回 DataLoader
  return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
  # 简单测试数据加载器
  dataloader = get_dataloader()
  for input_tensor, target_tensor in dataloader:
    print(input_tensor.shape, target_tensor.shape)
    break