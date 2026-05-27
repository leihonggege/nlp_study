# predict.py

import torch
import config

from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel

def predict_batch(input_tensor, model):
  """
  对一个 batch 的输入进行预测。

  :param input_tensor: 输入张量，形状 (batch_size, seq_len)。
  :param model: 模型。
  :return: 概率列表。
  """
  model.eval()
  with torch.no_grad():
    # 前向传播获取 logits
    output = model(input_tensor)

    # 使用 sigmoid 将 logits 转换为概率
    probs = torch.sigmoid(output)

  return probs.tolist()

def predict(user_input, model, tokenizer, device):
  """
  对单条用户输入进行预测。

  :param user_input: 用户输入文本。
  :param model: 模型。
  :param tokenizer: 分词器。
  :param device: 设备。
  :return: 概率值。
  """
  # 编码并填充输入文本
  input_ids = tokenizer.encode(user_input, config.SEQ_LEN)

  # 转换为张量并移动到设备
  input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

  # 获取预测概率
  probs = predict_batch(input_tensor, model)
  prob = probs[0]

  return prob

def run_predict():
  """
  启动预测交互程序。
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # 加载 tokenizer
  tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

  # 创建并加载模型
  model = ReviewAnalyzeModel(
    vocab_size=tokenizer.vocab_size,
    padding_idx=tokenizer.pad_token_index
  ).to(device)
  model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

  print('请输入要预测的评论：（输入 q 或 quit 退出）')

  while True:
    user_input = input('> ')
    if user_input in ['q', 'quit']:
      print('退出程序')
      break

    if not user_input:
      print('输入为空，请重新输入')
      continue

    # 预测结果
    prob = predict(user_input, model, tokenizer, device)
    if prob > 0.5:
      print(f'正面评价（置信度：{prob:.2f}）')
    else:
      print(f'负面评价（置信度：{1 - prob:.2f}）')

if __name__ == '__main__':
  run_predict()