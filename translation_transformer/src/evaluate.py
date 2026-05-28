import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationEncoder, TranslationDecoder
from dataset import get_dataloader
from predict import predict_batch

def evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device):
  """
  执行模型评估，计算 BLEU 分数。

  :param dataloader: 数据加载器。
  :param encoder: 编码器。
  :param decoder: 解码器。
  :param zh_tokenizer: 中文分词器。
  :param en_tokenizer: 英文分词器。
  :param device: 设备。
  :return: BLEU 分数。
  """
  all_references = []
  all_predictions = []
  special_tokens = [
    zh_tokenizer.pad_token_index,
    zh_tokenizer.eos_token_index,
    zh_tokenizer.sos_token_index
  ]

  for src, tgt in tqdm(dataloader, desc="评估"):
    src = src.to(device)
    tgt = tgt.tolist()

    predict_indexes = predict_batch(src, encoder, decoder, en_tokenizer, device)
    all_predictions.extend(predict_indexes)

    for indexes in tgt:
      indexes = [index for index in indexes if index not in special_tokens]
      all_references.append([indexes])

  bleu = corpus_bleu(all_references, all_predictions)
  return bleu

def run_evaluate():
  """
  启动评估流程。
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
  en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

  encoder = TranslationEncoder(zh_tokenizer.vocab_size, zh_tokenizer.pad_token_index).to(device)
  encoder.load_state_dict(torch.load(config.MODELS_DIR / 'encoder.pt'))

  decoder = TranslationDecoder(en_tokenizer.vocab_size, en_tokenizer.pad_token_index).to(device)
  decoder.load_state_dict(torch.load(config.MODELS_DIR / 'decoder.pt'))

  dataloader = get_dataloader(train=False)

  bleu = evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device)
  print('========== 评估结果 ==========')
  print(f'BLEU: {bleu:.2f}')
  print('=============================')

if __name__ == '__main__':
  run_evaluate()