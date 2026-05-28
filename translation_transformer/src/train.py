#m模型训练
import time
import torch
from torch.nn import CrossEntropyLoss #交叉熵损失
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  #进度条


from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel


def train_one_epoch(dataloader, model, optimizer, device):
    model.train()
    total_loss=0  #
    for src,tgt in tqdm(dataloader,desc='训练'):
        src = src.to(device)
        tgt = tgt.to(device)