import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset.ner_data import Data
from dataset.ner_dataset import CustomDataset
from model.mrc_model import MRC_NER_Model, MRC_NER_Network
from utils.config import Device, seed_every_where
import sys


if __name__ == '__main__':
    model_name = 'MRC-NER-Model-metric-0.99584.pth'

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # tokenizer = BertTokenizer.from_pretrained("../../../pretrained_model/chinese-bert-wwm-ext")
    # tokenizer = AutoTokenizer.from_pretrained("../../../pretrained_model/chinese-electra-180g-large-discriminator")
    saved_network = MRC_NER_Network()

    saved_network.load_state_dict(torch.load('./saved_model/{}'.format(model_name)))
    model = MRC_NER_Model(saved_network, tokenizer, Device)

    # input_string = '江苏省淮安市'
    # model.predict_single(input_string)

    input_string = input("input sequence: ")
    while input_string:
        re = model.predict_single(input_string)
        print('省级行政区：', re, end='\n\n')
        input_string = input("input sequence: ")

    print('Predict Done')
