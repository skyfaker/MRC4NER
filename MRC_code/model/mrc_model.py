import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
# from transformers import get_scheduler
from itertools import repeat
from transformers import BertModel

from model.base_model import BaseModel, FGM
from utils.config import tok
from utils.util import WarmUp_LinearDecay


class Spatial_Dropout(nn.Module):
    def __init__(self, drop_prob):

        super(Spatial_Dropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self, inputs):
        return inputs.new().resize_(inputs.size(0), *repeat(1, inputs.dim() - 2), inputs.size(2))


class biaffine(nn.Module):
    def __init__(self, in_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), in_size + int(bias_y)))
        # U.shape = [in_size,out_size,in_size]

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ij,byj->bxy', x, self.U, y)
        return bilinar_mapping


class MRC_NER_Network(nn.Module):
    def __init__(self, in_size=256):
        super(MRC_NER_Network, self).__init__()
        self.in_size = in_size
        # self.base_model = BertModel.from_pretrained("bert-base-chinese")
        self.base_model = BertModel.from_pretrained("../../../pretrained_model/chinese-bert-wwm-ext")
        # self.base_model = BertModel.from_pretrained("../../../pretrained_model/bert-train")  # 可以换成自己的预训练模型
        self.dropout_bert = Spatial_Dropout(drop_prob=0.25)

        self.query_layer = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=in_size),
                                               torch.nn.ReLU())
        self.context_layer = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=in_size),
                                                 torch.nn.ReLU())
        self.biaffine_layer = biaffine(in_size, bias_x=True, bias_y=True)

    def forward(self, inputs, mask, context_len):
        # bert
        bert_output = self.base_model(input_ids=inputs, attention_mask=mask)
        sequence_output = bert_output.last_hidden_state
        # x = self.dropout_bert(sequence_output)

        # query_embedding = sequence_output[:, -1 - context_len, :].unsqueeze(1).expand(-1, context_len, -1)
        context_embedding = sequence_output[:, -context_len:]

        query_embedding = self.query_layer(context_embedding)  # (batch_size, seq_length, 256)
        context_embedding = self.context_layer(context_embedding)  # (batch_size, seq_length, 256)

        # attention = torch.matmul(context_embedding * (self.in_size ** -0.5), context_embedding.permute(0, 2, 1))
        # return attention
        span_logits = self.biaffine_layer(query_embedding, context_embedding)
        span_logits = span_logits.contiguous()

        return span_logits


class MRC_NER_Model(BaseModel):
    def __init__(self, network,
                 tokenizer=None,
                 device='cpu',
                 train_dataloader=None,
                 dev_dataloader=None,
                 model_name='MRC-NER-Model'):
        super().__init__(network)
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = None
        self.network = network
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.network.to(self.device)

    def custom_collate(self, batch, train=False):
        batch_max_length = max([len(seq) for seq in batch])
        batch_input_ids, batch_attention_mask = [], []
        batch_label, batch_label_mask = [], []

        # query_list = ['中国有省、自治区、直辖市以及特别行政区4种省级行政单位，其中4个直辖市：北京市、上海市、天津市、重庆市。抽取下面句子中的省级行政区。',
        #               '省级行政区是中国最高级别的行政区划单位。包括23个省、4个直辖市、5个自治区和2个特别行政区。抽取下面句子中的省级行政区。',
        #               '中华人民共和国省级行政区包括23个省、5个自治区、4个直辖市、2个特别行政区，一共34个。抽取下面句子中的省级行政区。',
        #               '中华人民共和国的行政区划由省级行政区、地级行政区、县级行政区、乡级行政区组成。抽取下面句子中的省级行政区。']
        # query = random.choice(query_list)
        query = '省级行政区是中国最高级别的行政区划单位。包括23个省、4个直辖市、5个自治区和2个特别行政区。抽取下面句子中的省级行政区。'
        query_token = self.tokenizer.encode(query)
        query_mask = torch.ones((len(batch), len(query_token))).to(self.device)
        for sequence in batch:
            seq_length = len(sequence)
            pad_length = batch_max_length - seq_length

            seq_token = self.tokenizer.convert_tokens_to_ids(sequence.text_list)
            seq_token.extend([0] * pad_length)
            batch_input_ids.append(query_token + seq_token)

            attention_mask = [0 for _ in range(batch_max_length)]
            attention_mask[:seq_length] = [1] * seq_length
            batch_attention_mask.append(attention_mask)

            label_mask = torch.zeros((batch_max_length, batch_max_length))
            label_mask[:seq_length, :seq_length] = 1
            label_mask = torch.triu(label_mask)
            batch_label_mask.append(label_mask)

            if train:
                # label_list转成(seq_length, seq_length, 1)的tensor
                re = []
                start, end, label_type = -1, -1, 1  # label_type= 1 表示省级行政区
                for index, label in enumerate(sequence.gt_label_index_list):
                    if label == label_type:
                        end = index
                    else:
                        if start != end:
                            re.append((start + 1, end))
                        start, end = index, index

                if start != end:
                    re.append((start + 1, end))
                # print(re)

                label_tensor = torch.zeros((batch_max_length, batch_max_length), dtype=torch.long)
                for r in re:
                    label_tensor[r[0], r[1]] = 1
                batch_label.append(label_tensor)

        inputs_ids_tensor = torch.LongTensor(batch_input_ids).to(self.device)
        attention_mask = torch.LongTensor(batch_attention_mask).to(self.device)
        label_mask_tensor = torch.stack(batch_label_mask).to(self.device)

        if train:
            label_tensor = torch.stack(batch_label).to(self.device)
            return inputs_ids_tensor, torch.concat([query_mask, attention_mask], dim=-1), label_tensor, \
                label_mask_tensor, batch_max_length
        else:
            return inputs_ids_tensor, torch.concat([query_mask, attention_mask], dim=-1), label_mask_tensor, \
                batch_max_length

    def train(self, epoch_num=10, lr=1e-4):
        print("Training")

        if self.train_dataloader:
            with open('./model/{}_label_map.json'.format(self.model_name), 'w') as f:
                json.dump(self.train_dataloader.dataset.label_map, f)
            print("label_map saved to ./model/{}_label_map.json".format(self.model_name))

        base_params_id = list(map(id, self.network.base_model.parameters()))  # 返回的是parameters的 内存地址
        classifier_params = filter(lambda p: id(p) not in base_params_id, self.network.parameters())
        optimizer_grouped_parameters = [
            {'params': classifier_params, 'lr': lr},
            {'params': self.network.base_model.parameters(), 'lr': lr * 0.05},
        ]
        optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=lr)
        scheduler = WarmUp_LinearDecay(
            optimizer=optimizer,
            init_rate=lr,
            train_data_length=len(self.train_dataloader),
            warm_up_epoch=2,
            decay_epoch=3,
            epoch=epoch_num
        )

        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.BCELoss()
        best_F1 = 0.0
        fgm = FGM(self.network)

        for epoch in range(epoch_num):
            running_loss, running_bg_loss, running_class_loss = 0.0, 0.0, 0.0
            for i, sequence_list in enumerate(self.train_dataloader, 1):
                self.network.train()
                network_input, attn_mask, gt_label, label_mask, context_len = self.custom_collate(sequence_list,
                                                                                                  train=True)

                fgm.attack()  # lstm embedding被修改了

                output = self.network(network_input, attn_mask, context_len)
                assert output.shape == label_mask.shape, "output, label_msk不一致"
                active_loss = label_mask.view(-1) == 1
                active_logits = output.view(-1)[active_loss]
                active_labels = gt_label.view(-1)[active_loss]

                soft_logits = torch.sigmoid(active_logits)
                loss = loss_fn(soft_logits, active_labels.float())

                fgm.restore()  # 恢复Embedding的参数

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.25)
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item()

                batch_step = 10
                if i % batch_step == 0 or i == len(self.train_dataloader):
                    print('[epoch: %d/%d, batch: %5d] lr: %f loss: %.3f' %
                          (epoch + 1, epoch_num, i, optimizer.state_dict()['param_groups'][0]['lr'],
                           running_loss / batch_step))
                    running_loss = 0.0

            precision, recall, F1 = self.evaluate()
            print('For epoch', epoch + 1, 'Precision: {}, Recall: {}, F1: {}'.format(precision, recall, F1))
            # save the best model
            if F1 > best_F1:
                self.save_model(metric=F1)
                best_F1 = F1

        precision, recall, F1 = self.evaluate()
        print('Final Precision: {}, Recall: {}, F1: {}'.format(precision, recall, F1))

    def evaluate(self):
        print('Validation')

        assert self.dev_dataloader is not None, 'dev_dataloader should not be None'
        self.network.eval()

        id2label = self.dev_dataloader.dataset.index2label_map
        true_labels, true_predictions = [], []

        T, P, G = 0, 0, 0
        with torch.no_grad():
            for sequence_list in self.dev_dataloader:
                network_input, attn_mask, gt_label, label_mask, context_len \
                    = self.custom_collate(sequence_list, train=True)
                output = self.network(network_input, attn_mask, context_len)
                output = output * label_mask

                predictions = torch.sigmoid(output)

                binary_prediction = torch.where(predictions > 0.5,
                                                torch.ones(predictions.shape).to(predictions.device),
                                                torch.zeros(predictions.shape).to(predictions.device))
                zero_tensor = torch.zeros(binary_prediction.shape).to(binary_prediction.device)
                b_t = ((binary_prediction == gt_label) * (binary_prediction != zero_tensor)).sum()
                b_p = torch.norm(binary_prediction.float(), 0)
                b_g = torch.norm(gt_label.float(), 0)
                T += b_t
                P += b_p
                G += b_g

                predictions = predictions.cpu().numpy().tolist()
                labels = gt_label.cpu().numpy().tolist()
                true_labels += self.convert_label_matrix_to_label(labels, id2label)
                true_predictions += self.convert_label_matrix_to_label(predictions, id2label)
                # true_predictions += self.convert_matrix_to_label(output, id2label)
        precision = T / (P + 1e-8)
        recall = T / (G + 1e-8)
        F1 = 2 * precision * recall / (precision + recall)
        # print('background precession: {}, recall: {}, F1: {}'.format(precession, recall, F1))
        return float(precision), float(recall), float(F1)

    def convert_label_matrix_to_label(self, matrix, id2label):
        # matrix: (batch, seq_length, seq_length)
        all_re = []
        for batch_m in matrix:
            tmp_col = -1
            re = ['O'] * len(batch_m)
            for row_index, row in enumerate(batch_m):
                for col_index, label in enumerate(row[tmp_col + 1:], tmp_col + 1):
                    if label != 0:
                        label_string = id2label[int(label)]
                        if label_string == 'O':
                            re[row_index] = 'O'
                        elif row_index == col_index:
                            re[row_index] = 'S-' + label_string
                        else:
                            re[row_index] = 'B-' + label_string
                            re[col_index] = 'E-' + label_string
                            re[row_index + 1: col_index] = ["I-" + label_string] * (col_index - 1 - row_index)
                        tmp_col = col_index
                        break
            all_re.append(re)
        return all_re

    def convert_matrix_to_label(self, matrix, id2label):
        # matrix: (batch, seq_length, seq_length, number_labels)
        all_re = []
        for batch_m in matrix:
            re_list = []
            max_value, max_index = batch_m.max(-1)
            row_length = len(batch_m)
            bg_mask = torch.where(max_index > 0, torch.ones(max_index.shape).to(max_index.device), max_index)
            max_value *= bg_mask

            for row_index, row in enumerate(max_value):
                col_index = row.argmax(-1)
                label_index = int(max_index[row_index, col_index])
                if label_index != 0:
                    re_list.append([row_index, int(col_index), label_index])

            merge_r_list = [re_list.pop(0)] if re_list else []
            while re_list:
                p_r = merge_r_list.pop()
                r_r = re_list.pop(0)
                if p_r[1] == r_r[0] and p_r[-1] == r_r[-1]:
                    merge_r_list.append([p_r[0], r_r[1], r_r[-1]])
                else:
                    if r_r[1] < p_r[1]:
                        merge_r_list.append(p_r)
                    else:
                        merge_r_list.append(p_r)
                        merge_r_list.append(r_r)

            result = ['O'] * len(batch_m)
            for r in merge_r_list:
                label_str = id2label[r[2]]
                if label_str == 'O':
                    continue
                if r[0] == r[1]:
                    result[r[0]] = 'S-' + label_str
                else:
                    result[r[0]] = 'B-' + label_str
                    result[r[0] + 1: r[1]] = ['I-' + label_str] * (r[1] - 1 - r[0])
                    result[r[1]] = 'E-' + label_str

            all_re.append(result)
        return all_re

    def predict(self, predict_data=None, index2label=None):
        print('Predict')
        id2label = index2label if index2label is not None else predict_data.dataset.index2label_map
        self.network.eval()
        with torch.no_grad():
            total_length = len(predict_data)
            for batch_index, sequence_list in enumerate(predict_data):
                print("\r{}/{}".format(batch_index + 1, total_length), end="")
                network_input, attn_mask, label_mask, word_pos = self.custom_collate(sequence_list)

                # run the model on the test set to predict labels
                output = self.network(network_input, attn_mask, word_pos)
                label_mask = label_mask.unsqueeze(-1).repeat(1, 1, 1, output.shape[-1])
                output = output * label_mask

                predictions = output.argmax(dim=-1).cpu().tolist()
                # the label with the highest energy will be our prediction
                true_predictions = self.convert_label_matrix_to_label(predictions, id2label)
                for seq, seq_pred in zip(sequence_list, true_predictions):
                    seq_length = len(seq)
                    seq_pred = seq_pred[:seq_length]
                    seq.pred_label_list = seq_pred

    def convert_predictions(self, output):
        re_list = []
        max_value, max_index = output.max(-1)
        row_length = len(output)
        max_value = torch.triu(max_value)

        while torch.gt(max_value, 0).any():
            current_max = max_value.argmax()  # 最大值的索引
            row = int(current_max / row_length)
            col = int(current_max % row_length)
            re_list.append((row, col, int(max_index[row, col])))
            max_value[row:col + 1] = 0
            max_value[:, row:col + 1] = 0
            max_value[:row, col + 1:] = 0
        result = [0] * 64
        for r in re_list:
            if r[0] == r[1]:
                result[r[0]] = 'S-' + str(r[2])
            else:
                label = str(r[2])
                result[r[0]] = 'B-' + label
                result[r[0] + 1: r[1]] = ['I-' + label] * (r[1] - 1 - r[0])
                result[r[1]] = 'E-' + label

        return result

    def testF1(self, outside_data=None, **kwargs):
        dev_dataloader = self.dev_dataloader if outside_data is None else outside_data
        self.network.eval()
        with torch.no_grad():
            P, G, T = 0, 0, 0
            for batch_index, sequence_list in enumerate(dev_dataloader):
                network_input, gt_label, mask, seq_length = self.custom_biaffine_collate(sequence_list)
                device = network_input.device
                background, output = self.network(network_input, mask)
                # output = torch.nn.functional.softmax(output, dim=-1)

                background = torch.nn.functional.sigmoid(background)
                foreground = torch.where(background > 0.5, background, torch.zeros(background.shape).to(device))
                foreground_max, foreground_col = torch.max(background, dim=-1)
                max_mask = foreground_max.unsqueeze(-1).repeat(1, 1, 64)
                foreground_mask = foreground / max_mask
                foreground_pos = torch.where(foreground_mask == 1, foreground_mask,
                                             torch.zeros(background.shape).to(device))

                # foreground = torch.where(background > 0.5, torch.ones(background.shape).to(device),
                #                          torch.zeros(background.shape).to(device))
                background_gt_label = torch.where(gt_label > 0, torch.ones(gt_label.shape).to(gt_label.device),
                                                  torch.zeros(gt_label.shape).to(gt_label.device))
                # background_gt_label_max = torch.argmax(background_gt_label, dim=-1)

                zero_tensor = torch.zeros(foreground_pos.shape).to(foreground_pos.device)
                b_t = ((foreground_pos == background_gt_label) * (foreground_pos != zero_tensor)).sum()
                b_p = torch.norm(foreground_pos.float(), 0)
                b_g = torch.norm(background_gt_label.float(), 0)

                T += int(b_t)
                P += int(b_p)
                G += int(b_g)

        print("P: {}, G: {}, T:{} ".format(P, G, T))
        precision = T / (P + 1e-8)
        recall = T / (G + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, F1

    def predict_single(self, string):
        self.network.eval()
        with torch.no_grad():
            seq_length = len(string)
            # query = '省级行政区是中国最高级别的行政区划单位。包括23个省、4个直辖市、5个自治区和2个特别行政区。抽取下面句子中的省级行政区。'
            # query = '县级行政区，即行政地位与县相同的行政区，包括市辖区、县级市、县、自治县、旗、自治旗、特区、林区。抽取下面句子中的县级行政区。'
            query = '省级行政区是中国最高级别的行政区划单位。抽取下面句子中的省级行政区。'
            query_token = self.tokenizer.encode(query)
            seq_token = self.tokenizer.convert_tokens_to_ids(list(string))
            total_token = query_token + seq_token
            input_tensor = torch.LongTensor(total_token).to(self.device).view(1, -1)
            attention_mask = torch.ones((1, len(total_token))).to(self.device)
            label_mask = torch.ones((seq_length, seq_length))
            label_mask = torch.triu(label_mask).to(self.device)

            # run the model on the test set to predict labels
            output = self.network(input_tensor, attention_mask, seq_length)
            output = output.squeeze() * label_mask
            prediction = torch.sigmoid(output)

            re = []
            for row in range(seq_length):
                col, value = prediction[row].argmax(), prediction[row].max()
                if value > 0.5:
                    re.append(string[row: col+1])
            return re
