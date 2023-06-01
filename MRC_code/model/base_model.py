import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel:
    def __init__(self, network):
        self.model_name = 'base_model'
        self.network = network

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_model(self, **kwargs):
        saved_model_dir = './saved_model/'
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
        model_name = self.model_name
        if "metric" in kwargs:
            model_name = model_name + "-metric-" + str(round(kwargs["metric"], 5)) + '.pth'
        path = saved_model_dir + model_name
        torch.save(self.network.state_dict(), path)
        print("model saved to {}".format(path))


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='biaffine_layer'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='biaffine_layer'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class MultiHeadAttention:
    """
    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
        weight_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr` .
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False
                 ):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but received {}".format(embed_dim))
        assert num_heads > 0, ("Expected num_heads to be greater than 0, "
                               "but received {}".format(num_heads))

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _prepare_qkv(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = torch.reshape(q, [0, 0, self.num_heads, self.head_dim])
        q = q.permute(0, 2, 1, 3)
        k = torch.reshape(k, [0, 0, self.num_heads, self.head_dim])
        k = k.permute(0, 2, 1, 3)
        v = torch.reshape(v, [0, 0, self.num_heads, self.head_dim])
        v = v.permute(0, 2, 1, 3)

        return q, k, v

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        product = torch.matmul(q * (self.head_dim ** -0.5), k)
        product = product * attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(weights, self.dropout)

        out = torch.matmul(weights, v)

        # combine heads
        out = out.permute([0, 2, 1, 3])
        out = torch.reshape(out, [0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)
