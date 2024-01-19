"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn



class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_dict=1500, num_pos_feats=256):
        super().__init__()
        # print('num_pos_dict: ', num_pos_dict)  # num_pos_dict:  512
        self.row_embed = nn.Embedding(num_pos_dict, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)

    def forward(self, x, mask):
        x_len = x.shape[-1]
        # print('x_len: ', x_len)  # x_len:  100
        # print('x.device: ', x.device, x.shape)  # cuda:0 torch.Size([2, 600, 100])
        i = torch.arange(x_len, device=x.device)
        # print('i.shape: ', i.shape)  # i.shape:  torch.Size([100])
        x_emb = self.row_embed(i)
        # print('x_emb.shape: ', x_emb.shape)  # x_emb.shape:  torch.Size([100, 256])
        pos = x_emb.unsqueeze(0).permute(0, 2, 1).repeat(x.shape[0], 1, 1)
        # print('pos.shape: ', pos.shape)  # pos.shape:  torch.Size([30, 256, 100])
        return pos
'''

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_dict=1500, num_pos_feats=256):
        super().__init__()
        # print('num_pos_dict: ', num_pos_dict)  # num_pos_dict:  512
        self.row_embed = nn.Embedding(num_pos_dict, 100)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)

    def forward(self, x, mask):
        # x_len = x.shape[-1]
        # print('x_len: ', x_len)  # x_len:  100
        # print('x.device: ', x.device, x.shape)  # cuda:0 torch.Size([2, 600, 100])
        i = torch.arange(256, device=x.device)
        # print('i.shape: ', i.shape)  # i.shape:  torch.Size([100])
        x_emb = self.row_embed(i)
        # print('x_emb.shape: ', x_emb.shape)  # x_emb.shape:  torch.Size([100, 256])
        pos = x_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # print('pos.shape: ', pos.shape)  # pos.shape:  torch.Size([30, 256, 100])
        return pos
'''

class PositionEmbeddingFixed(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, num_pos_dict, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbeddingFixed, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x, mask):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x.transpose(0, 1).unsqueeze(-1)  # if this is feature, simply x = x.transpose(0, 1), the shape of x will be torch.Size([100, 30, 256])
        # print('x.shape: ', x.shape)
        # print('0 x.shape: ', x.shape, self.pe[:x.size(0), :].repeat(1, x.shape[1], 1).shape)  # 0 x.shape:  torch.Size([100, 30, 1]) torch.Size([100, 30, 256])  # position.shape:  torch.Size([100, 30, 1])
        position = x  # torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (100, 30)
        pe = torch.zeros(x.shape[0], x.shape[1], self.d_model).to(position.device)  # (100, 30, 256)
        # print('position.shape: ', position.shape)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(position.device)
        print('div_term.shape: ', div_term.shape)
        # print('(position * div_term).shape: ', (position * div_term).shape)  # (position * div_term).shape:  torch.Size([100, 30, 128])
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)

        # x = x + self.pe[:x.size(0), :].repeat(1, x.shape[1], 1)
        x = pe
        # print('1 x.shape: ', x.shape)  # 1 x.shape:  torch.Size([100, 30, 256])
        x = x.permute(1, 2, 0)
        # print('2 x.shape: ', x.shape)  # 2 x.shape:  torch.Size([30, 256, 100])
        return self.dropout(x)

class PositionEmbeddingFixed_v0(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, num_pos_dict, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbeddingFixed, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, mask):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # print('x.shape, self.pe.shape: ', x.shape, self.pe[:x.size(0), :].shape)  # x.shape, self.pe.shape:  torch.Size([30, 100]) torch.Size([30, 1, 256])
        x = x.transpose(0, 1).unsqueeze(-1)  # if this is feature, simply x = x.transpose(0, 1), the shape of x will be torch.Size([100, 30, 256])
        # print('0 x.shape: ', x.shape, self.pe[:x.size(0), :].repeat(1, x.shape[1], 1).shape)  # 0 x.shape:  torch.Size([100, 30, 1]) torch.Size([100, 30, 256])
        x = x + self.pe[:x.size(0), :].repeat(1, x.shape[1], 1)
        # print('1 x.shape: ', x.shape)  # 1 x.shape:  torch.Size([100, 30, 256])
        x = x.permute(1, 2, 0)
        # print('2 x.shape: ', x.shape)  # 2 x.shape:  torch.Size([30, 256, 100])
        return self.dropout(x)


def build_position_encoding(args):
    N_steps = args.hidden_dim
    N_elements = args.num_pos_embed_dict
    if args.position_embedding in ('learned'):
        position_embedding = PositionEmbeddingLearned(N_elements, N_steps)
    elif args.position_embedding in ['fixed']:
        position_embedding = PositionEmbeddingFixed(N_elements, N_steps)
        # position_embedding = PositionEmbeddingLearned(N_elements, N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
