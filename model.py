import math
import yaml
import torch
import torch.nn as nn

from torch.nn import functional as F
from yaml.loader import FullLoader
from mixer import MLPMixer
from typing import List

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.size() == target.size()
        assert len(pred.size()) == 2
        target_sum = torch.sum(target, dim=1)
        target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
        target = target/target_div
        logsoftmax = nn.LogSoftmax(dim=1).to(pred.device)
        output = torch.sum(-target * logsoftmax(pred), 1)
        return torch.mean(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2).contiguous()
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        positions = torch.arange(x.size(0)).to(self.embeddings.weight.device)
        pe = self.embeddings(positions).unsqueeze(1).to(x.device)
        x = x + pe
        return x

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.first = nn.Sequential(
            nn.Conv1d(self.config['feature_dim'], self.config['classifier_transformer_unit'], kernel_size=5, padding=2),
            NewGELU(),
            nn.Conv1d(self.config['classifier_transformer_unit'], self.config['classifier_transformer_unit'], kernel_size=5, padding=2),
            NewGELU(),
        )
        self.class_token = nn.Embedding(num_embeddings=1, embedding_dim=self.config['classifier_transformer_unit'])
        self.layernorm = nn.LayerNorm(self.config['classifier_transformer_unit'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['classifier_transformer_unit'], nhead=8, dim_feedforward=config['classifier_transformer_fc_unit'], activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=config['classifier_transformer_layer_num'])
        self.mlp_head = nn.Sequential(
            nn.Linear(self.config['classifier_transformer_unit'], self.config['classifier_fc_unit']),
            NewGELU(),
            nn.Linear(self.config['classifier_fc_unit'], self.config['class_num'])
        )
        self.to(config['device'])

    def forward(self, x, mask):
        #IN: [B, L, E], [B, L]
        #for transformer, [L,B,E]
        #for conv, [B, L, E]->[B,E,L]
        x = x.permute(0,2,1)
        x = self.first(x)
        #[B,E,L] -> [B,L,E]
        x = x.permute(0,2,1)
        x = torch.nn.functional.normalize(x, dim=-1)
        #[B,L,E] -> [L,B,E]
        x = x.permute(1,0,2)
        x = self.encoder(x, src_key_padding_mask=mask)
        
        #[L,B,E] -> [B,L,E]
        x = x.permute(1,0,2)
        x[mask] = torch.zeros_like(x[0,0])
        valid = torch.logical_not(mask)
        valid_length = torch.sum(valid, dim=1).unsqueeze(1)
        x = torch.sum(x, dim=1)/(valid_length+1e-8)
        
        x = self.mlp_head(x)
        return x




class CustomCell(nn.Module):
    def __init__(self, config):
        super(CustomCell, self).__init__()
        self.x_layernorm = nn.LayerNorm(config['n_recurrent_hidden'])
        self.h_layernorm = nn.LayerNorm(config['n_recurrent_hidden'])
        self.gru = nn.GRUCell(config['n_recurrent_hidden'], config['n_recurrent_hidden'])
        self.dropout = nn.Dropout(config['recurrent_pdrop'])
        self.linear = nn.Linear(config['n_recurrent_hidden'], config['n_recurrent_hidden'])
        self.processed_layernorm = nn.LayerNorm(config['n_recurrent_hidden'])
        self.act = nn.GELU()

    def forward(self, x, h):
        '''
        IN: x: [B, E], h: [B, E]
        OUT: y: [B, E], next_h: [B, E]
        '''
        normalized_x = self.x_layernorm(x)
        normalized_h = self.h_layernorm(h)
        next_h = self.gru(normalized_x, normalized_h)
        processed = self.dropout(next_h)
        y = processed
        return y, next_h
    


class OADModel(nn.Module):
    def __init__(self, config):
        super(OADModel, self).__init__()
        self.config = config
        
        self.preprocess = nn.Sequential(
            nn.Linear(config['n_feature'], config['n_recurrent_hidden']),
            nn.LayerNorm(config['n_recurrent_hidden']),
            nn.GELU(),
        )
        self.cells = nn.ModuleList([CustomCell(config) for _ in range(config['n_recurrent_layer'])])
        self.proj = nn.Linear(config["n_recurrent_hidden"], config["n_projection_hidden"])
        
        self.classifier = MLPMixer(
            token_num=config["n_recurrent_layer"], dim=config['n_projection_hidden'], 
            depth=config['n_classifier_layer'], num_classes=config['n_state'], expansion_factor=4, expansion_factor_token=4
        )  
        
        self.opt = torch.optim.AdamW(self.parameters(), lr=config['oad_base_lr'], betas=config['oad_betas'], weight_decay=config['oad_weight_decay'])
        assert config['n_state'] == len(config['cross_entropy_weight'])
        self.cross_entropy_loss = nn.CrossEntropyLoss(torch.Tensor(config['cross_entropy_weight']))
        self.focal_loss = None

    def encode(self, feature_in:torch.FloatTensor, hs:List[torch.FloatTensor]):
        x = self.preprocess(feature_in)
        n_layer = len(hs)
        next_hs = []
        projected_xs = []
        for i in range(n_layer):
            h = hs[i]
            x, next_h = self.cells[i](x, h)
            next_hs.append(next_h)
            projected_xs.append(self.proj(x))
        if self.config['use_projection']:
            x = torch.stack(projected_xs, dim=1)
        score = self.classifier(x)
        return next_hs, score

    def forward(self, feature_in:torch.FloatTensor, target_a_1:torch.LongTensor, target_a_2:torch.LongTensor):
        if target_a_1 is not None:
            self.train()
        batch_size = feature_in.size(0)
        feature_len = feature_in.size(1)
        device = feature_in.device
        hs = [torch.zeros(batch_size, self.config['n_recurrent_hidden']).to(device) for _ in range(self.config['n_recurrent_layer'])]
        score_stack = []

        for step in range(feature_len):
            hs, score = self.encode(
                feature_in[:, step], hs
            )
            score_stack.append(score)
        logits = torch.stack(score_stack, dim=1)
        # if we are given some desired targets, calculate the loss
        loss = None
        if target_a_1 is not None:
            if 'focal' in self.config['prefix']:
                raise NotImplementedError()
            else:
                loss_1 = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), target_a_1.view(-1))
                loss_2 = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), target_a_2.view(-1))
                loss = torch.min(loss_1, loss_2)
        return logits, loss