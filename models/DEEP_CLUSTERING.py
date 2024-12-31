import torch as torch
import torch.nn.functional as F 

class DEEP_CLUSTERING(torch.nn.Module):
    def __init__(self, input_length_t, input_length_f, n_hidden_layers, hidden_dim, emb_dim):
        super(DEEP_CLUSTERING, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.n_hidden_layers = n_hidden_layers
        self.input_length_t = input_length_t
        self.input_length_f = input_length_f
        self.bi_lstm = torch.nn.LSTM(input_length_t, hidden_dim, n_hidden_layers, batch_first=True, bidirectional= True)
        self.dense = torch.nn.Linear(2*hidden_dim, input_length_f* emb_dim)
        self.activation = torch.tanh
        self.bn = torch.nn.BatchNorm1d(hidden_dim*2)


    def forward(self, x):
        x = x.permute(0,2,1)
        self.bi_lstm.flatten_parameters()
        x, _ = self.bi_lstm(x)
        x = self.bn(x.permute(0,2,1)).permute(0,2,1)
        x = self.dense(x)
        x = self.activation(x)
        x = x.view(-1, self.input_length_t*self.input_length_f, self.emb_dim)
        x = F.normalize(x, 2, dim=-1)
        x = x.view(-1, self.input_length_t, self.input_length_f, self.emb_dim)
        return x 


