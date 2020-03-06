import torch
import torch.nn as nn
import torch.optim as optim

class NameFormatModel(nn.Module):
    def __init__(self, device: str, input_sz: int, hidden_sz: int, output_sz: int, num_layers: int = 4, drop_out: float = 0.1):
        super(NameFormatModel, self).__init__()
        self.device = device
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_sz, self.hidden_sz, num_layers=num_layers, bidirectional=True)
        self.fc1 = nn.Linear(self.hidden_sz * (4 * self.num_layers), output_sz) # 2 * num layers * hidden * 2 hs tensors 
        self.dropout = nn.Dropout(drop_out)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input: torch.Tensor, hidden: torch.Tensor = None):
        batch_sz = input.shape[1]
        seq_len = input.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch_sz)
        
        for i in range(seq_len):
            lstm_input = input[i]
            lstm_probs, hidden = self.lstm(lstm_input.unsqueeze(0), hidden)
        
        # hidden is a tuple of 2 hidden tensors that are a forward and backward tensor in one
        hidden = torch.cat((hidden[0], hidden[1]), 2)
        hidden_tuple = tuple([hs for hs in hidden])
        hidden = torch.cat(hidden_tuple, 1)

        output = self.fc1(hidden)
        output = self.dropout(output)
        output = self.softmax(output)

        return output

    def init_hidden(self, batch_sz: int):
        return (torch.zeros(2 * self.num_layers, batch_sz, self.hidden_sz).to(self.device),
                torch.zeros(2 * self.num_layers, batch_sz, self.hidden_sz).to(self.device))