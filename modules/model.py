# Models

from turtle import forward
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index) -> None:
        super(LSTM, self).__init__()

        self.bidirectional = bidirectional
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout_rate, 
                            batch_first=True)
        self.drop = nn.Dropout(p=dropout_rate)

        self.fc = nn.Linear(2*hidden_dim if bidirectional else hidden_dim, output_dim)

    def forward(self, x, x_len):
        embedded = self.drop(self.embeddings(x))
        packed_out = nn.utils.rnn.pack_padded_sequence(embedded, x_len, batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed_out)

        if self.bidirectional:
            hidden = self.drop(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.drop(hidden[-1])

        prediction = self.fc(hidden)
        return prediction



if __name__ == "__main__":
    # unit test
    model = LSTM(1000, 128)
    print(model)